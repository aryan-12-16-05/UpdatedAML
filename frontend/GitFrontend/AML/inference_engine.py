"""
Inference Engine for GAT AML Detection — Stage 1
Handles:
  1. Checkpoint loading (model + serialized feature engine)
  2. Graph construction from strict causal history (Timestamp < T)
  3. Feature engineering exactly matching 13 node / 20 edge contract
  4. Full inference pipeline -> AML probability
"""

import os
import sys
import warnings
import numpy as np
import pandas as pd
from datetime import datetime
import torch

_AML_DIR = os.path.dirname(os.path.abspath(__file__))
_MODELS_DIR = os.path.join(_AML_DIR, "models")
sys.path.insert(0, _AML_DIR)

MODEL_PATH = os.path.join(_MODELS_DIR, "GAT_model.pt")
FRAUD_THRESHOLD = 0.5

# Ensure AMLFeatureEngine is unpicklable from __main__
class AMLFeatureEngine:
    pass
import __main__
if not hasattr(__main__, 'AMLFeatureEngine'):
    __main__.AMLFeatureEngine = AMLFeatureEngine

_model = None
_feature_engine = None
_device = None

def _get_device():
    global _device
    if _device is None:
        _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return _device

def load_checkpoint():
    """Load model and feature engine from the GAT_model.pt checkpoint."""
    global _model, _feature_engine
    if _model is not None and _feature_engine is not None:
        return _model, _feature_engine

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")

    device = _get_device()
    print(f"[InferenceEngine] Loading Stage-1 GAT checkpoint from {MODEL_PATH} on {device}")
    
    ckpt = torch.load(MODEL_PATH, map_location=device, weights_only=False)
    
    from models.gat_model_def import GATv2AMLModel
    _model = GATv2AMLModel().to(device)
    
    if 'model_state' in ckpt:
        _model.load_state_dict(ckpt['model_state'])
    elif 'state_dict' in ckpt:
        _model.load_state_dict(ckpt['state_dict'])
    else:
        raise ValueError("Could not find 'model_state' in checkpoint")
        
    _model.eval()

    if 'feature_engine' not in ckpt:
        raise ValueError("Could not find 'feature_engine' in checkpoint")
        
    _feature_engine = ckpt['feature_engine']

    print("Loaded Stage-1 GAT model")
    print("Node features: 13")
    print("Edge features: 20")
    print("Hidden: 64")
    print("Heads: 4")
    print(f"Device: {device}")
    
_db_engine = None

def _get_node_features(account_id: str, target_ts: str, fe) -> np.ndarray:
    global _db_engine
    import fanout_updater
    from sqlalchemy import text
    acc = str(account_id).strip()
    
    if _db_engine is None:
        _db_engine = fanout_updater._get_engine()
        
    query = text("""
        SELECT 
            SUM(CASE WHEN from_account = :acc THEN 1 ELSE 0 END) as out_count,
            SUM(CASE WHEN to_account = :acc THEN 1 ELSE 0 END) as in_count,
            SUM(CASE WHEN from_account = :acc THEN amount_paid ELSE 0 END) as out_total,
            SUM(CASE WHEN to_account = :acc THEN amount_received ELSE 0 END) as in_total,
            MAX(CASE WHEN from_account = :acc THEN amount_paid ELSE 0 END) as out_max,
            COUNT(DISTINCT CASE WHEN from_account = :acc THEN to_account END) as unique_receivers,
            COUNT(DISTINCT CASE WHEN to_account = :acc THEN from_account END) as unique_senders
        FROM transactions
        WHERE (from_account = :acc OR to_account = :acc)
          AND timestamp < :ts
    """)
    with _db_engine.connect() as conn:
        res = conn.execute(query, {'acc': acc, 'ts': target_ts}).fetchone()
        
    out_count = float(res[0] or 0)
    in_count = float(res[1] or 0)
    out_total = float(res[2] or 0)
    in_total = float(res[3] or 0)
    out_max = float(res[4] or 0)
    unique_receivers = float(res[5] or 0)
    unique_senders = float(res[6] or 0)
    
    out_mean = out_total / out_count if out_count > 0 else 0.0
    in_mean = in_total / in_count if in_count > 0 else 0.0
    net_flow = in_total - out_total
    total_degree = out_count + in_count
    fanout_ratio = unique_receivers / in_count if in_count > 0 else unique_receivers
    pass_through_ratio = out_total / in_total if in_total > 0 else 0.0
    
    feats = [
        out_count, in_count, out_total, in_total, out_mean, in_mean, out_max,
        unique_receivers, unique_senders, net_flow, total_degree, fanout_ratio, pass_through_ratio
    ]
    
    arr = np.array(feats, dtype=np.float32).reshape(1, -1)
    if hasattr(fe, 'node_scaler') and fe.node_scaler is not None:
        arr = fe.node_scaler.transform(arr)
    return arr[0]

def _engineer_edge_features(tx: dict, fe, history_df: pd.DataFrame = None) -> np.ndarray:
    import math
    
    amt_paid_raw = float(tx.get("Amount Paid", tx.get("amount_paid", 0.0)))
    amt_recv_raw = float(tx.get("Amount Received", tx.get("amount_received", 0.0)))
    pay_cur = str(tx.get("Payment Currency", tx.get("payment_currency", "US Dollar")))
    recv_cur = str(tx.get("Receiving Currency", tx.get("receiving_currency", "US Dollar")))
    
    fx_map = fe.FX_TO_USD if hasattr(fe, 'FX_TO_USD') else {}
    pay_rate = fx_map.get(pay_cur, 1.0)
    recv_rate = fx_map.get(recv_cur, 1.0)
    
    amt_paid_usd = amt_paid_raw * pay_rate
    amt_recv_usd = amt_recv_raw * recv_rate
    
    log_amt_paid = np.log1p(amt_paid_usd)
    log_amt_recv = np.log1p(amt_recv_usd)
    amt_diff = abs(amt_paid_usd - amt_recv_usd)
    amt_ratio = amt_recv_usd / amt_paid_usd if amt_paid_usd > 0 else 1.0
    
    ts_raw = tx.get("Timestamp", tx.get("timestamp", datetime.utcnow()))
    if isinstance(ts_raw, str):
        try:
            ts = pd.to_datetime(ts_raw)
        except:
            ts = datetime.utcnow()
    else:
        ts = ts_raw
        
    hour_sin = math.sin(2 * math.pi * ts.hour / 24)
    hour_cos = math.cos(2 * math.pi * ts.hour / 24)
    day_of_week = ts.weekday()
    weekend_flag = 1.0 if day_of_week >= 5 else 0.0
    
    sender = str(tx.get("From Account", tx.get("from_account", ""))).strip()
    receiver = str(tx.get("To Account", tx.get("to_account", ""))).strip()
    self_loop = 1.0 if sender == receiver else 0.0
    
    from_bank = str(tx.get("From Bank", tx.get("from_bank", "")))
    to_bank = str(tx.get("To Bank", tx.get("to_bank", "")))
    same_bank = 1.0 if from_bank == to_bank else 0.0
    
    near_threshold = 1.0 if (9900 <= amt_paid_usd <= 10000) else 0.0
    threshold_proximity = max(0.0, 10000.0 - amt_paid_usd)
    
    pair_recency_hours = 24.0 * 30
    first_pair = 1.0
    sender_zscore = 0.0
    
    if history_df is not None and not history_df.empty:
        from_col = "From Account" if "From Account" in history_df.columns else "from_account"
        to_col = "To Account" if "To Account" in history_df.columns else "to_account"
        ts_col = "Timestamp" if "Timestamp" in history_df.columns else "timestamp"
        amt_col = "Amount Paid" if "Amount Paid" in history_df.columns else "amount_paid"
        
        pair_df = history_df[(history_df[from_col].astype(str).str.strip() == sender) & 
                             (history_df[to_col].astype(str).str.strip() == receiver)]
        if not pair_df.empty:
            first_pair = 0.0
            last_ts = pd.to_datetime(pair_df[ts_col].iloc[-1])
            delta = (ts - last_ts).total_seconds() / 3600.0
            pair_recency_hours = max(0.0, delta)
            
        sender_df = history_df[history_df[from_col].astype(str).str.strip() == sender]
        if not sender_df.empty:
            amts = sender_df[amt_col].astype(float) * pay_rate
            mean_amt = amts.mean()
            std_amt = amts.std()
            if pd.isna(std_amt) or std_amt == 0:
                std_amt = 1.0
            sender_zscore = (amt_paid_usd - mean_amt) / std_amt

    pf = str(tx.get("Payment Format", tx.get("payment_format", "")))
    pf_map = fe.payment_format_map if hasattr(fe, 'payment_format_map') else {}
    pf_idx = pf_map.get(pf, -1)
    
    pc_map = fe.payment_currency_map if hasattr(fe, 'payment_currency_map') else {}
    pc_idx = pc_map.get(pay_cur, -1)
    
    rc_map = fe.receiving_currency_map if hasattr(fe, 'receiving_currency_map') else {}
    rc_idx = rc_map.get(recv_cur, -1)
    
    feats = [
        amt_paid_usd, amt_recv_usd, log_amt_paid, log_amt_recv, amt_diff, amt_ratio,
        hour_sin, hour_cos, day_of_week, weekend_flag, self_loop, same_bank,
        near_threshold, threshold_proximity, pair_recency_hours, first_pair,
        sender_zscore, float(pf_idx), float(pc_idx), float(rc_idx)
    ]
    
    arr = np.array(feats, dtype=np.float32).reshape(1, -1)
    if hasattr(fe, 'edge_scaler') and fe.edge_scaler is not None:
        arr = fe.edge_scaler.transform(arr)
    return arr[0]

def build_sender_graph(sender_account: str, history_df: pd.DataFrame,
                       new_tx: dict, profiles_df: pd.DataFrame, fe):
    device = _get_device()
    from_col = "From Account" if "From Account" in history_df.columns else "from_account"
    to_col   = "To Account"   if "To Account"   in history_df.columns else "to_account"

    new_receiver = str(new_tx.get("To Account", new_tx.get("to_account", "UNKNOWN"))).strip()
    all_accounts = (
        list(history_df[from_col].astype(str).str.strip().unique()) +
        list(history_df[to_col].astype(str).str.strip().unique()) +
        [str(sender_account).strip(), new_receiver]
    )
    unique_accounts = list(dict.fromkeys(all_accounts))
    account_to_idx  = {acc: i for i, acc in enumerate(unique_accounts)}

    target_ts_str = pd.to_datetime(new_tx.get("Timestamp", new_tx.get("timestamp", datetime.utcnow()))).strftime("%Y-%m-%d %H:%M:%S")
    node_feats = np.stack([_get_node_features(acc, target_ts_str, fe) for acc in unique_accounts], axis=0)

    edge_src, edge_dst, edge_attrs = [], [], []

    for i in range(len(history_df)):
        row = history_df.iloc[i]
        src = str(row.get(from_col, "")).strip()
        dst = str(row.get(to_col,   "")).strip()
        if src not in account_to_idx or dst not in account_to_idx:
            continue
        edge_src.append(account_to_idx[src])
        edge_dst.append(account_to_idx[dst])
        
        # historical causal features logic
        past_df = history_df.iloc[:i]
        edge_attrs.append(_engineer_edge_features(row.to_dict(), fe, past_df))

    sender_idx   = account_to_idx[str(sender_account).strip()]
    receiver_idx = account_to_idx[new_receiver]

    new_edge_feat_raw = _engineer_edge_features(new_tx, fe, history_df)

    edge_src.append(sender_idx)
    edge_dst.append(receiver_idx)
    edge_attrs.append(new_edge_feat_raw)

    edge_attrs_np = np.stack(edge_attrs, axis=0)
    
    from torch_geometric.data import Data
    x          = torch.tensor(node_feats,  dtype=torch.float32).to(device)
    edge_index = torch.tensor([edge_src, edge_dst], dtype=torch.long).to(device)
    edge_attr  = torch.tensor(edge_attrs_np, dtype=torch.float32).to(device)

    if x.shape[1] != 13: raise ValueError(f"Node features must be 13, got {x.shape[1]}")
    if edge_attr.shape[1] != 20: raise ValueError(f"Edge features must be 20, got {edge_attr.shape[1]}")
    if not torch.isfinite(x).all(): raise ValueError("NaN/Inf in node features")
    if not torch.isfinite(edge_attr).all(): raise ValueError("NaN/Inf in edge features")

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    return data, sender_idx, receiver_idx, edge_attr[-1]

def score_transaction(new_tx: dict, history_df: pd.DataFrame,
                      profiles_df: pd.DataFrame) -> dict:
    model, fe = load_checkpoint()
    
    sender_account = str(new_tx.get("From Account", new_tx.get("from_account", ""))).strip()
    
    ts_col = "Timestamp" if "Timestamp" in history_df.columns else "timestamp"
    history_df = history_df.copy()
    history_df[ts_col] = pd.to_datetime(history_df[ts_col])
    new_ts = pd.to_datetime(new_tx.get("Timestamp", new_tx.get("timestamp", datetime.utcnow())))
    
    strict_history = history_df[history_df[ts_col] < new_ts].copy()
    tx_id = new_tx.get("tx_id", new_tx.get("id"))
    if tx_id:
        tx_id_col = "tx_id" if "tx_id" in strict_history.columns else "id"
        if tx_id_col in strict_history.columns:
            strict_history = strict_history[strict_history[tx_id_col] != tx_id]
            
    print(f"[Inference] Scoring tx from {sender_account}. Historical context edges: {len(strict_history)}")

    data, sender_idx, receiver_idx, new_edge_feat = build_sender_graph(
        sender_account, strict_history, new_tx, profiles_df, fe
    )

    with torch.no_grad():
        prob = model(data.x, data.edge_index, data.edge_attr, sender_idx, receiver_idx, new_edge_feat)
        prob_val = float(prob.item())
        if not (0.0 <= prob_val <= 1.0):
            raise ValueError(f"Invalid probability value: {prob_val}")

    is_fraud   = prob_val > FRAUD_THRESHOLD
    gat_signal = "HIGH RISK" if is_fraud else "LEGITIMATE"
    
    print(f"[Inference] Target transaction probability: {prob_val:.4f} -> {gat_signal}")

    return {
        "probability" : round(prob_val, 6),
        "is_fraud"    : is_fraud,
        "gat_signal"  : gat_signal,
        "num_nodes"   : data.x.shape[0],
        "num_edges"   : data.edge_index.shape[1],
    }

def smoke_test():
    """Read-only deterministic smoke test."""
    print("--- Running Smoke Test ---")
    load_checkpoint()
    
    import fraud_data
    dm = fraud_data.get_dm()
    df = dm.get_df()
    
    if len(df) == 0:
        print("No data in DB for smoke test.")
        return
        
    test_idx = 10 if len(df) > 10 else 0
    test_tx = df.iloc[test_idx].to_dict()
    
    sender = str(test_tx.get("From Account", test_tx.get("from_account")))
    from_col = "From Account" if "From Account" in df.columns else "from_account"
    sender_history = df[df[from_col].astype(str).str.strip() == sender]
    
    res = score_transaction(test_tx, sender_history, pd.DataFrame())
    print("Smoke Test Result:", res)
    print("--- Smoke Test Passed ---")
    
if __name__ == "__main__":
    smoke_test()
