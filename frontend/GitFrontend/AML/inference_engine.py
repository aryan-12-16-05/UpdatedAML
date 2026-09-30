"""
Inference Engine for GAT AML Detection — Stage 1
Handles:
  1. Model + scaler loading (cached at module level)
  2. Graph construction from MySQL sender history
  3. Feature engineering for node features (13) and edge features (20)
  4. Full inference pipeline → AML probability
"""

import os
import sys
import pickle
import warnings
import numpy as np
import pandas as pd
from datetime import datetime

# ── Path setup ───────────────────────────────────────────────────────────────
_AML_DIR    = os.path.dirname(__file__)
_MODELS_DIR = os.path.join(_AML_DIR, "models")

sys.path.insert(0, _AML_DIR)

# ── Model file paths (rename your .pt / .pkl to match these exactly) ─────────
MODEL_PATH  = os.path.join(_MODELS_DIR, "gat_aml_stage1.pt")   # GAT weights
SCALER_PATH = os.path.join(_MODELS_DIR, "scaler.pkl")           # StandardScaler for edge features

# ── Detection threshold ───────────────────────────────────────────────────────
FRAUD_THRESHOLD = 0.5  # probability above which a transaction is flagged


# ─────────────────────────────────────────────────────────────────────────────
#  FEATURE DEFINITIONS
#  IMPORTANT: These must exactly match what was used during model training.
#  If your training code used different features / ordering, adjust here.
# ─────────────────────────────────────────────────────────────────────────────

# 13 causal node features — sourced from customer_profiles table
NODE_FEATURE_COLS = [
    "incoming_transactions",      # 1  Historical incoming count
    "outgoing_transactions",      # 2  Historical outgoing count
    "total_incoming_amount",      # 3  Sum of all received amounts (USD)
    "total_outgoing_amount",      # 4  Sum of all sent amounts (USD)
    "average_incoming_amount",    # 5  Avg incoming per tx (USD)
    "average_outgoing_amount",    # 6  Avg outgoing per tx (USD)
    "maximum_incoming_amount",    # 7  Max single received amount (USD)
    "maximum_outgoing_amount",    # 8  Max single sent amount (USD)
    "net_flow",                   # 9  total_incoming - total_outgoing
    "unique_senders",             # 10 Unique accounts this node has received from
    "unique_receivers",           # 11 Unique accounts this node has sent to
    "fan_out_ratio",              # 12 unique_receivers / total_degree
    "pass_through_ratio",         # 13 min(in,out) / max(in,out)
]

# 20 edge features — engineered from raw transaction fields
PAYMENT_FORMAT_CATEGORIES = [
    "Cheque", "Credit Card", "Wire Transfer", "ACH", "Bitcoin", "Reinvestment"
    # "Cash" is the implicit drop-one (all zeros = Cash)
]

CURRENCY_MAP = {
    "US Dollar": 0, "Euro": 1, "UK Pound": 2, "Rupee": 3,
    "Yen": 4, "Yuan": 5, "Swiss Franc": 6, "Canadian Dollar": 7,
    "Australian Dollar": 8, "Ruble": 9, "Shekel": 10, "Bitcoin": 11,
    "Mexican Peso": 12, "Saudi Riyal": 13, "Brazil Real": 14
}
NUM_CURRENCIES = len(CURRENCY_MAP) + 1  # +1 for unknown


# ─────────────────────────────────────────────────────────────────────────────
#  LAZY-LOADED SINGLETONS
# ─────────────────────────────────────────────────────────────────────────────
_model  = None
_scaler = None
_device = None


def _get_device():
    global _device
    if _device is None:
        import torch
        _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return _device


def load_model():
    """Load GAT model weights (cached). Returns model in eval mode."""
    global _model
    if _model is not None:
        return _model

    import torch
    from models.gat_model_def import GATv2AMLModel

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")

    device = _get_device()
    model  = GATv2AMLModel().to(device)

    state_dict = torch.load(MODEL_PATH, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()

    _model = model
    print(f"[InferenceEngine] GAT model loaded from {MODEL_PATH} on {device}")
    return _model


def load_scaler():
    """Load the edge feature scaler (cached)."""
    global _scaler
    if _scaler is not None:
        return _scaler

    if not os.path.exists(SCALER_PATH):
        warnings.warn(
            f"[InferenceEngine] Scaler not found at {SCALER_PATH}. "
            "Edge features will NOT be scaled. Accuracy may be reduced."
        )
        return None

    with open(SCALER_PATH, "rb") as f:
        _scaler = pickle.load(f)

    print(f"[InferenceEngine] Scaler loaded from {SCALER_PATH}")
    return _scaler


# ─────────────────────────────────────────────────────────────────────────────
#  FEATURE ENGINEERING
# ─────────────────────────────────────────────────────────────────────────────

def _get_node_features(account_id: str, profiles_df: pd.DataFrame) -> np.ndarray:
    """
    Return the 13-dimensional node feature vector for an account.
    Falls back to zeros for unknown accounts.
    """
    acc = str(account_id).strip()
    if acc in profiles_df.index:
        row = profiles_df.loc[acc]
        if isinstance(row, pd.DataFrame):
            row = row.iloc[0]
        feats = []
        for col in NODE_FEATURE_COLS:
            # Try both original and MySQL-lowercased column names
            val = row.get(col, row.get(col.lower(), 0.0))
            feats.append(float(val) if pd.notna(val) else 0.0)
        return np.array(feats, dtype=np.float32)
    return np.zeros(len(NODE_FEATURE_COLS), dtype=np.float32)


def _engineer_edge_features(tx: dict) -> np.ndarray:
    """
    Build the 20-dimensional edge feature vector for a single transaction dict.
    The dict must have: Amount Paid, Amount Received, Payment Format,
    Payment Currency, Receiving Currency, Timestamp, previous_outgoing.
    """
    feats = []

    # ── 1-3: Amount features ─────────────────────────────────────────────
    amt_paid = max(float(tx.get("Amount Paid",     tx.get("amount_paid",     0.0))), 0.0)
    amt_recv = max(float(tx.get("Amount Received", tx.get("amount_received", 0.0))), 0.0)
    ratio    = (amt_recv / amt_paid) if amt_paid > 0 else 1.0
    ratio    = min(ratio, 5.0)  # clip outliers

    feats.append(np.log1p(amt_paid))   # 1
    feats.append(np.log1p(amt_recv))   # 2
    feats.append(ratio)                # 3

    # ── 4-8: Timestamp features ───────────────────────────────────────────
    ts_raw = tx.get("Timestamp", tx.get("timestamp", datetime.utcnow()))
    if isinstance(ts_raw, str):
        try:
            ts = pd.to_datetime(ts_raw)
        except Exception:
            ts = datetime.utcnow()
    elif isinstance(ts_raw, datetime):
        ts = ts_raw
    else:
        ts = datetime.utcnow()

    feats.append(float(ts.hour))                      # 4  hour (0-23)
    feats.append(float(ts.weekday()))                  # 5  day of week (0-6)
    feats.append(float(ts.month))                      # 6  month (1-12)
    feats.append(float(ts.day))                        # 7  day of month (1-31)
    feats.append(1.0 if ts.weekday() >= 5 else 0.0)   # 8  is_weekend

    # ── 9-14: Payment format one-hot (6 categories; Cash = all-zeros) ─────
    pf = str(tx.get("Payment Format", tx.get("payment_format", ""))).strip()
    for cat in PAYMENT_FORMAT_CATEGORIES:
        feats.append(1.0 if pf == cat else 0.0)       # 9-14

    # ── 15-17: Currency features ──────────────────────────────────────────
    pay_cur  = str(tx.get("Payment Currency",  tx.get("payment_currency",  "US Dollar")))
    recv_cur = str(tx.get("Receiving Currency", tx.get("receiving_currency", "US Dollar")))

    feats.append(1.0 if pay_cur == recv_cur else 0.0)                       # 15 is_same_currency
    feats.append(float(CURRENCY_MAP.get(pay_cur,  NUM_CURRENCIES - 1)))     # 16 pay currency idx
    feats.append(float(CURRENCY_MAP.get(recv_cur, NUM_CURRENCIES - 1)))     # 17 recv currency idx

    # ── 18-19: Bank features ──────────────────────────────────────────────
    from_bank = float(tx.get("From Bank", tx.get("from_bank", 0)))
    to_bank   = float(tx.get("To Bank",   tx.get("to_bank",   0)))
    feats.append(from_bank / 300_000.0)   # 18  normalized bank ID (approx max)
    feats.append(to_bank   / 300_000.0)   # 19

    # ── 20: Sender historical activity ───────────────────────────────────
    prev_out = max(float(tx.get("previous_outgoing", 0)), 0.0)
    feats.append(np.log1p(prev_out))       # 20

    return np.array(feats, dtype=np.float32)  # [20]


# ─────────────────────────────────────────────────────────────────────────────
#  GRAPH CONSTRUCTION
# ─────────────────────────────────────────────────────────────────────────────

def build_sender_graph(sender_account: str, history_df: pd.DataFrame,
                       new_tx: dict, profiles_df: pd.DataFrame):
    """
    Build a PyTorch Geometric Data object for the sender's full sub-graph.

    Nodes:  all unique accounts that appear in sender's history + the new receiver
    Edges:  all historical transactions + the new transaction (appended last)

    Returns:
        data         : torch_geometric.data.Data object
        sender_idx   : int — node index of the sender in the graph
        receiver_idx : int — node index of the new transaction's receiver
        new_edge_feat: torch.Tensor [20] — features of the NEW transaction (scaled)
    """
    import torch
    from torch_geometric.data import Data

    scaler = load_scaler()
    device = _get_device()

    # ── 1. Collect all accounts ─────────────────────────────────────────────
    from_col = "From Account" if "From Account" in history_df.columns else "from_account"
    to_col   = "To Account"   if "To Account"   in history_df.columns else "to_account"

    new_receiver = str(new_tx.get("To Account", new_tx.get("to_account", "UNKNOWN"))).strip()
    all_accounts = (
        list(history_df[from_col].astype(str).str.strip().unique()) +
        list(history_df[to_col].astype(str).str.strip().unique()) +
        [str(sender_account).strip(), new_receiver]
    )
    unique_accounts = list(dict.fromkeys(all_accounts))  # preserve order, deduplicate
    account_to_idx  = {acc: i for i, acc in enumerate(unique_accounts)}

    # ── 2. Build node feature matrix [num_nodes, 13] ────────────────────────
    node_feats = np.stack(
        [_get_node_features(acc, profiles_df) for acc in unique_accounts],
        axis=0
    )

    # ── 3. Build edge list and edge features ────────────────────────────────
    edge_src, edge_dst, edge_attrs = [], [], []

    for _, row in history_df.iterrows():
        src = str(row.get(from_col, "")).strip()
        dst = str(row.get(to_col,   "")).strip()
        if src not in account_to_idx or dst not in account_to_idx:
            continue
        edge_src.append(account_to_idx[src])
        edge_dst.append(account_to_idx[dst])
        edge_attrs.append(_engineer_edge_features(row.to_dict()))

    # ── 4. Append the new (target) transaction as the last edge ─────────────
    sender_idx   = account_to_idx[str(sender_account).strip()]
    receiver_idx = account_to_idx[new_receiver]

    new_edge_feat_raw = _engineer_edge_features(new_tx)  # [20]

    edge_src.append(sender_idx)
    edge_dst.append(receiver_idx)
    edge_attrs.append(new_edge_feat_raw)

    # ── 5. Scale edge features ───────────────────────────────────────────────
    edge_attrs_np = np.stack(edge_attrs, axis=0)  # [E, 20]
    if scaler is not None:
        try:
            edge_attrs_np = scaler.transform(edge_attrs_np)
        except Exception as e:
            warnings.warn(f"[InferenceEngine] Scaler transform failed: {e}. Using raw features.")

    new_edge_feat = torch.tensor(edge_attrs_np[-1], dtype=torch.float32).to(device)

    # ── 6. Assemble PyG Data object ──────────────────────────────────────────
    x          = torch.tensor(node_feats,  dtype=torch.float32).to(device)    # [N, 13]
    edge_index = torch.tensor(
        [edge_src, edge_dst], dtype=torch.long
    ).to(device)                                                                # [2, E]
    edge_attr  = torch.tensor(edge_attrs_np, dtype=torch.float32).to(device)   # [E, 20]

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

    return data, sender_idx, receiver_idx, new_edge_feat


# ─────────────────────────────────────────────────────────────────────────────
#  MAIN INFERENCE FUNCTION
# ─────────────────────────────────────────────────────────────────────────────

def score_transaction(new_tx: dict, history_df: pd.DataFrame,
                      profiles_df: pd.DataFrame) -> dict:
    """
    Run the full GAT inference pipeline for a single new transaction.

    Args:
        new_tx      : dict with transaction fields (see API schema below)
        history_df  : DataFrame of ALL previous transactions for this sender
                      (fetched from MySQL by the caller)
        profiles_df : customer_profiles DataFrame indexed by account_number

    Returns:
        {
          "probability" : float,   AML probability in [0, 1]
          "is_fraud"    : bool,    True if probability >= FRAUD_THRESHOLD
          "gat_signal"  : str,     "HIGH RISK" | "LEGITIMATE"
          "num_nodes"   : int,     nodes in the constructed graph
          "num_edges"   : int,     edges in the constructed graph
        }
    """
    import torch

    model   = load_model()
    device  = _get_device()

    sender_account = str(new_tx.get("From Account", new_tx.get("from_account", ""))).strip()

    # Build the graph
    data, sender_idx, receiver_idx, new_edge_feat = build_sender_graph(
        sender_account, history_df, new_tx, profiles_df
    )

    # Forward pass (no gradient needed)
    with torch.no_grad():
        prob = model(
            data.x,
            data.edge_index,
            data.edge_attr,
            sender_idx,
            receiver_idx,
            new_edge_feat
        )
        prob_val = float(prob.item())

    is_fraud   = prob_val >= FRAUD_THRESHOLD
    gat_signal = "HIGH RISK" if is_fraud else "LEGITIMATE"

    return {
        "probability" : round(prob_val, 6),
        "is_fraud"    : is_fraud,
        "gat_signal"  : gat_signal,
        "num_nodes"   : data.x.shape[0],
        "num_edges"   : data.edge_index.shape[1],
    }
