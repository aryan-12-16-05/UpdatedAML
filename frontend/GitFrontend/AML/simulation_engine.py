"""
Simulation Engine for AML Rule-Based Override & GAT Ingestion
Processes transactions from GitData/simulate_trans.csv:
  - If sender status is 'SAFE'  -> Bypass GAT model; classify as LEGITIMATE (LOW); no fraud alert.
  - If sender status is 'FRAUD' -> Bypass GAT model; classify directly as LAUNDERING (HIGH); display in dashboard.
  - If sender status is 'NULL'  -> Evaluate through GAT model inference.
"""

import os
import sys
import uuid
from datetime import datetime
import pandas as pd

_AML_DIR = os.path.dirname(os.path.abspath(__file__))
_GITDATA_DIR = os.path.abspath(os.path.join(_AML_DIR, "..", "..", "GitData"))
sys.path.insert(0, _AML_DIR)

import decisions_db
import fanout_updater
import fraud_data

def get_simulate_csv_path() -> str:
    """Finds simulate_trans.csv in GitData directory."""
    primary_path = os.path.join(_GITDATA_DIR, "simulate_trans.csv")
    if os.path.exists(primary_path):
        return primary_path
    
    fallback_path = r"c:\Users\aryan\frontend\GitData\simulate_trans.csv"
    if os.path.exists(fallback_path):
        return fallback_path
        
    return primary_path

def run_simulation(csv_path: str = None) -> dict:
    """
    Executes transaction simulation from simulate_trans.csv.
    Applies rule-based injection overrides or model evaluation.
    """
    if csv_path is None:
        csv_path = get_simulate_csv_path()

    if not os.path.exists(csv_path):
        return {
            "status": "error",
            "message": f"Simulation file not found at: {csv_path}"
        }

    # Check if file has content
    if os.path.getsize(csv_path) == 0:
        return {
            "status": "empty",
            "message": "simulate_trans.csv is currently empty (0 bytes). Add transaction records to simulate."
        }

    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        return {
            "status": "error",
            "message": f"Error reading CSV file: {str(e)}"
        }

    if df.empty:
        return {
            "status": "empty",
            "message": "simulate_trans.csv contains no rows to process."
        }

    results = []
    counts = {"SAFE": 0, "FRAUD": 0, "NULL": 0}

    # Load profiles if available for inference
    try:
        dm = fraud_data.get_dm()
        profiles_df = dm._profiles_df if hasattr(dm, "_profiles_df") else pd.DataFrame()
    except Exception:
        profiles_df = pd.DataFrame()

    engine = fanout_updater._get_engine()

    for idx, row in df.iterrows():
        # Normalize column mappings
        sender = str(row.get("Account", row.get("From Account", row.get("from_account", "")))).strip()
        receiver = str(row.get("Account.1", row.get("To Account", row.get("to_account", "")))).strip()
        
        tx_id = str(row.get("transaction_id", row.get("Transaction ID", f"SIM-{uuid.uuid4().hex[:8]}"))).strip()
        if not tx_id or tx_id == "nan":
            tx_id = f"SIM-{uuid.uuid4().hex[:8]}"

        amt_paid = float(row.get("Amount Paid", row.get("amount_paid", row.get("Amount Paid USD", 0.0))))
        amt_recv = float(row.get("Amount Received", row.get("amount_received", row.get("Amount Received USD", amt_paid))))
        pay_cur = str(row.get("Payment Currency", row.get("payment_currency", "US Dollar")))
        recv_cur = str(row.get("Receiving Currency", row.get("receiving_currency", "US Dollar")))
        pf = str(row.get("Payment Format", row.get("payment_format", "Wire Transfer")))
        from_bank = int(row.get("From Bank", row.get("from_bank", 0)))
        to_bank = int(row.get("To Bank", row.get("to_bank", 0)))
        
        ts_val = row.get("Timestamp", row.get("timestamp", datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")))
        if pd.isna(ts_val):
            ts_str = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
        else:
            ts_str = str(ts_val).replace("/", "-")

        tx_dict = {
            "Transaction ID": tx_id,
            "From Account": sender,
            "To Account": receiver,
            "Amount Paid": amt_paid,
            "Amount Received": amt_recv,
            "Payment Currency": pay_cur,
            "Receiving Currency": recv_cur,
            "Payment Format": pf,
            "From Bank": from_bank,
            "To Bank": to_bank,
            "Timestamp": ts_str
        }

        # ── Check Account Status Override ──────────────────────────────────
        account_status = decisions_db.get_account_status(sender)

        if account_status == "SAFE":
            # Rule 1: SAFE sender -> Bypass GAT model; classify as LEGITIMATE (LOW); no fraud alert
            prob = 0.05
            is_fraud = False
            gat_signal = "LOW"
            fan_out_group = None

            fanout_updater.insert_new_transaction(
                tx_dict=tx_dict,
                gat_prob=prob,
                gat_signal=gat_signal,
                is_fraud=is_fraud,
                fan_out_group=fan_out_group,
                actual_label=0
            )

            counts["SAFE"] += 1
            results.append({
                "tx_id": tx_id,
                "sender": sender,
                "receiver": receiver,
                "status": "SAFE",
                "action": "Model Bypassed (Rule: SAFE Account)",
                "risk": "LOW",
                "is_fraud": False,
                "fan_out_group": "None"
            })

        elif account_status == "FRAUD":
            # Rule 2: FRAUD sender -> Bypass GAT model; classify directly as LAUNDERING (HIGH); show in dashboard
            prob = 0.99
            is_fraud = True
            gat_signal = "HIGH"

            # Resolve or assign fan_out_group
            fan_out_group = None
            try:
                from sqlalchemy import text
                with engine.connect() as conn:
                    res = conn.execute(
                        text("SELECT fan_out_group FROM transactions WHERE from_account = :acc AND fan_out_group IS NOT NULL LIMIT 1"),
                        {"acc": sender}
                    ).fetchone()
                    fan_out_group = int(res[0]) if res else fanout_updater._assign_new_group(conn)
            except Exception:
                fan_out_group = 999999

            fanout_updater.insert_new_transaction(
                tx_dict=tx_dict,
                gat_prob=prob,
                gat_signal=gat_signal,
                is_fraud=is_fraud,
                fan_out_group=fan_out_group,
                actual_label=1
            )

            # Retroactively update historical transactions from this sender so full group is high-risk
            fanout_updater.retroactively_label_sender(sender, prob, fan_out_group)

            counts["FRAUD"] += 1
            results.append({
                "tx_id": tx_id,
                "sender": sender,
                "receiver": receiver,
                "status": "FRAUD",
                "action": "Model Bypassed (Rule: FRAUD Account)",
                "risk": "HIGH",
                "is_fraud": True,
                "fan_out_group": fan_out_group
            })

        else:
            # Rule 3: NULL / Unassigned sender -> Run GAT Model Inference
            try:
                history_query = """
                    SELECT * FROM transactions 
                    WHERE (from_account = %s OR to_account = %s OR from_account = %s OR to_account = %s)
                      AND timestamp < %s
                    ORDER BY timestamp ASC
                """
                history_df = pd.read_sql(history_query, engine, params=(sender, sender, receiver, receiver, ts_str))
                
                col_mapping = {
                    "transaction_id": "Transaction ID",
                    "from_account": "From Account",
                    "to_account": "To Account",
                    "amount_paid": "Amount Paid",
                    "amount_received": "Amount Received",
                    "payment_currency": "Payment Currency",
                    "receiving_currency": "Receiving Currency",
                    "payment_format": "Payment Format",
                    "timestamp": "Timestamp",
                    "fan_out_group": "Fan-out Group"
                }
                history_df.rename(columns=col_mapping, inplace=True)

                import inference_engine
                score_res = inference_engine.score_transaction(tx_dict, history_df, profiles_df)
                prob = score_res["probability"]
                is_fraud = score_res["is_fraud"]
                gat_signal = score_res["gat_signal"]

                fan_out_group = None
                if is_fraud:
                    grp_col = "Fan-out Group" if "Fan-out Group" in history_df.columns else "fan_out_group"
                    if grp_col in history_df.columns:
                        existing_grp = history_df[grp_col].dropna()
                        if not existing_grp.empty:
                            fan_out_group = int(existing_grp.iloc[0])
                    if fan_out_group is None:
                        from sqlalchemy import text
                        with engine.connect() as conn:
                            fan_out_group = fanout_updater._assign_new_group(conn)

                fanout_updater.insert_new_transaction(
                    tx_dict=tx_dict,
                    gat_prob=prob,
                    gat_signal=gat_signal,
                    is_fraud=is_fraud,
                    fan_out_group=fan_out_group,
                    actual_label=1 if is_fraud else 0
                )

                if is_fraud:
                    fanout_updater.retroactively_label_sender(sender, prob, fan_out_group)

                counts["NULL"] += 1
                results.append({
                    "tx_id": tx_id,
                    "sender": sender,
                    "receiver": receiver,
                    "status": "NULL",
                    "action": f"GAT Inferred (Prob: {prob:.4f})",
                    "risk": "HIGH" if is_fraud else "LOW",
                    "is_fraud": is_fraud,
                    "fan_out_group": fan_out_group if fan_out_group else "None"
                })

            except Exception as e:
                results.append({
                    "tx_id": tx_id,
                    "sender": sender,
                    "receiver": receiver,
                    "status": "NULL (Error)",
                    "action": f"Inference failed: {str(e)}",
                    "risk": "UNKNOWN",
                    "is_fraud": False,
                    "fan_out_group": "None"
                })

    # Reset in-memory cache so Streamlit immediately reloads updated DB rows
    fanout_updater._invalidate_dataset_cache()

    return {
        "status": "success",
        "total": len(df),
        "counts": counts,
        "results": results
    }
