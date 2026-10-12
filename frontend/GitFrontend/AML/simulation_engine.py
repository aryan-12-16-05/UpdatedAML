"""
Simulation Engine for AML Rule-Based Override & GAT Ingestion
Processes transactions from GitData/simulate_trans.csv.

CSV Input — only these columns are required (others are derived/computed):
  transaction_id, Timestamp, From Bank, Account, To Bank, Account.1,
  Amount Received, Receiving Currency, Amount Paid, Payment Currency, Payment Format

Missing derived fields (Is Laundering, episode_no, Amount Paid USD, etc.)
are computed automatically here before DB insertion.

Routing Logic:
  SAFE  -> analyst approved this sender   -> bypass model -> LEGITIMATE / LOW RISK
  FRAUD -> analyst blocked this sender    -> bypass model -> LAUNDERING / HIGH RISK (shown in dashboard)
  NULL  -> no decision on this sender     -> run GAT model inference + retroactive update
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

# Currency conversion rates (mirrors fraud_data.USD_CONVERSION_RATES)
_FALLBACK_USD_RATES = {
    "US Dollar":         1.0,
    "Euro":              1.10,
    "Rupee":             0.012,
    "UK Pound":          1.26,
    "Yen":               0.0067,
    "Yuan":              0.14,
    "Swiss Franc":       1.11,
    "Canadian Dollar":   0.74,
    "Australian Dollar": 0.65,
    "Ruble":             0.011,
    "Shekel":            0.27,
    "Bitcoin":           65000.0,
    "Mexican Peso":      0.059,
    "Saudi Riyal":       0.27,
    "Brazil Real":       0.20,
}


def _to_usd(amount: float, currency: str) -> float:
    """Convert raw amount in `currency` to USD."""
    rates = getattr(fraud_data, "USD_CONVERSION_RATES", _FALLBACK_USD_RATES)
    rate  = rates.get(str(currency).strip(), 1.0)
    return round(float(amount) * rate, 6)


def get_simulate_csv_path() -> str:
    """Finds simulate_trans.csv in GitData directory."""
    primary_path = os.path.join(_GITDATA_DIR, "simulate_trans.csv")
    if os.path.exists(primary_path):
        return primary_path
    fallback_path = r"c:\Users\aryan\frontend\GitData\simulate_trans.csv"
    if os.path.exists(fallback_path):
        return fallback_path
    return primary_path


def _parse_row(row: pd.Series) -> dict:
    """
    Parse one CSV row into a normalised dict.
    Handles column aliases and converts raw currency amounts to USD.
    """
    sender   = str(row.get("Account",   row.get("From Account",  row.get("from_account",  "")))).strip()
    receiver = str(row.get("Account.1", row.get("To Account",    row.get("to_account",    "")))).strip()

    tx_id = str(row.get("transaction_id", row.get("Transaction ID", ""))).strip()
    if not tx_id or tx_id.lower() in ("nan", ""):
        tx_id = f"SIM-{uuid.uuid4().hex[:8]}"

    from_bank = int(float(str(row.get("From Bank", row.get("from_bank", 0))).replace(",", "") or 0))
    to_bank   = int(float(str(row.get("To Bank",   row.get("to_bank",   0))).replace(",", "") or 0))

    pay_cur  = str(row.get("Payment Currency",   row.get("payment_currency",   "US Dollar"))).strip()
    recv_cur = str(row.get("Receiving Currency", row.get("receiving_currency", "US Dollar"))).strip()

    amt_paid_raw = float(str(row.get("Amount Paid",     row.get("amount_paid",     0.0))).replace(",", "") or 0.0)
    amt_recv_raw = float(str(row.get("Amount Received", row.get("amount_received", amt_paid_raw))).replace(",", "") or 0.0)

    # Convert to USD (what the DB stores as amount_paid / amount_received)
    amt_paid_usd = _to_usd(amt_paid_raw, pay_cur)
    amt_recv_usd = _to_usd(amt_recv_raw, recv_cur)

    pf = str(row.get("Payment Format", row.get("payment_format", "Wire Transfer"))).strip()

    ts_val = row.get("Timestamp", row.get("timestamp", None))
    if ts_val is None or (isinstance(ts_val, float) and pd.isna(ts_val)):
        ts_str = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
    else:
        ts_str = str(ts_val).replace("/", "-").strip()
        if len(ts_str) == 16:
            ts_str += ":00"

    return {
        "tx_id":        tx_id,
        "sender":       sender,
        "receiver":     receiver,
        "from_bank":    from_bank,
        "to_bank":      to_bank,
        "pay_cur":      pay_cur,
        "recv_cur":     recv_cur,
        "amt_paid_usd": amt_paid_usd,
        "amt_recv_usd": amt_recv_usd,
        "pf":           pf,
        "ts_str":       ts_str,
    }


def run_simulation(csv_path: str = None) -> dict:
    """
    Process every row in simulate_trans.csv using rule-based status overrides
    or full GAT model inference.

    Returns:
        {
          "status":  "success" | "empty" | "error",
          "total":   int,
          "counts":  {"SAFE": int, "FRAUD": int, "NULL": int},
          "results": [per-transaction outcome dicts]
        }
    """
    if csv_path is None:
        csv_path = get_simulate_csv_path()

    if not os.path.exists(csv_path):
        return {"status": "error", "message": f"Simulation file not found at: {csv_path}"}

    if os.path.getsize(csv_path) == 0:
        return {"status": "empty", "message": "simulate_trans.csv is empty. Add transaction records to simulate."}

    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        return {"status": "error", "message": f"Error reading CSV: {str(e)}"}

    if df.empty:
        return {"status": "empty", "message": "simulate_trans.csv contains no rows."}

    results = []
    counts  = {"SAFE": 0, "FRAUD": 0, "NULL": 0}

    # Best-effort: load customer profiles for GAT inference
    try:
        dm = fraud_data.get_dm()
        profiles_df = dm._profiles_df if hasattr(dm, "_profiles_df") else pd.DataFrame()
    except Exception:
        profiles_df = pd.DataFrame()

    engine = fanout_updater._get_engine()

    for idx, row in df.iterrows():
        try:
            p = _parse_row(row)
        except Exception as e:
            results.append({
                "tx_id": f"ROW-{idx}", "sender": "PARSE ERROR", "receiver": "",
                "status": "Error", "action": f"Row parse failed: {e}",
                "risk": "UNKNOWN", "is_fraud": False, "fan_out_group": "None",
            })
            continue

        sender   = p["sender"]
        receiver = p["receiver"]
        tx_id    = p["tx_id"]
        ts_str   = p["ts_str"]

        # Build the tx_dict with USD-normalised amounts for DB storage
        tx_dict = {
            "Transaction ID":     tx_id,
            "From Account":       sender,
            "To Account":         receiver,
            "Amount Paid":        p["amt_paid_usd"],
            "Amount Received":    p["amt_recv_usd"],
            "Payment Currency":   p["pay_cur"],
            "Receiving Currency": p["recv_cur"],
            "Payment Format":     p["pf"],
            "From Bank":          p["from_bank"],
            "To Bank":            p["to_bank"],
            "Timestamp":          ts_str,
        }

        # Determine routing via analyst decision
        account_status = decisions_db.get_account_status(sender)

        # ==============================================================
        # SAFE: analyst approved this sender -> legitimate, no model
        # ==============================================================
        if account_status == "SAFE":
            prob          = 0.05
            is_fraud      = False
            gat_signal    = "LOW"
            fan_out_group = None

            fanout_updater.insert_new_transaction(
                tx_dict=tx_dict, gat_prob=prob, gat_signal=gat_signal,
                is_fraud=is_fraud, fan_out_group=fan_out_group, actual_label=0
            )
            counts["SAFE"] += 1
            results.append({
                "tx_id": tx_id, "sender": sender, "receiver": receiver,
                "status": "SAFE",
                "action": "Model Bypassed - Rule: SAFE Account (Analyst Approved)",
                "risk": "LOW", "is_fraud": False, "fan_out_group": "None",
            })

        # ==============================================================
        # FRAUD: analyst blocked this sender -> high-risk, no model
        # ==============================================================
        elif account_status == "FRAUD":
            prob       = 0.99
            is_fraud   = True
            gat_signal = "HIGH"

            fan_out_group = None
            try:
                from sqlalchemy import text
                with engine.connect() as conn:
                    res = conn.execute(
                        text("SELECT fan_out_group FROM transactions "
                             "WHERE from_account = :acc AND fan_out_group IS NOT NULL LIMIT 1"),
                        {"acc": sender}
                    ).fetchone()
                    fan_out_group = int(res[0]) if res else fanout_updater._assign_new_group(conn)
            except Exception:
                fan_out_group = 999999

            fanout_updater.insert_new_transaction(
                tx_dict=tx_dict, gat_prob=prob, gat_signal=gat_signal,
                is_fraud=is_fraud, fan_out_group=fan_out_group, actual_label=1
            )
            fanout_updater.retroactively_label_sender(sender, prob, fan_out_group)

            counts["FRAUD"] += 1
            results.append({
                "tx_id": tx_id, "sender": sender, "receiver": receiver,
                "status": "FRAUD",
                "action": "Model Bypassed - Rule: FRAUD Account (Analyst Blocked)",
                "risk": "HIGH", "is_fraud": True, "fan_out_group": fan_out_group,
            })

        # ==============================================================
        # NULL: no decision -> full GAT model inference + retroactive update
        # ==============================================================
        else:
            try:
                history_query = """
                    SELECT * FROM transactions
                    WHERE (from_account = %s OR to_account = %s
                        OR from_account = %s OR to_account = %s)
                      AND timestamp < %s
                    ORDER BY timestamp ASC
                """
                history_df = pd.read_sql(
                    history_query, engine,
                    params=(sender, sender, receiver, receiver, ts_str)
                )
                history_df.rename(columns={
                    "transaction_id":     "Transaction ID",
                    "from_account":       "From Account",
                    "to_account":         "To Account",
                    "amount_paid":        "Amount Paid",
                    "amount_received":    "Amount Received",
                    "payment_currency":   "Payment Currency",
                    "receiving_currency": "Receiving Currency",
                    "payment_format":     "Payment Format",
                    "timestamp":          "Timestamp",
                    "fan_out_group":      "Fan-out Group",
                }, inplace=True)

                import inference_engine
                score_res  = inference_engine.score_transaction(tx_dict, history_df, profiles_df)
                prob       = score_res["probability"]
                is_fraud   = score_res["is_fraud"]
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
                    tx_dict=tx_dict, gat_prob=prob, gat_signal=gat_signal,
                    is_fraud=is_fraud, fan_out_group=fan_out_group,
                    actual_label=1 if is_fraud else 0
                )
                if is_fraud:
                    fanout_updater.retroactively_label_sender(sender, prob, fan_out_group)

                counts["NULL"] += 1
                results.append({
                    "tx_id": tx_id, "sender": sender, "receiver": receiver,
                    "status": "NULL",
                    "action": f"GAT Model Inferred (Prob: {prob:.4f}, Signal: {gat_signal})",
                    "risk": "HIGH" if is_fraud else "LOW",
                    "is_fraud": is_fraud,
                    "fan_out_group": fan_out_group if fan_out_group else "None",
                })

            except Exception as e:
                results.append({
                    "tx_id": tx_id, "sender": sender, "receiver": receiver,
                    "status": "NULL (Error)",
                    "action": f"Inference failed: {str(e)}",
                    "risk": "UNKNOWN", "is_fraud": False, "fan_out_group": "None",
                })

    # Invalidate in-memory cache so Streamlit reloads from DB on next render
    fanout_updater._invalidate_dataset_cache()

    return {
        "status":  "success",
        "total":   len(df),
        "counts":  counts,
        "results": results,
    }