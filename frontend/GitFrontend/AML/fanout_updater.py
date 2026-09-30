"""
Fan-Out Retroactive Updater
When the GAT model detects a new transaction from sender S as fraudulent (AML),
this module:
  1. Looks up ALL previous transactions from sender S in the database
  2. Updates their labels: actual_label=1, gat_risk_level='HIGH', gat_risk_probability=detected_prob
  3. Assigns or confirms the fan_out_group for all these transactions
  4. Optionally refreshes the in-memory DatasetManager so the dashboard sees the change

This implements the "retroactive episode labeling" behavior:
  Tx1 (legit) → Tx2 (legit) → Tx3 (legit) → ... → Tx6 (DETECTED!) 
  ↓ trigger update
  Tx1 → Tx5 all get flipped to laundering=1 in the database
"""

import os
import sys
import warnings
from datetime import datetime

_AML_DIR = os.path.dirname(__file__)
sys.path.insert(0, _AML_DIR)


def _get_engine():
    """Build and return a SQLAlchemy engine using the same credentials as decisions_db."""
    import urllib.parse
    from sqlalchemy import create_engine
    import streamlit as st

    DB_HOST     = "gateway01.ap-northeast-1.prod.aws.tidbcloud.com"
    DB_PORT     = 4000
    DB_USER     = "3JTMKeEP2m268Uk.root"
    DB_PASSWORD = "KUOElkUyvNZX8HBL"
    DB_NAME     = "aml_fraud_db"

    # Override from Streamlit secrets if available
    try:
        if hasattr(st, "secrets"):
            DB_HOST     = st.secrets.get("DB_HOST",     DB_HOST)
            DB_PORT     = st.secrets.get("DB_PORT",     DB_PORT)
            DB_USER     = st.secrets.get("DB_USER",     DB_USER)
            DB_PASSWORD = st.secrets.get("DB_PASSWORD", DB_PASSWORD)
            DB_NAME     = st.secrets.get("DB_NAME",     DB_NAME)
    except Exception:
        pass

    encoded_password = urllib.parse.quote_plus(DB_PASSWORD)
    return create_engine(
        f"mysql+pymysql://{DB_USER}:{encoded_password}@{DB_HOST}:{DB_PORT}/{DB_NAME}",
        connect_args={"ssl": {"ssl_cert": None}, "connect_timeout": 15}
    )


def retroactively_label_sender(sender_account: str, detected_probability: float,
                                fan_out_group: int = None) -> dict:
    """
    Mark ALL historical transactions from `sender_account` as money laundering.

    Args:
        sender_account      : The From Account that was detected as fraudulent
        detected_probability: The GAT probability that triggered detection
        fan_out_group       : If known, the fan-out group ID to assign

    Returns:
        {
          "updated_rows"  : int   — number of rows retroactively updated,
          "fan_out_group" : int   — the group ID used,
          "success"       : bool
        }
    """
    import pandas as pd
    from sqlalchemy import text

    engine = _get_engine()
    sender = str(sender_account).strip()
    ts     = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")

    try:
        with engine.connect() as conn:
            # ── 1. Find the fan_out_group for this sender ─────────────────────
            if fan_out_group is None:
                result = conn.execute(
                    text("SELECT fan_out_group FROM transactions "
                         "WHERE from_account = :acc AND fan_out_group IS NOT NULL "
                         "LIMIT 1"),
                    {"acc": sender}
                ).fetchone()
                fan_out_group = int(result[0]) if result else _assign_new_group(conn)

            # ── 2. Update all historical transactions from this sender ─────────
            update_result = conn.execute(
                text("""
                    UPDATE transactions
                    SET
                        actual_label         = 1,
                        gat_risk_level       = 'HIGH',
                        gat_risk_probability = :prob,
                        behavior_signal      = 'FAN-OUT',
                        fan_out_group        = :grp
                    WHERE from_account = :acc
                      AND actual_label != 1
                """),
                {
                    "prob": round(detected_probability, 6),
                    "grp" : fan_out_group,
                    "acc" : sender
                }
            )
            conn.commit()
            updated_rows = update_result.rowcount

        print(
            f"[FanOutUpdater] Retroactively labeled {updated_rows} transactions "
            f"from sender {sender} → group {fan_out_group} (prob={detected_probability:.4f})"
        )

        # ── 3. Invalidate the in-memory DatasetManager so dashboard refreshes ─
        _invalidate_dataset_cache()

        return {
            "updated_rows" : updated_rows,
            "fan_out_group": fan_out_group,
            "success"      : True
        }

    except Exception as e:
        warnings.warn(f"[FanOutUpdater] Failed to retroactively label sender {sender}: {e}")
        return {
            "updated_rows" : 0,
            "fan_out_group": fan_out_group,
            "success"      : False,
            "error"        : str(e)
        }


def _assign_new_group(conn) -> int:
    """Get the next available fan_out_group ID from the database."""
    from sqlalchemy import text
    result = conn.execute(
        text("SELECT COALESCE(MAX(fan_out_group), 0) + 1 FROM transactions")
    ).fetchone()
    return int(result[0]) if result else 1


def insert_new_transaction(tx_dict: dict, gat_prob: float, gat_signal: str,
                            is_fraud: bool, fan_out_group: int) -> bool:
    """
    Insert a newly ingested transaction into the MySQL `transactions` table.

    Args:
        tx_dict      : raw transaction fields (see API schema)
        gat_prob     : model-predicted AML probability
        gat_signal   : "HIGH RISK" | "LEGITIMATE"
        is_fraud     : True if detected as money laundering
        fan_out_group: assigned fan-out group ID

    Returns:
        True on success, False on failure
    """
    from sqlalchemy import text

    engine = _get_engine()
    actual_label    = 1 if is_fraud else 0
    behavior_signal = "FAN-OUT" if is_fraud else "LEGITIMATE"

    try:
        ts_raw = tx_dict.get("Timestamp", tx_dict.get("timestamp", datetime.utcnow()))
        if hasattr(ts_raw, "strftime"):
            ts_str = ts_raw.strftime("%Y-%m-%d %H:%M:%S")
        else:
            ts_str = str(ts_raw)

        with engine.connect() as conn:
            conn.execute(
                text("""
                    INSERT INTO transactions (
                        transaction_id, from_account, to_account,
                        amount_paid, amount_received,
                        payment_currency, receiving_currency,
                        payment_format, timestamp,
                        from_bank, to_bank,
                        fan_out_group,
                        actual_label, gat_risk_level, gat_risk_probability,
                        behavior_signal,
                        previous_outgoing, previous_incoming,
                        detection_reason
                    ) VALUES (
                        :txid, :from_acc, :to_acc,
                        :amt_paid, :amt_recv,
                        :pay_cur, :recv_cur,
                        :pf, :ts,
                        :from_bank, :to_bank,
                        :grp,
                        :label, :gat_lvl, :gat_prob,
                        :bsig,
                        :prev_out, :prev_in,
                        :reason
                    )
                    ON DUPLICATE KEY UPDATE
                        gat_risk_level       = VALUES(gat_risk_level),
                        gat_risk_probability = VALUES(gat_risk_probability),
                        actual_label         = VALUES(actual_label)
                """),
                {
                    "txid"    : str(tx_dict.get("Transaction ID",   tx_dict.get("transaction_id",   "LIVE-" + datetime.utcnow().strftime("%f")))),
                    "from_acc": str(tx_dict.get("From Account",     tx_dict.get("from_account",     ""))),
                    "to_acc"  : str(tx_dict.get("To Account",       tx_dict.get("to_account",       ""))),
                    "amt_paid": float(tx_dict.get("Amount Paid",    tx_dict.get("amount_paid",       0.0))),
                    "amt_recv": float(tx_dict.get("Amount Received", tx_dict.get("amount_received",  0.0))),
                    "pay_cur" : str(tx_dict.get("Payment Currency",  tx_dict.get("payment_currency",  "US Dollar"))),
                    "recv_cur": str(tx_dict.get("Receiving Currency",tx_dict.get("receiving_currency","US Dollar"))),
                    "pf"      : str(tx_dict.get("Payment Format",   tx_dict.get("payment_format",   "Wire Transfer"))),
                    "ts"      : ts_str,
                    "from_bank": int(tx_dict.get("From Bank",       tx_dict.get("from_bank",         0))),
                    "to_bank" : int(tx_dict.get("To Bank",          tx_dict.get("to_bank",           0))),
                    "grp"     : fan_out_group,
                    "label"   : actual_label,
                    "gat_lvl" : gat_signal,
                    "gat_prob": gat_prob,
                    "bsig"    : behavior_signal,
                    "prev_out": int(tx_dict.get("previous_outgoing", 0)),
                    "prev_in" : int(tx_dict.get("previous_incoming", 0)),
                    "reason"  : f"Live inference: prob={gat_prob:.4f}"
                }
            )
            conn.commit()

        print(f"[FanOutUpdater] Inserted new transaction → group {fan_out_group}, label={actual_label}")
        return True

    except Exception as e:
        warnings.warn(f"[FanOutUpdater] Failed to insert transaction: {e}")
        return False


def _invalidate_dataset_cache():
    """
    Reset the in-memory DatasetManager singleton so the dashboard
    reloads from the database on next render.
    """
    try:
        import fraud_data
        fraud_data.DatasetManager._instance = None
        # Also clear Streamlit cache if available
        try:
            import streamlit as st
            fraud_data.get_transactions_df.clear()
            fraud_data.get_all_flagged_senders.clear()
        except Exception:
            pass
    except Exception as e:
        warnings.warn(f"[FanOutUpdater] Could not invalidate cache: {e}")
