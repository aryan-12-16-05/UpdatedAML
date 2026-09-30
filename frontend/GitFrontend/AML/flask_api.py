"""
Flask REST API Backend for AML Fraud Detection System
Endpoints:
  GET  /api/transactions           — all transactions
  GET  /api/flagged_senders        — all high-risk fan-out groups
  GET  /api/customer/<account_id>  — customer profile
  GET  /api/decisions              — analyst decisions
  POST /api/decision               — record analyst decision
  POST /api/ingest_transaction     — NEW: live transaction scoring + DB ingestion

POST /api/ingest_transaction
  Request body (JSON):
    {
      "Transaction ID"     : "TX-20241001-001",   # optional; auto-generated if omitted
      "From Account"       : "100428660",
      "To Account"         : "80D2819B0",
      "From Bank"          : 70,
      "To Bank"            : 213952,
      "Amount Paid"        : 1500.00,
      "Amount Received"    : 1500.00,
      "Payment Currency"   : "US Dollar",
      "Receiving Currency" : "US Dollar",
      "Payment Format"     : "Wire Transfer",
      "Timestamp"          : "2024-10-01 14:30:00",  # optional; defaults to now
      "previous_outgoing"  : 12,
      "previous_incoming"  : 3
    }
  Response (JSON):
    {
      "transaction_id"  : "TX-20241001-001",
      "from_account"    : "100428660",
      "probability"     : 0.912345,
      "is_fraud"        : true,
      "gat_signal"      : "HIGH RISK",
      "fan_out_group"   : 42,
      "retroactive_update" : {
          "triggered"   : true,
          "updated_rows": 7
      },
      "graph_info"      : { "num_nodes": 9, "num_edges": 8 },
      "db_inserted"     : true
    }
"""

from flask import Flask, jsonify, request
from flask_cors import CORS
import fraud_data
import decisions_db

app = Flask(__name__)
CORS(app)


# ─────────────────────────────────────────────────────────────────────────────
#  EXISTING ENDPOINTS
# ─────────────────────────────────────────────────────────────────────────────

@app.route('/api/transactions', methods=['GET'])
def get_transactions():
    df = fraud_data.get_transactions_df()
    return jsonify(df.to_dict(orient='records'))


@app.route('/api/flagged_senders', methods=['GET'])
def get_flagged_senders():
    groups = fraud_data.get_all_flagged_senders()
    return jsonify(groups)


@app.route('/api/customer/<account_id>', methods=['GET'])
def get_customer(account_id):
    profile = fraud_data.get_customer_profile(account_id)
    if not profile:
        return jsonify({"error": "Profile not found"}), 404
    return jsonify(profile)


@app.route('/api/decisions', methods=['GET'])
def get_decisions():
    decisions = decisions_db.get_analyst_decisions()
    return jsonify(decisions)


@app.route('/api/decision', methods=['POST'])
def post_decision():
    data = request.json
    if not data or not all(k in data for k in ("tx_id", "decision", "notes", "timestamp")):
        return jsonify({"error": "Missing required fields"}), 400

    decisions_db.record_analyst_decision(
        data["tx_id"], data["decision"], data["notes"], data["timestamp"]
    )
    return jsonify({"status": "success", "message": f"Recorded decision for {data['tx_id']}"})


# ─────────────────────────────────────────────────────────────────────────────
#  NEW: LIVE TRANSACTION INGESTION + GAT INFERENCE
# ─────────────────────────────────────────────────────────────────────────────

@app.route('/api/ingest_transaction', methods=['POST'])
def ingest_transaction():
    """
    Accept a new transaction, score it with the GAT model,
    insert it into the database, and retroactively update all
    prior transactions from the same sender if fraud is detected.
    """
    from datetime import datetime
    import pandas as pd

    # ── 1. Parse + validate input ─────────────────────────────────────────────
    data = request.json
    if not data:
        return jsonify({"error": "Request body must be JSON"}), 400

    required_fields = ["From Account", "To Account", "Amount Paid"]
    missing = [f for f in required_fields if f not in data and f.lower().replace(" ", "_") not in data]
    if missing:
        return jsonify({"error": f"Missing required fields: {missing}"}), 400

    # Set defaults
    if "Timestamp" not in data and "timestamp" not in data:
        data["Timestamp"] = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
    if "Amount Received" not in data and "amount_received" not in data:
        data["Amount Received"] = data.get("Amount Paid", data.get("amount_paid", 0.0))
    if "Transaction ID" not in data and "transaction_id" not in data:
        data["Transaction ID"] = f"LIVE-{datetime.utcnow().strftime('%Y%m%d%H%M%S%f')}"

    sender_account = str(data.get("From Account", data.get("from_account", ""))).strip()
    tx_id          = str(data.get("Transaction ID", data.get("transaction_id", "")))

    # ── 2. Fetch sender's historical transactions from database ───────────────
    try:
        dm          = fraud_data.get_dm()
        full_df     = dm.get_df()
        profiles_df = dm._profiles_df if hasattr(dm, "_profiles_df") else pd.DataFrame()

        from_col = "From Account" if "From Account" in full_df.columns else "from_account"
        history_df = full_df[
            full_df[from_col].astype(str).str.strip() == sender_account
        ].copy()

    except Exception as e:
        return jsonify({"error": f"Failed to fetch sender history: {str(e)}"}), 500

    # ── 3. Run GAT inference ──────────────────────────────────────────────────
    try:
        import inference_engine
        result = inference_engine.score_transaction(data, history_df, profiles_df)
    except Exception as e:
        return jsonify({"error": f"Inference failed: {str(e)}"}), 500

    prob       = result["probability"]
    is_fraud   = result["is_fraud"]
    gat_signal = result["gat_signal"]

    # ── 4. Determine fan-out group ────────────────────────────────────────────
    fan_out_group = None
    if not history_df.empty:
        grp_col = "Fan-out Group" if "Fan-out Group" in history_df.columns else "fan_out_group"
        if grp_col in history_df.columns:
            existing_grp = history_df[grp_col].dropna()
            if not existing_grp.empty:
                fan_out_group = int(existing_grp.iloc[0])

    # ── 5. Insert new transaction into database ───────────────────────────────
    import fanout_updater

    if fan_out_group is None:
        # Need a new group ID — get from DB
        try:
            engine = fanout_updater._get_engine()
            from sqlalchemy import text
            with engine.connect() as conn:
                fan_out_group = fanout_updater._assign_new_group(conn)
        except Exception:
            fan_out_group = 999999  # fallback

    db_inserted = fanout_updater.insert_new_transaction(
        data, prob, gat_signal, is_fraud, fan_out_group
    )

    # ── 6. Retroactive labeling if fraud detected ─────────────────────────────
    retro_result = {"triggered": False, "updated_rows": 0}
    if is_fraud and not history_df.empty:
        retro = fanout_updater.retroactively_label_sender(
            sender_account, prob, fan_out_group
        )
        retro_result = {
            "triggered"   : True,
            "updated_rows": retro.get("updated_rows", 0),
            "success"     : retro.get("success", False)
        }

    # ── 7. Return full response ────────────────────────────────────────────────
    return jsonify({
        "transaction_id"     : tx_id,
        "from_account"       : sender_account,
        "to_account"         : str(data.get("To Account", data.get("to_account", ""))),
        "probability"        : prob,
        "is_fraud"           : is_fraud,
        "gat_signal"         : gat_signal,
        "fan_out_group"      : fan_out_group,
        "retroactive_update" : retro_result,
        "graph_info"         : {
            "num_nodes": result.get("num_nodes", 0),
            "num_edges": result.get("num_edges", 0)
        },
        "db_inserted"        : db_inserted
    })


# ─────────────────────────────────────────────────────────────────────────────
#  HEALTH CHECK
# ─────────────────────────────────────────────────────────────────────────────

@app.route('/api/health', methods=['GET'])
def health():
    return jsonify({"status": "ok", "service": "AML Fraud Detection API"})


if __name__ == '__main__':
    print("Starting AML Fraud API Backend...")
    app.run(port=5000, debug=True)
