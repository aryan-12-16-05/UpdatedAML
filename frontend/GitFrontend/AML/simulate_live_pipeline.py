import requests
import time
import pandas as pd
from datetime import datetime, timedelta
import urllib.parse
from sqlalchemy import create_engine, text
import math

API_URL = "http://localhost:5000/api/ingest_transaction"

SIM_RUN_ID = f"LIVE_SIM_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
MALICIOUS_SENDER = f"{SIM_RUN_ID}_SENDER"

DB_HOST = "gateway01.ap-northeast-1.prod.aws.tidbcloud.com"
DB_PORT = 4000
DB_USER = "3JTMKeEP2m268Uk.root"
DB_PASSWORD = "KUOElkUyvNZX8HBL"
DB_NAME = "aml_fraud_db"

def get_engine():
    encoded_password = urllib.parse.quote_plus(DB_PASSWORD)
    engine = create_engine(
        f"mysql+pymysql://{DB_USER}:{encoded_password}@{DB_HOST}:{DB_PORT}/{DB_NAME}",
        connect_args={"ssl": {"ssl_cert": None}, "connect_timeout": 15}
    )
    return engine

def get_db_count(engine, tx_prefix):
    with engine.connect() as conn:
        res = conn.execute(
            text("SELECT COUNT(*) FROM transactions WHERE transaction_id LIKE :prefix"),
            {"prefix": f"{tx_prefix}%"}
        ).fetchone()
        return res[0]

def get_sender_history(engine, sender, target_ts_str):
    with engine.connect() as conn:
        res = conn.execute(
            text("SELECT * FROM transactions WHERE from_account = :sender AND timestamp < :ts"),
            {"sender": sender, "ts": target_ts_str}
        ).fetchall()
        return res

def generate_transactions():
    csv_path = r"C:\Users\aryan\frontend\GitData\test_transactions.csv"
    print(f"Loading real test transactions from {csv_path}...")
    df = pd.read_csv(csv_path)
    
    # 1. 10 real laundering transactions from episode 51
    episodes = df[df["Is Laundering"] == 1]["episode_no"].value_counts()
    target_ep = 51 if 51 in episodes.index else episodes.index[0]
    df_laund = df[df["episode_no"] == target_ep].head(10).copy()
    
    # 2. 10 real legitimate transactions
    df_norm = df[df["Is Laundering"] == 0].head(10).copy()
    
    base_time = datetime.utcnow()
    transactions = []
    
    # Process normal transactions
    for i, (_, row) in enumerate(df_norm.iterrows(), start=1):
        tx = {
            "Transaction ID": f"{SIM_RUN_ID}_NORM_{str(i).zfill(2)}_{row['transaction_id']}",
            "From Account": str(row["Account"]).strip(),
            "To Account": str(row["Account.1"]).strip(),
            "Amount Paid": round(float(row.get("Amount Paid USD", row.get("Amount Paid", 1500.0))), 2),
            "Amount Received": round(float(row.get("Amount Received USD", row.get("Amount Received", 1500.0))), 2),
            "Payment Format": str(row.get("Payment Format", "ACH")),
            "Payment Currency": str(row.get("Payment Currency", "US Dollar")),
            "Receiving Currency": str(row.get("Receiving Currency", "US Dollar")),
            "From Bank": int(row.get("From Bank", 100)),
            "To Bank": int(row.get("To Bank", 200)),
            "Timestamp": (base_time + timedelta(minutes=i*2)).strftime("%Y-%m-%d %H:%M:%S"),
            "type": "normal"
        }
        transactions.append(tx)
        
    # Process laundering transactions
    for i, (_, row) in enumerate(df_laund.iterrows(), start=1):
        tx = {
            "Transaction ID": f"{SIM_RUN_ID}_LAUND_{str(i).zfill(2)}_{row['transaction_id']}",
            "From Account": str(row["Account"]).strip(),
            "To Account": str(row["Account.1"]).strip(),
            "Amount Paid": round(float(row.get("Amount Paid USD", row.get("Amount Paid", 9900.0))), 2),
            "Amount Received": round(float(row.get("Amount Received USD", row.get("Amount Received", 9900.0))), 2),
            "Payment Format": str(row.get("Payment Format", "ACH")),
            "Payment Currency": str(row.get("Payment Currency", "US Dollar")),
            "Receiving Currency": str(row.get("Receiving Currency", "US Dollar")),
            "From Bank": int(row.get("From Bank", 100)),
            "To Bank": int(row.get("To Bank", 200)),
            "Timestamp": (base_time + timedelta(minutes=i*2 + 1)).strftime("%Y-%m-%d %H:%M:%S"),
            "type": "malicious"
        }
        transactions.append(tx)
        
    # Sort strictly by timestamp to interleave
    transactions.sort(key=lambda x: x["Timestamp"])
    return transactions

def run_simulation():
    print(f"============================================================")
    print(f"STARTING LIVE PIPELINE SIMULATION (REAL TEST TRANSACTIONS)")
    print(f"RUN ID: {SIM_RUN_ID}")
    print(f"============================================================\n")
    
    engine = get_engine()
    transactions = generate_transactions()
    
    report = []
    
    for idx, tx_data in enumerate(transactions, start=1):
        tx_type = tx_data.pop("type")
        tx_id = tx_data["Transaction ID"]
        sender = tx_data["From Account"]
        target_ts = tx_data["Timestamp"]
        
        print(f"--- TX{str(idx).zfill(2)}: {tx_id} ({tx_type}) ---")
        print(f"Timestamp: {target_ts}")
        
        # Stream Database Barrier (Phase 6)
        count_before = get_db_count(engine, SIM_RUN_ID)
        print(f"DB simulation records before: {count_before}")
        
        if count_before != (idx - 1):
            print(f"CRITICAL ERROR: DB count before TX{idx} is {count_before}, expected {idx - 1}. STOPPING.")
            return

        # Verification of current history (Phase 10)
        history_in_db = get_sender_history(engine, sender, target_ts)
        print(f"Source historical rows in DB for {sender} before TX: {len(history_in_db)}")

        # Post Request
        print(f"Sending POST request...")
        req_start = datetime.utcnow()
        try:
            resp = requests.post(API_URL, json=tx_data)
        except Exception as e:
            print(f"POST failed: {e}")
            return
            
        req_end = datetime.utcnow()
        print(f"Received Response in {(req_end - req_start).total_seconds():.3f}s")
        
        if resp.status_code != 200:
            print(f"API Error {resp.status_code}: {resp.text}")
            return
            
        result = resp.json()
        prob = result.get("probability", 0.0)
        signal = result.get("gat_signal", "")
        retro = result.get("retroactive_update", {})
        graph_info = result.get("graph_info", {})
        
        history_visible = graph_info.get("history_visible", -1)
        
        print(f"Result: {prob:.6f} | {signal}")
        if retro.get("triggered"):
             print(f"RETROACTIVE SWEEP TRIGGERED! Updated {retro.get('updated_rows')} past rows. Group ID: {result.get('fan_out_group')}")
        
        # Verify DB (Phase 6)
        count_after = get_db_count(engine, SIM_RUN_ID)
        print(f"DB simulation records after: {count_after}")
        
        if count_after != idx:
            print(f"CRITICAL ERROR: DB count after TX{idx} is {count_after}, expected {idx}. STOPPING.")
            return
            
        # Log for report
        report.append({
            "tx_id": tx_id,
            "type": tx_type,
            "idx": idx,
            "before": count_before,
            "after": count_after,
            "history_visible": history_visible,
            "prob": prob,
            "signal": signal,
            "retro": retro.get("triggered", False)
        })
        
        print("\n")
        
    print(f"============================================================")
    print(f"LIVE STREAMING CAUSALITY REPORT")
    print(f"============================================================")
    
    malicious_idx = 1
    for r in report:
        if r["type"] == "malicious":
            prefix = f"MALICIOUS TX{str(malicious_idx).zfill(2)}"
            malicious_idx += 1
        else:
            prefix = f"NORMAL TX   "
            
        print(f"TX{str(r['idx']).zfill(2)} ({prefix}):")
        print(f"DB simulation records before = {r['before']}")
        print(f"DB simulation records after  = {r['after']}")
        print(f"history visible to inference = {r['history_visible']}")
        print(f"Probability = {r['prob']:.6f} | {r['signal']}")
        print(f"Retroactive sweep = {r['retro']}\n")

if __name__ == "__main__":
    run_simulation()
