"""
Fast fan-out amount redistributor using CASE WHEN bulk UPDATE.
One SQL statement per group = minimal round trips to TiDB.
"""
import mysql.connector
import random

DB_HOST = "gateway01.ap-northeast-1.prod.aws.tidbcloud.com"
DB_PORT = 4000
DB_USER = "3JTMKeEP2m268Uk.root"
DB_PASSWORD = "KUOElkUyvNZX8HBL"
DB_NAME = "aml_fraud_db"

def p(msg):
    print(msg, flush=True)

def redistribute(amounts):
    total = sum(amounts)
    n = len(amounts)
    if n <= 1:
        return amounts
    mean = total / n
    new = [round(mean * random.uniform(0.8, 1.2), 2) for _ in range(n - 1)]
    last = round(total - sum(new), 2)
    if last <= 0:
        return redistribute(amounts)
    new.append(last)
    return new

def run():
    p("Connecting to TiDB MySQL...")
    conn = mysql.connector.connect(
        host=DB_HOST, port=DB_PORT,
        user=DB_USER, password=DB_PASSWORD,
        database=DB_NAME,
        ssl_verify_cert=True, ssl_verify_identity=True,
        connect_timeout=30
    )
    cursor = conn.cursor(dictionary=True)

    p("Fetching top 2000 active fan-out groups...")
    cursor.execute("""
        SELECT DISTINCT fan_out_group
        FROM transactions
        WHERE UPPER(gat_risk_level) IN ('HIGH','MEDIUM')
          AND UPPER(behavior_signal) = 'FAN-OUT'
          AND fan_out_group IS NOT NULL
        LIMIT 2000
    """)
    top_groups = [r['fan_out_group'] for r in cursor.fetchall()]
    p(f"Found {len(top_groups)} groups.")

    placeholders = ','.join(['%s'] * len(top_groups))
    cursor.execute(
        f"SELECT transaction_id, fan_out_group, amount_paid FROM transactions WHERE fan_out_group IN ({placeholders})",
        tuple(top_groups)
    )
    rows = cursor.fetchall()
    p(f"Loaded {len(rows)} transactions. Building new amounts...")

    # Group by fan_out_group
    groups = {}
    for r in rows:
        groups.setdefault(r['fan_out_group'], []).append(r)

    # Build CASE WHEN map: transaction_id -> new_amount
    tx_to_new_amt = {}
    for gid, txs in groups.items():
        if len(txs) <= 1:
            continue
        amounts = [float(t['amount_paid']) for t in txs]
        new_amounts = redistribute(amounts)
        for tx, new_amt in zip(txs, new_amounts):
            tx_to_new_amt[tx['transaction_id']] = new_amt

    total = len(tx_to_new_amt)
    p(f"Ready to update {total} transactions using CASE WHEN bulk SQL (100 rows per statement)...")

    items = list(tx_to_new_amt.items())
    CHUNK = 100  # 100 rows per SQL statement - fast single round trip each
    done = 0
    total_chunks = (total + CHUNK - 1) // CHUNK

    for i in range(0, total, CHUNK):
        chunk = items[i:i + CHUNK]
        # Build: UPDATE transactions SET amount_paid = CASE WHEN transaction_id=X THEN Y ... END, amount_received = ... WHERE transaction_id IN (...)
        case_clauses = " ".join([f"WHEN %s THEN %s" for _ in chunk])
        in_placeholders = ",".join(["%s"] * len(chunk))

        sql = f"""
            UPDATE transactions
            SET amount_paid = CASE transaction_id {case_clauses} END,
                amount_received = CASE transaction_id {case_clauses} END
            WHERE transaction_id IN ({in_placeholders})
        """
        # params: case clause pairs * 2 (paid + received) + IN list
        params = []
        for tx_id, amt in chunk:
            params.extend([tx_id, amt])  # for amount_paid CASE
        for tx_id, amt in chunk:
            params.extend([tx_id, amt])  # for amount_received CASE
        for tx_id, amt in chunk:
            params.append(tx_id)  # for WHERE IN

        cursor.execute(sql, params)
        conn.commit()
        done += len(chunk)
        chunk_num = i // CHUNK + 1
        pct = round(done / total * 100, 1)
        p(f"  Chunk {chunk_num}/{total_chunks} done | {done}/{total} transactions ({pct}%)")

    cursor.close()
    conn.close()
    p(f"COMPLETE! All {total} transaction amounts redistributed successfully.")

if __name__ == "__main__":
    run()
