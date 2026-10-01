import pandas as pd
import numpy as np
import mysql.connector
import os
import random

# DB connection details
DB_HOST = "gateway01.ap-northeast-1.prod.aws.tidbcloud.com"
DB_PORT = 4000
DB_USER = "3JTMKeEP2m268Uk.root"
DB_PASSWORD = "KUOElkUyvNZX8HBL"
DB_NAME = "aml_fraud_db"

def redistribute_amounts(amounts):
    """
    Given a list of amounts, sums them and redistributes the total
    roughly equally, varying by +/- 20% around the mean.
    Returns a list of new amounts.
    """
    total = sum(amounts)
    n = len(amounts)
    if n == 0:
        return []
    if n == 1:
        return [total]
    
    mean = total / n
    new_amounts = []
    
    # Generate N-1 amounts
    for _ in range(n - 1):
        # random around mean +/- 20%
        val = mean * random.uniform(0.8, 1.2)
        new_amounts.append(round(val, 2))
    
    # The last amount takes whatever is left to preserve the total exact sum
    last_val = total - sum(new_amounts)
    # Just in case last_val goes negative (extremely rare but possible if all previous are 1.2x)
    if last_val <= 0:
        # Re-distribute strictly
        return redistribute_amounts(amounts)
        
    new_amounts.append(round(last_val, 2))
    return new_amounts

def update_mysql():
    print("Connecting to TiDB MySQL...")
    conn = mysql.connector.connect(
        host=DB_HOST,
        port=DB_PORT,
        user=DB_USER,
        password=DB_PASSWORD,
        database=DB_NAME,
        ssl_verify_cert=True,
        ssl_verify_identity=True,
        connect_timeout=15
    )
    cursor = conn.cursor(dictionary=True)
    
    print("Fetching fan-out groups from MySQL...")
    # Fetch all fan-out transactions
    cursor.execute("""
        SELECT transaction_id, fan_out_group, amount_paid, amount_received 
        FROM transactions 
        WHERE UPPER(behavior_signal) = 'FAN-OUT' AND fan_out_group IS NOT NULL
    """)
    rows = cursor.fetchall()
    
    # Group by fan_out_group
    groups = {}
    for r in rows:
        gid = r['fan_out_group']
        if gid not in groups:
            groups[gid] = []
        groups[gid].append(r)
    
    print(f"Found {len(groups)} fan-out groups. Redistributing amounts...")
    
    update_data = []
    for gid, txs in groups.items():
        if len(txs) <= 1:
            continue
            
        amounts = [float(t['amount_paid']) for t in txs]
        new_amounts = redistribute_amounts(amounts)
        
        for tx, new_amt in zip(txs, new_amounts):
            # We also set amount_received to new_amt to keep it simple, 
            # or we could keep the same ratio. Since it's a fan-out, usually paid = received (approx)
            update_data.append((new_amt, new_amt, tx['transaction_id']))
    
    print(f"Updating {len(update_data)} rows in MySQL...")
    
    # Batch update
    update_query = """
        UPDATE transactions 
        SET amount_paid = %s, amount_received = %s 
        WHERE transaction_id = %s
    """
    cursor.executemany(update_query, update_data)
    conn.commit()
    
    cursor.close()
    conn.close()
    print("MySQL update complete.")

def update_csv():
    csv_path = r"C:\Users\aryan\frontend\GitData\frontend (1).csv"
    if not os.path.exists(csv_path):
        print(f"CSV file not found: {csv_path}")
        return
        
    print(f"Loading CSV from {csv_path} (This might take a moment...)")
    df = pd.read_csv(csv_path)
    
    print("Filtering fan-out groups in CSV...")
    # Identify fan-out rows
    is_fanout = df['behavior_signal'].str.upper() == 'FAN-OUT'
    has_group = df['fan_out_group'].notna()
    
    fanout_df = df[is_fanout & has_group].copy()
    
    print("Redistributing amounts in CSV...")
    for gid, group_df in fanout_df.groupby('fan_out_group'):
        n = len(group_df)
        if n <= 1:
            continue
        
        amounts = group_df['amount_paid'].astype(float).tolist()
        new_amounts = redistribute_amounts(amounts)
        
        df.loc[group_df.index, 'amount_paid'] = new_amounts
        df.loc[group_df.index, 'amount_received'] = new_amounts
        
    print("Saving updated CSV...")
    df.to_csv(csv_path, index=False)
    print("CSV update complete.")

if __name__ == "__main__":
    update_mysql()
    update_csv()
    print("All tasks finished successfully.")
