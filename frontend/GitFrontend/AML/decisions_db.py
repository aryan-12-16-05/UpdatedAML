def _get_db_connection():
    """Helper to establish a connection to TiDB/MySQL."""
    import mysql.connector
    import streamlit as st

    DB_HOST = "gateway01.ap-northeast-1.prod.aws.tidbcloud.com"
    DB_PORT = 4000
    DB_USER = "3JTMKeEP2m268Uk.root"
    DB_PASSWORD = "KUOElkUyvNZX8HBL"
    DB_NAME = "aml_fraud_db"

    try:
        if hasattr(st, "secrets"):
            DB_HOST = st.secrets.get("DB_HOST", DB_HOST)
            DB_PORT = st.secrets.get("DB_PORT", DB_PORT)
            DB_USER = st.secrets.get("DB_USER", DB_USER)
            DB_PASSWORD = st.secrets.get("DB_PASSWORD", DB_PASSWORD)
            DB_NAME = st.secrets.get("DB_NAME", DB_NAME)
    except Exception:
        pass

    return mysql.connector.connect(
        host=DB_HOST,
        port=DB_PORT,
        user=DB_USER,
        password=DB_PASSWORD,
        database=DB_NAME,
        ssl_verify_cert=True,
        ssl_verify_identity=True,
        connect_timeout=15
    )

def _init_db_schema(conn):
    """Ensures analyst_decisions table and necessary columns exist."""
    cursor = conn.cursor()
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS analyst_decisions (
        tx_id VARCHAR(255) PRIMARY KEY,
        decision VARCHAR(255),
        notes TEXT,
        timestamp VARCHAR(255),
        account_number VARCHAR(255)
    )
    """)
    # Try adding account_number column if table already existed without it
    try:
        cursor.execute("ALTER TABLE analyst_decisions ADD COLUMN account_number VARCHAR(255)")
        conn.commit()
    except Exception:
        pass
    cursor.close()

def record_analyst_decision(tx_id, decision, notes, timestamp, account=None):
    """
    Saves an auditor decision to the MySQL database.
    Associates the decision with the sender account_number for rule-based overrides.
    """
    try:
        # If account is not provided, try resolving from tx_id/group
        if not account:
            account = _resolve_account_from_tx(tx_id)

        conn = _get_db_connection()
        _init_db_schema(conn)
        cursor = conn.cursor()

        sql = """
        INSERT INTO analyst_decisions (tx_id, decision, notes, timestamp, account_number)
        VALUES (%s, %s, %s, %s, %s)
        ON DUPLICATE KEY UPDATE 
            decision = VALUES(decision), 
            notes = VALUES(notes), 
            timestamp = VALUES(timestamp),
            account_number = IF(VALUES(account_number) IS NOT NULL, VALUES(account_number), account_number)
        """
        cursor.execute(sql, (str(tx_id), str(decision), str(notes), str(timestamp), str(account) if account else None))
        conn.commit()
        cursor.close()
        conn.close()
        print(f"Recorded decision for {tx_id} (Account: {account}) in MySQL.")
    except Exception as e:
        print(f"Error saving decision to MySQL: {e}")

def remove_analyst_decision(tx_id):
    """
    Removes a decision from the MySQL database.
    """
    try:
        conn = _get_db_connection()
        cursor = conn.cursor()
        alt_id = str(tx_id).replace("GROUP-", "") if "GROUP-" in str(tx_id) else f"GROUP-{tx_id}"
        sql = "DELETE FROM analyst_decisions WHERE tx_id = %s OR tx_id = %s"
        cursor.execute(sql, (str(tx_id), alt_id))
        conn.commit()
        cursor.close()
        conn.close()
        print(f"Removed decision for {tx_id} from MySQL.")
    except Exception as e:
        print(f"Error removing decision from MySQL: {e}")

def get_analyst_decisions():
    """
    Retrieves all decisions from the MySQL database.
    """
    decisions = {}
    try:
        conn = _get_db_connection()
        _init_db_schema(conn)
        cursor = conn.cursor(dictionary=True)
        cursor.execute("SELECT * FROM analyst_decisions")
        for row in cursor.fetchall():
            acc = row.get('account_number')
            tx_k = row['tx_id']
            decisions[tx_k] = {
                "decision": row['decision'],
                "notes": row['notes'],
                "timestamp": row['timestamp'],
                "account_number": str(acc).strip() if acc else None
            }
        cursor.close()
        conn.close()
    except Exception as e:
        print(f"Error retrieving decisions from MySQL: {e}")
    return decisions

def _resolve_account_from_tx(tx_id: str):
    """Lookup sender account for a group or tx_id from fraud data."""
    try:
        import fraud_data
        senders = fraud_data.get_all_flagged_senders()
        for s in senders:
            if s.get("tx_id") == tx_id or f"GROUP-{s.get('group_id')}" == tx_id:
                return str(s.get("account")).strip()
    except Exception:
        pass
    return None

def get_account_status(account_id: str) -> str:
    """
    Returns 'SAFE', 'FRAUD', or 'NULL' based on recorded analyst decisions.
    - 'SAFE'  if the analyst approved transactions/groups for this account
    - 'FRAUD' if the analyst blocked transactions/groups for this account
    - 'NULL'  if no decision or other status
    """
    if not account_id:
        return "NULL"
    
    acc_clean = str(account_id).strip()
    try:
        decisions = get_analyst_decisions()
        # Search decisions matching this account
        for tx_id, dec_info in decisions.items():
            dec_acc = str(dec_info.get("account_number") or "").strip()
            if not dec_acc:
                dec_acc = _resolve_account_from_tx(tx_id)
            
            if dec_acc and dec_acc == acc_clean:
                dec_text = dec_info.get("decision", "")
                if "Approve" in dec_text:
                    return "SAFE"
                elif "Block" in dec_text:
                    return "FRAUD"
                
    except Exception as e:
        print(f"Error getting account status for {account_id}: {e}")
        
    return "NULL"

def get_all_account_statuses() -> dict:
    """Returns mapping of {account_number: 'SAFE' | 'FRAUD' | 'NULL'}."""
    statuses = {}
    try:
        decisions = get_analyst_decisions()
        for tx_id, dec_info in decisions.items():
            acc = str(dec_info.get("account_number") or "").strip()
            if not acc:
                acc = _resolve_account_from_tx(tx_id)
            if acc:
                dec_text = dec_info.get("decision", "")
                if "Approve" in dec_text:
                    statuses[acc] = "SAFE"
                elif "Block" in dec_text:
                    statuses[acc] = "FRAUD"
                else:
                    statuses[acc] = "NULL"
    except Exception as e:
        print(f"Error getting all account statuses: {e}")
    return statuses
