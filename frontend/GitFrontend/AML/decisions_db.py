
# --- Analyst Decisions Database ---

def record_analyst_decision(tx_id, decision, notes, timestamp):
    """
    Saves an auditor decision to the MySQL database.
    """
    import mysql.connector
    try:
        import streamlit as st
        DB_HOST = st.secrets.get("DB_HOST", "gateway01.ap-northeast-1.prod.aws.tidbcloud.com")
        DB_PORT = st.secrets.get("DB_PORT", 4000)
        DB_USER = st.secrets.get("DB_USER", "3JTMKeEP2m268Uk.root")
        DB_PASSWORD = st.secrets.get("DB_PASSWORD", "KUOElkUyvNZX8HBL")
        DB_NAME = st.secrets.get("DB_NAME", "aml_fraud_db")
        
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
        cursor = conn.cursor()
        
        sql = """
        INSERT INTO analyst_decisions (tx_id, decision, notes, timestamp)
        VALUES (%s, %s, %s, %s)
        ON DUPLICATE KEY UPDATE 
            decision = VALUES(decision), 
            notes = VALUES(notes), 
            timestamp = VALUES(timestamp)
        """
        cursor.execute(sql, (tx_id, decision, notes, timestamp))
        conn.commit()
        cursor.close()
        conn.close()
        print(f"Recorded decision for {tx_id} in MySQL.")
    except Exception as e:
        print(f"Error saving decision to MySQL: {e}")

def get_analyst_decisions():
    """
    Retrieves all decisions from the MySQL database.
    """
    import mysql.connector
    decisions = {}
    try:
        import streamlit as st
        DB_HOST = st.secrets.get("DB_HOST", "gateway01.ap-northeast-1.prod.aws.tidbcloud.com")
        DB_PORT = st.secrets.get("DB_PORT", 4000)
        DB_USER = st.secrets.get("DB_USER", "3JTMKeEP2m268Uk.root")
        DB_PASSWORD = st.secrets.get("DB_PASSWORD", "KUOElkUyvNZX8HBL")
        DB_NAME = st.secrets.get("DB_NAME", "aml_fraud_db")
        
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
        cursor.execute("SELECT * FROM analyst_decisions")
        for row in cursor.fetchall():
            decisions[row['tx_id']] = {
                "decision": row['decision'],
                "notes": row['notes'],
                "timestamp": row['timestamp']
            }
        cursor.close()
        conn.close()
    except Exception as e:
        print(f"Error retrieving decisions from MySQL: {e}")
    return decisions
