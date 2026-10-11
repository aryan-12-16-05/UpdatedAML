import os
import glob
import pandas as pd
import mysql.connector
from sqlalchemy import create_engine

# TiDB connection details
DB_HOST = "gateway01.ap-northeast-1.prod.aws.tidbcloud.com"
DB_PORT = 4000
DB_USER = "3JTMKeEP2m268Uk.root"
DB_PASSWORD = "KUOElkUyvNZX8HBL"
DB_NAME = "aml_fraud_db"

def find_file(pattern, directory):
    matches = glob.glob(os.path.join(directory, pattern))
    if matches:
        return matches[0]
    return None

def migrate_data():
    print("Connecting to MySQL Server...")
    # Connect to MySQL Server (without specifying a database) to create it if needed
    try:
        conn = mysql.connector.connect(
            host=DB_HOST,
            port=DB_PORT,
            user=DB_USER,
            password=DB_PASSWORD,
            ssl_verify_cert=True,
            ssl_verify_identity=True
        )
        cursor = conn.cursor()
        cursor.execute(f"CREATE DATABASE IF NOT EXISTS {DB_NAME};")
        cursor.close()
        conn.close()
        print(f"Database '{DB_NAME}' ensured.")
    except Exception as e:
        print(f"Error connecting to MySQL: {e}")
        return

    # Use SQLAlchemy for pandas to_sql
    import urllib.parse
    encoded_password = urllib.parse.quote_plus(DB_PASSWORD)
    engine = create_engine(
        f"mysql+pymysql://{DB_USER}:{encoded_password}@{DB_HOST}:{DB_PORT}/{DB_NAME}",
        connect_args={"ssl": {"ssl_cert": None}} # PyMySQL basic SSL for TiDB
    )

    gitdata_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "GitData"))
    print(f"Looking for data in: {gitdata_dir}")

    # 1. Migrate Transactions
    trans_file = find_file("test_transactions.csv", gitdata_dir) or find_file("frontend (1).csv", gitdata_dir) or find_file("frontend*.csv", gitdata_dir)
    if trans_file:
        print(f"Migrating transactions from {trans_file}...")
        try:
            df_trans = pd.read_csv(trans_file)
            # Standardize column names (make them lowercase, replace spaces with underscores)
            df_trans.columns = [c.strip().lower().replace(" ", "_").replace("-", "_") for c in df_trans.columns]
            
            # Map column variations (e.g. Account -> from_account, Account.1 -> to_account)
            col_rename = {}
            if "account" in df_trans.columns and "from_account" not in df_trans.columns:
                col_rename["account"] = "from_account"
            if "account.1" in df_trans.columns and "to_account" not in df_trans.columns:
                col_rename["account.1"] = "to_account"
            elif "account_1" in df_trans.columns and "to_account" not in df_trans.columns:
                col_rename["account_1"] = "to_account"
            if col_rename:
                df_trans.rename(columns=col_rename, inplace=True)

            # Ensure GAT & Fanout columns exist for downstream queries
            if "episode_no" in df_trans.columns and "fan_out_group" not in df_trans.columns:
                df_trans["fan_out_group"] = df_trans["episode_no"].apply(
                    lambda x: int(x) if pd.notnull(x) and str(x).strip() not in ('', '-1', '-1.0') else None
                )
            if "gat_risk_level" not in df_trans.columns and "is_laundering" in df_trans.columns:
                df_trans["gat_risk_level"] = df_trans["is_laundering"].apply(lambda x: "HIGH" if x == 1 else "LOW")
            if "behavior_signal" not in df_trans.columns and "is_laundering" in df_trans.columns:
                df_trans["behavior_signal"] = df_trans["is_laundering"].apply(lambda x: "FAN-OUT" if x == 1 else "NORMAL")
            if "gat_risk_probability" not in df_trans.columns and "is_laundering" in df_trans.columns:
                df_trans["gat_risk_probability"] = df_trans["is_laundering"].apply(lambda x: 0.95 if x == 1 else 0.10)

            # Write to MySQL table in chunks with progress
            print(f"Writing {len(df_trans)} transaction rows to TiDB (chunksize=10000)...")
            df_trans.to_sql("transactions", con=engine, if_exists="replace", index=False, chunksize=10000)
            print("Successfully migrated 'transactions' table.")
        except Exception as e:
            print(f"Failed to migrate transactions: {e}")
    else:
        print("Transactions CSV not found!")

    # 2. Migrate Customer Profiles
    profiles_file = find_file("test_customer_profiles.csv", gitdata_dir) or find_file("customer_profiles.csv", gitdata_dir) or find_file("*profile*.csv", gitdata_dir)
    if profiles_file:
        print(f"Migrating customer profiles from {profiles_file}...")
        try:
            df_prof = pd.read_csv(profiles_file)
            df_prof.columns = [c.strip().lower().replace(" ", "_").replace("-", "_") for c in df_prof.columns]
            
            # Map column variations (e.g. account_id -> account_number)
            prof_rename = {}
            if "account_id" in df_prof.columns and "account_number" not in df_prof.columns:
                prof_rename["account_id"] = "account_number"
            if prof_rename:
                df_prof.rename(columns=prof_rename, inplace=True)

            print(f"Writing {len(df_prof)} profile rows to TiDB (chunksize=10000)...")
            df_prof.to_sql("customer_profiles", con=engine, if_exists="replace", index=False, chunksize=10000)
            print("Successfully migrated 'customer_profiles' table.")
        except Exception as e:
            print(f"Failed to migrate customer profiles: {e}")
    else:
        print("Customer Profiles CSV not found!")

    # 3. Create indices for fast lookup
    print("Creating indexes on TiDB tables...")
    try:
        from sqlalchemy import text
        with engine.connect() as conn:
            conn.execute(text("CREATE INDEX IF NOT EXISTS idx_tx_from_acc ON transactions (from_account(64));"))
            conn.execute(text("CREATE INDEX IF NOT EXISTS idx_tx_to_acc ON transactions (to_account(64));"))
            conn.execute(text("CREATE INDEX IF NOT EXISTS idx_tx_ts ON transactions (timestamp(32));"))
            conn.execute(text("CREATE INDEX IF NOT EXISTS idx_tx_fanout ON transactions (fan_out_group);"))
            conn.execute(text("CREATE INDEX IF NOT EXISTS idx_cp_acc ON customer_profiles (account_number(64));"))
            conn.commit()
            print("Indexes successfully created.")
    except Exception as e:
        print(f"Index creation notice: {e}")

    print("Migration complete!")

if __name__ == "__main__":
    migrate_data()

