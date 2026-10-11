"""
Server-side bulk recalculation of customer_profiles.
Sends aggregation subqueries to TiDB so there are NO per-account round trips.
Uses batches of account numbers in the IN clause.
"""
import mysql.connector

DB_HOST = "gateway01.ap-northeast-1.prod.aws.tidbcloud.com"
DB_PORT = 4000
DB_USER = "3JTMKeEP2m268Uk.root"
DB_PASSWORD = "KUOElkUyvNZX8HBL"
DB_NAME = "aml_fraud_db"

def p(msg):
    print(msg, flush=True)

def make_conn():
    return mysql.connector.connect(
        host=DB_HOST, port=DB_PORT,
        user=DB_USER, password=DB_PASSWORD,
        database=DB_NAME,
        ssl_verify_cert=True, ssl_verify_identity=True,
        connect_timeout=60,
        connection_timeout=60
    )

def run():
    p("Connecting to TiDB MySQL...")
    conn = make_conn()
    cursor = conn.cursor(dictionary=True)

    # Step 1: Get the top 2000 active fan-out groups
    p("Fetching top 2000 active fan-out group IDs...")
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

    # Step 2: Get all unique accounts involved in those groups
    ph = ','.join(['%s'] * len(top_groups))
    cursor.execute(f"""
        SELECT DISTINCT account_number FROM (
            SELECT from_account AS account_number FROM transactions WHERE fan_out_group IN ({ph})
            UNION
            SELECT to_account   AS account_number FROM transactions WHERE fan_out_group IN ({ph})
        ) AS combined
    """, tuple(top_groups) * 2)
    affected_accounts = [r['account_number'] for r in cursor.fetchall()]
    p(f"Found {len(affected_accounts)} affected accounts. Running server-side aggregation in batches...")
    cursor.close()
    conn.close()

    total = len(affected_accounts)
    done = 0
    BATCH = 300  # accounts per batch - server does all the math

    for batch_idx, i in enumerate(range(0, total, BATCH)):
        batch = affected_accounts[i:i + BATCH]
        ph_batch = ','.join(['%s'] * len(batch))

        # Reconnect for each batch to avoid connection drops
        conn = make_conn()
        cursor = conn.cursor()

        # Single UPDATE with aggregation subquery JOIN - all computation on the server
        sql = f"""
            UPDATE customer_profiles cp
            JOIN (
                SELECT
                    accs.account_number,
                    COALESCE(o.out_count, 0)  AS out_count,
                    COALESCE(o.out_total, 0)  AS out_total,
                    COALESCE(o.out_avg,   0)  AS out_avg,
                    COALESCE(o.out_max,   0)  AS out_max,
                    COALESCE(o.uniq_recv, 0)  AS uniq_recv,
                    COALESCE(inn.in_count, 0) AS in_count,
                    COALESCE(inn.in_total, 0) AS in_total,
                    COALESCE(inn.in_avg,   0) AS in_avg,
                    COALESCE(inn.in_max,   0) AS in_max,
                    COALESCE(inn.uniq_sndr,0) AS uniq_sndr,
                    COALESCE(inn.in_total, 0) - COALESCE(o.out_total, 0) AS net_flow,
                    COALESCE(o.out_count, 0) + COALESCE(inn.in_count, 0) AS total_degree,
                    CASE WHEN COALESCE(inn.in_count, 0) > 0
                         THEN COALESCE(o.uniq_recv, 0) / inn.in_count
                         ELSE COALESCE(o.uniq_recv, 0) END AS fan_out_ratio,
                    CASE WHEN COALESCE(inn.in_total, 0) > 0
                         THEN COALESCE(o.out_total, 0) / inn.in_total
                         ELSE 0 END AS pass_through_ratio
                FROM (SELECT column_value AS account_number FROM
                    (SELECT %s AS column_value {' UNION ALL SELECT %s' * (len(batch)-1)}) AS v
                ) AS accs
                LEFT JOIN (
                    SELECT from_account,
                        COUNT(*)                        AS out_count,
                        SUM(amount_paid)                AS out_total,
                        AVG(amount_paid)                AS out_avg,
                        MAX(amount_paid)                AS out_max,
                        COUNT(DISTINCT to_account)      AS uniq_recv
                    FROM transactions
                    WHERE from_account IN ({ph_batch})
                    GROUP BY from_account
                ) o ON accs.account_number = o.from_account
                LEFT JOIN (
                    SELECT to_account,
                        COUNT(*)                        AS in_count,
                        SUM(amount_received)            AS in_total,
                        AVG(amount_received)            AS in_avg,
                        MAX(amount_received)            AS in_max,
                        COUNT(DISTINCT from_account)    AS uniq_sndr
                    FROM transactions
                    WHERE to_account IN ({ph_batch})
                    GROUP BY to_account
                ) inn ON accs.account_number = inn.to_account
            ) AS agg ON cp.account_number = agg.account_number
            SET
                cp.outgoing_transactions   = agg.out_count,
                cp.total_outgoing_amount   = agg.out_total,
                cp.average_outgoing_amount = agg.out_avg,
                cp.maximum_outgoing_amount = agg.out_max,
                cp.unique_receivers        = agg.uniq_recv,
                cp.incoming_transactions   = agg.in_count,
                cp.total_incoming_amount   = agg.in_total,
                cp.average_incoming_amount = agg.in_avg,
                cp.maximum_incoming_amount = agg.in_max,
                cp.unique_senders          = agg.uniq_sndr,
                cp.net_flow                = agg.net_flow,
                cp.total_degree            = agg.total_degree,
                cp.fan_out_ratio           = agg.fan_out_ratio,
                cp.pass_through_ratio      = agg.pass_through_ratio
            WHERE cp.account_number IN ({ph_batch})
        """
        # params: batch values for the inline table, then batch for outgoing IN, then batch for incoming IN, then batch for WHERE IN
        params = tuple(batch) + tuple(batch) + tuple(batch) + tuple(batch)
        cursor.execute(sql, params)
        conn.commit()
        done += len(batch)
        pct = round(done / total * 100, 1)
        p(f"  Batch {batch_idx+1} | {done}/{total} accounts updated ({pct}%)")
        cursor.close()
        conn.close()

    p(f"DONE! {total} customer profiles fully recalculated — stats now match actual transactions.")

if __name__ == "__main__":
    run()
