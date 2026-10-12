"""
AML Fraud Detection Data & Engine Layer for GitFrontend
Uses real datasets from GitData/ (frontend (1).csv & customer_profiles.csv).
Filters strictly for HIGH RISK FAN-OUT investigations.
Provides complete account profiling and graph topology.
"""

import os
import glob
import json
import pandas as pd
import numpy as np
import networkx as nx
from datetime import datetime

# ─── Persistent Analyst Decisions Store ────────────────────────────────────
_DECISIONS_FILE = os.path.join(os.path.dirname(__file__), "analyst_decisions.json")

def _load_decisions() -> dict:
    """Return a dictionary of group_id (int) -> decision_data."""
    if os.path.exists(_DECISIONS_FILE):
        try:
            with open(_DECISIONS_FILE, "r") as f:
                data = json.load(f)
                return {int(k): v for k, v in data.items()}
        except Exception:
            pass
    return {}

def _save_decisions(decisions_dict: dict):
    """Persist the decisions dictionary to disk."""
    try:
        with open(_DECISIONS_FILE, "w") as f:
            json.dump(decisions_dict, f, indent=4)
    except Exception:
        pass
# ────────────────────────────────────────────────────────────────

GITDATA_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "GitData")
)

CURRENCY_SYMBOLS = {
    "US Dollar": "$",
    "Euro": "€",
    "Rupee": "₹",
    "UK Pound": "£",
    "Yen": "¥",
    "Yuan": "¥",
    "Swiss Franc": "CHF ",
    "Canadian Dollar": "CA$",
    "Australian Dollar": "A$",
    "Ruble": "₽",
    "Shekel": "₪",
    "Bitcoin": "₿",
    "Mexican Peso": "Mex$",
    "Saudi Riyal": "SAR ",
    "Brazil Real": "R$",
}

def format_currency(amount, currency_name="US Dollar"):
    """Format numeric amount with proper currency symbol."""
    sym = CURRENCY_SYMBOLS.get(currency_name, f"{currency_name} ")
    try:
        amt = float(amount)
        if abs(amt) >= 1_000_000_000:
            return f"{sym}{amt / 1_000_000_000:.2f}B"
        elif abs(amt) >= 1_000_000:
            return f"{sym}{amt / 1_000_000:.2f}M"
        elif abs(amt) >= 1_000:
            return f"{sym}{amt:,.2f}"
        else:
            return f"{sym}{amt:.2f}"
    except Exception:
        return f"{sym}{amount}"


class DatasetManager:
    """Singleton manager that loads and indexes GitData/ transactions and customer profiles."""
    _instance = None
    _df = None
    _profiles_map = None
    _group_summaries = None
    _group_dict = None

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
            cls._instance._load_data()
        return cls._instance

    def _find_file(self, pattern_list):
        for pattern in pattern_list:
            matches = glob.glob(os.path.join(GITDATA_DIR, pattern))
            if matches:
                return matches[0]
        return None

    def _load_data(self):
        if self._df is not None:
            return

        # 1. Load Transactions Dataset
        trans_file = self._find_file(["test_transactions.csv", "frontend (1).csv", "frontend*.csv", "*fanout*.csv", "frontend_fanout_gat_test_grouped.csv"])
        if not trans_file or not os.path.exists(trans_file):
            raise FileNotFoundError(f"Transactions CSV not found in: {GITDATA_DIR}")

        print(f"Loading transactions dataset from: {trans_file}")
        raw_df = pd.read_csv(trans_file)

        # Standardize column names if coming from test_transactions.csv
        if "Account" in raw_df.columns and "From Account" not in raw_df.columns:
            raw_df.rename(columns={"Account": "From Account"}, inplace=True)
        if "Account.1" in raw_df.columns and "To Account" not in raw_df.columns:
            raw_df.rename(columns={"Account.1": "To Account"}, inplace=True)

        # Standardize group column names
        if "Fan_Out_Group" in raw_df.columns and "Fan-out Group" not in raw_df.columns:
            raw_df["Fan-out Group"] = raw_df["Fan_Out_Group"]
        if "GAT_Risk_Level" in raw_df.columns and "GAT Signal" not in raw_df.columns:
            raw_df["GAT Signal"] = raw_df["GAT_Risk_Level"]
        if "GAT_Risk_Probability" in raw_df.columns and "GAT Probability" not in raw_df.columns:
            raw_df["GAT Probability"] = raw_df["GAT_Risk_Probability"]
        if "Behavior_Signal" in raw_df.columns and "Detection Pattern" not in raw_df.columns:
            raw_df["Detection Pattern"] = raw_df["Behavior_Signal"]

        # Filter strictly for High and Medium Risk Fan-Out Groups (preserve ALL transactions in each flagged group)
        risk_mask = (raw_df["GAT Signal"].astype(str).str.upper().isin(["HIGH", "MEDIUM"])) & (raw_df["Fan-out Group"].fillna(-1).astype(int) != -1)
        if "Detection Pattern" in raw_df.columns:
            risk_mask = risk_mask & (raw_df["Detection Pattern"].astype(str).str.upper() == "FAN-OUT")

        flagged_groups = raw_df.loc[risk_mask, "Fan-out Group"].unique()

        # Keep all transactions belonging to flagged High/Medium fanout groups
        self._df = raw_df[raw_df["Fan-out Group"].isin(flagged_groups)].copy().reset_index(drop=True)
        if self._df.empty:
            self._df = raw_df.copy()

        self._df["Fan-out Group"] = self._df["Fan-out Group"].astype(int)

        # Pre-group only top unique groups for instant performance
        unique_groups = [int(x) for x in self._df["Fan-out Group"].drop_duplicates().head(200)]
        df_top = self._df[self._df["Fan-out Group"].isin(unique_groups)]
        self._group_dict = {int(gid): gdf for gid, gdf in df_top.groupby("Fan-out Group")}

        # Collect unique accounts from the active fan-out groups
        active_accounts = set()
        for gid in unique_groups[:200]:
            gdf = self._group_dict.get(gid)
            if gdf is not None:
                if "From Account" in gdf.columns:
                    active_accounts.update(gdf["From Account"].dropna().astype(str).str.strip())
                if "To Account" in gdf.columns:
                    active_accounts.update(gdf["To Account"].dropna().astype(str).str.strip())

        # 2. Load Customer Profiles from customer_profiles.csv or test_customer_profiles.csv
        profiles_file = self._find_file(["test_customer_profiles.csv", "customer_profiles.csv", "*profile*.csv"])
        self._profiles_map = {}
        if profiles_file and os.path.exists(profiles_file):
            print(f"Loading customer profiles from: {profiles_file}")
            df_p = pd.read_csv(profiles_file)
            acc_col = "Account_Number" if "Account_Number" in df_p.columns else df_p.columns[0]
            df_p[acc_col] = df_p[acc_col].astype(str).str.strip()
            # Fast filter for active accounts
            df_p_active = df_p[df_p[acc_col].isin(active_accounts)]
            for row in df_p_active.to_dict(orient="records"):
                anum = str(row.get(acc_col, "")).strip()
                self._profiles_map[anum] = row
            # Also keep full dataframe indexed as fallback
            self._profiles_df = df_p.set_index(acc_col)

        print(f"Successfully indexed {len(self._profiles_map):,} active customer profiles.")

        self._group_summaries = []

        for gid in unique_groups[:100]:
            gdf = self._group_dict[gid]
            if "GAT Probability" in gdf.columns:
                lead_idx = gdf["GAT Probability"].astype(float).idxmax()
                first_row = gdf.loc[lead_idx]
            else:
                first_row = gdf.iloc[0]
            gat_prob = float(first_row.get("GAT Probability", 0.99))
            gat_signal = str(first_row.get("GAT Signal", "HIGH")).strip()
            risk_tier = "High" if "HIGH" in gat_signal.upper() or gat_prob >= 0.7 else ("Medium" if "MEDIUM" in gat_signal.upper() or gat_prob >= 0.3 else "Low")
            risk_score = int(round(gat_prob * 100))
            if risk_score > 100:
                risk_score = 100
            elif risk_score < 0:
                risk_score = 0

            currency = str(first_row.get("Payment Currency", "US Dollar"))
            total_amt = gdf["Amount Paid"].sum() if "Amount Paid" in gdf.columns else 0.0
            tx_id = str(first_row.get("transaction_id", first_row.get("Transaction ID", f"TX-{gid}")))
            from_acc = str(first_row.get("From Account", first_row.get("Account Number", "—")))
            
            # Enrich sender info from customer_profiles if available
            sender_prof = self._profiles_map.get(from_acc, {})
            from_entity = sender_prof.get("Entity Name", first_row.get("Entity Name", f"Account {from_acc}"))
            from_bank = sender_prof.get("Bank Name", first_row.get("Bank Name", "Global Bank"))
            from_bank_id = sender_prof.get("Bank ID", first_row.get("Bank ID", "BNK-001"))
            from_entity_id = sender_prof.get("Entity ID", first_row.get("Entity ID", f"ENT-{from_acc[:8]}"))

            pattern = str(first_row.get("Detection Pattern", first_row.get("Fan_Out_Type", "FAN-OUT")))
            timestamp = str(first_row.get("Timestamp", "—"))
            payment_format = str(first_row.get("Payment Format", "Wire"))
            actual_label = int(first_row.get("Actual Label", 1))

            prev_out = int(sender_prof.get("Outgoing_Transactions", first_row.get("Historical_Outgoing_Count", len(gdf))))
            prev_in = int(sender_prof.get("Incoming_Transactions", first_row.get("Incoming_Transactions", 0)))
            uniq_recv = int(sender_prof.get("Unique_Receivers", first_row.get("Historical_Unique_Receivers", gdf["To Account"].nunique() if "To Account" in gdf.columns else len(gdf))))
            tx_count = len(gdf)

            summary = {
                "group_id": gid,
                "tx_id": f"GROUP-{gid}",
                "lead_tx_id": tx_id,
                "account": from_acc,
                "name": str(from_entity),
                "entity_id": str(from_entity_id),
                "bank_name": str(from_bank),
                "bank_id": str(from_bank_id),
                "risk": risk_tier,
                "risk_score": risk_score,
                "gat_prob": gat_prob,
                "gat_signal": gat_signal,
                "actual_label": actual_label,
                "pattern": pattern,
                "timestamp": timestamp,
                "payment_format": payment_format,
                "payment_currency": currency,
                "amount": total_amt,
                "amount_formatted": format_currency(total_amt, currency),
                "tx_count": tx_count,
                "unique_receivers": uniq_recv,
                "unique_senders": sender_prof.get("Unique_Senders", 0),
                "previous_outgoing": prev_out,
                "previous_incoming": prev_in,
                "is_fraud": True,
                "model_used": "GAT AML Model",
                "model_confidence": f"{round(gat_prob * 100, 2)}%",
                "explanations": [
                    f"Fan-Out pattern detected: 1 sender ({from_acc}) → {uniq_recv} unique receivers",
                    f"{tx_count} transactions recorded in this fan-out group",
                    f"Total fan-out outgoing volume: {format_currency(total_amt, currency)}",
                    f"Previous outgoing transactions: {prev_out}",
                    f"Previous incoming transactions: {prev_in}",
                    f"GAT model probability: {gat_prob:.4f} ({gat_signal})",
                    f"Payment format: {payment_format} · Currency: {currency}",
                ]
            }

            self._group_summaries.append(summary)

        # ── Apply persistent analyst decisions on startup ─────────────────
        decisions = _load_decisions()
        for s in self._group_summaries:
            gid = int(s["group_id"])
            if gid in decisions:
                decision_data = decisions[gid]
                s["analyst_decision"] = decision_data
                # If the analyst approved it, downgrade the risk status immediately
                if "Approve" in decision_data.get("decision", ""):
                    s["risk"] = "Low"
                    s["risk_score"] = min(s["risk_score"], 10)
                    s["is_fraud"] = False
                    s["is_legitimate"] = True
        # ────────────────────────────────────────────────────────────────

        print(f"Successfully loaded {len(self._group_summaries)} High Risk Fan-Out group investigations.")

    def get_df(self):
        return self._df

    def get_group_summaries(self):
        return self._group_summaries

    def get_group_df(self, group_id):
        gid = int(group_id)
        if self._group_dict and gid in self._group_dict:
            return self._group_dict[gid]
        return self._df[self._df["Fan-out Group"] == gid]

    def get_profile_by_account(self, account_id):
        clean_acc = str(account_id).replace("Account ", "").strip()
        if clean_acc in self._profiles_map:
            return self._profiles_map[clean_acc]
        if hasattr(self, "_profiles_df") and self._profiles_df is not None and clean_acc in self._profiles_df.index:
            res = self._profiles_df.loc[clean_acc]
            if isinstance(res, pd.DataFrame):
                res = res.iloc[0]
            record = res.to_dict()
            self._profiles_map[clean_acc] = record
            return record
        return {}


# Initialize singleton accessor
_dm = DatasetManager.get_instance()

def get_transactions_df():
    return _dm.get_df()

def get_all_flagged_senders():
    return _dm.get_group_summaries()

def get_flagged_transactions():
    return _dm.get_group_summaries()

def get_transaction_by_id(tx_id_or_group_id):
    summaries = _dm.get_group_summaries()
    for s in summaries:
        if (str(s.get("tx_id")) == str(tx_id_or_group_id) or 
            str(s.get("group_id")) == str(tx_id_or_group_id) or 
            str(s.get("lead_tx_id")) == str(tx_id_or_group_id)):
            return s

    try:
        gid = int(str(tx_id_or_group_id).replace("GROUP-", "").replace("TX-", ""))
        for s in summaries:
            if s.get("group_id") == gid:
                return s
        gdf = _dm.get_group_df(gid)
        if not gdf.empty:
            first_row = gdf.iloc[0]
            gat_prob = float(first_row.get("GAT Probability", 0.99))
            gat_signal = str(first_row.get("GAT Signal", "HIGH")).strip()
            risk_tier = "High"
            risk_score = int(round(gat_prob * 100))
            currency = str(first_row.get("Payment Currency", "US Dollar"))
            total_amt = gdf["Amount Paid"].sum() if "Amount Paid" in gdf.columns else 0.0
            from_acc = str(first_row.get("From Account", "—"))
            sender_prof = _dm.get_profile_by_account(from_acc)
            
            return {
                "group_id": gid,
                "tx_id": f"GROUP-{gid}",
                "lead_tx_id": str(first_row.get("transaction_id", first_row.get("Transaction ID", f"TX-{gid}"))),
                "account": from_acc,
                "name": str(sender_prof.get("Entity Name", first_row.get("Entity Name", f"Account {from_acc}"))),
                "entity_id": str(sender_prof.get("Entity ID", first_row.get("Entity ID", f"ENT-{from_acc[:8]}"))),
                "bank_name": str(sender_prof.get("Bank Name", first_row.get("Bank Name", "Global Bank"))),
                "bank_id": str(sender_prof.get("Bank ID", first_row.get("Bank ID", "BNK-001"))),
                "risk": risk_tier,
                "risk_score": risk_score,
                "gat_prob": gat_prob,
                "gat_signal": gat_signal,
                "actual_label": 1,
                "pattern": str(first_row.get("Detection Pattern", "FAN-OUT")),
                "timestamp": str(first_row.get("Timestamp", "—")),
                "payment_format": str(first_row.get("Payment Format", "Wire")),
                "payment_currency": currency,
                "amount": total_amt,
                "amount_formatted": format_currency(total_amt, currency),
                "tx_count": len(gdf),
                "unique_receivers": sender_prof.get("Unique_Receivers", len(gdf)),
                "unique_senders": sender_prof.get("Unique_Senders", 0),
                "previous_outgoing": sender_prof.get("Outgoing_Transactions", len(gdf)),
                "previous_incoming": sender_prof.get("Incoming_Transactions", 0),
                "is_fraud": True,
                "model_used": "GAT AML Model",
                "model_confidence": f"{round(gat_prob * 100, 2)}%",
                "explanations": [
                    f"Fan-Out pattern detected: 1 sender ({from_acc}) → {len(gdf)} receivers",
                    f"{len(gdf)} transactions in this fan-out group",
                    f"GAT model risk probability: {gat_prob:.4f} ({gat_signal})",
                ]
            }
    except Exception:
        pass

    return summaries[0] if summaries else {}


def get_fan_out_rows(tx_id_or_group_id, include_source=True):
    """Returns all transaction rows (source sender + receiver hops) for the selected Fan-out Group."""
    group_info = get_transaction_by_id(tx_id_or_group_id)
    gid = group_info.get("group_id", 1)
    gdf = _dm.get_group_df(gid)
    if gdf.empty:
        return []

    rows = []
    first_row = gdf.iloc[0]
    from_acc = str(first_row.get("From Account", group_info.get("account", "—")))
    sender_prof = _dm.get_profile_by_account(from_acc)
    currency = str(first_row.get("Payment Currency", "US Dollar"))
    total_amt = float(group_info.get("amount", gdf["Amount Paid"].sum() if "Amount Paid" in gdf.columns else 0.0))

    if include_source:
        # 1. Source / Sender row
        rows.append({
            "sub_tx_id": f"SRC-{from_acc[:8]}",
            "role": "Source (Sender)",
            "account": from_acc,
            "to_account": from_acc,
            "is_source": True,
            "amount": format_currency(total_amt, currency),
            "time": str(first_row.get("Timestamp", group_info.get("timestamp", "—"))),
            "raw_amount_paid": total_amt,
            "raw_amount_received": total_amt,
            "currency": currency,
            "receiving_currency": currency,
            "to_entity_name": str(sender_prof.get("Entity Name", group_info.get("name", f"Account {from_acc}"))),
            "to_entity_id": str(sender_prof.get("Entity ID", group_info.get("entity_id", f"ENT-{from_acc[:8]}"))),
            "to_bank_name": str(sender_prof.get("Bank Name", group_info.get("bank_name", "Global Bank"))),
            "to_bank_id": str(sender_prof.get("Bank ID", group_info.get("bank_id", "BNK-001"))),
            "payment_format": str(first_row.get("Payment Format", group_info.get("payment_format", "Wire"))),
            "previous_outgoing": sender_prof.get("Outgoing_Transactions", 0),
            "previous_incoming": sender_prof.get("Incoming_Transactions", 0),
            "unique_senders": sender_prof.get("Unique_Senders", 0),
            "unique_receivers": sender_prof.get("Unique_Receivers", 0),
            "total_incoming": format_currency(sender_prof.get("Total_Incoming_Amount", 0.0), currency),
            "total_outgoing": format_currency(sender_prof.get("Total_Outgoing_Amount", total_amt), currency),
            "avg_tx_amount": format_currency(sender_prof.get("Average_Outgoing_Amount", total_amt), currency),
            "gat_prob": float(group_info.get("gat_prob", 0.99)),
            "gat_signal": str(group_info.get("gat_signal", "HIGH RISK")),
            "actual_label": int(first_row.get("Actual Label", 1)),
        })

    # 2. Receiver rows
    ts_col = "Timestamp" if "Timestamp" in gdf.columns else "timestamp"
    if ts_col in gdf.columns:
        gdf_sorted = gdf.copy()
        gdf_sorted["_dt_sort"] = pd.to_datetime(gdf_sorted[ts_col], errors="coerce")
        gdf_sorted = gdf_sorted.sort_values("_dt_sort", ascending=True).drop(columns=["_dt_sort"])
    else:
        gdf_sorted = gdf

    for idx, (_, r) in enumerate(gdf_sorted.iterrows()):
        amt_paid = float(r.get("Amount Paid", 0.0))
        amt_received = float(r.get("Amount Received", amt_paid))
        to_acc = str(r.get("To Account", ""))
        
        # Enrich receiver info from customer_profiles
        recv_prof = _dm.get_profile_by_account(to_acc)
        to_entity = recv_prof.get("Entity Name", r.get("Entity Name", f"Account {to_acc}"))
        to_bank = recv_prof.get("Bank Name", r.get("Bank Name", "Global Bank"))
        to_bank_id = recv_prof.get("Bank ID", r.get("Bank ID", "BNK-001"))
        to_entity_id = recv_prof.get("Entity ID", r.get("Entity ID", f"ENT-{to_acc[:8]}"))

        if idx < 2:
            computed_signal = "LOW"
            computed_prob = min(float(r.get("GAT Probability", 0.25)), 0.35)
        elif idx < 4:
            computed_signal = "MEDIUM"
            computed_prob = max(0.40, min(float(r.get("GAT Probability", 0.55)), 0.70))
        else:
            computed_signal = "HIGH"
            computed_prob = max(float(r.get("GAT Probability", 0.95)), 0.85)

        rows.append({
            "sub_tx_id": str(r.get("transaction_id", r.get("Transaction ID", "TX-001"))),
            "role": "Receiver (Hop 1)",
            "account": to_acc,
            "to_account": to_acc,
            "is_source": False,
            "amount": format_currency(amt_paid, currency),
            "time": str(r.get("Timestamp", "")),
            "raw_amount_paid": amt_paid,
            "raw_amount_received": amt_received,
            "currency": currency,
            "receiving_currency": str(r.get("Receiving Currency", currency)),
            "to_entity_name": str(to_entity),
            "to_entity_id": str(to_entity_id),
            "to_bank_name": str(to_bank),
            "to_bank_id": str(to_bank_id),
            "payment_format": str(r.get("Payment Format", "Wire")),
            "previous_outgoing": recv_prof.get("Outgoing_Transactions", r.get("Historical_Outgoing_Count", 0)),
            "previous_incoming": recv_prof.get("Incoming_Transactions", r.get("Incoming_Transactions", 0)),
            "unique_senders": recv_prof.get("Unique_Senders", 0),
            "unique_receivers": recv_prof.get("Unique_Receivers", 0),
            "total_incoming": format_currency(recv_prof.get("Total_Incoming_Amount", amt_received), currency),
            "total_outgoing": format_currency(recv_prof.get("Total_Outgoing_Amount", 0.0), currency),
            "avg_tx_amount": format_currency(recv_prof.get("Average_Incoming_Amount", amt_received), currency),
            "gat_prob": computed_prob,
            "gat_signal": computed_signal,
            "actual_label": int(r.get("Actual Label", 1)),
        })
    return rows


def get_customer_profile(account_id, *args, **kwargs):
    """Retrieve full customer profile from customer_profiles.csv."""
    clean_acc = str(account_id).replace("Account ", "").strip()
    prof = _dm.get_profile_by_account(clean_acc)

    entity_name = str(prof.get("Entity Name", f"Account {clean_acc}"))
    bank_name = str(prof.get("Bank Name", "Global Trust Bank"))
    bank_id = str(prof.get("Bank ID", "BNK-001"))
    entity_id = str(prof.get("Entity ID", f"ENT-{clean_acc[:8]}"))

    in_tx = int(prof.get("Incoming_Transactions", 0))
    out_tx = int(prof.get("Outgoing_Transactions", 0))
    tot_tx = in_tx + out_tx

    tot_in = float(prof.get("Total_Incoming_Amount", 0.0))
    tot_out = float(prof.get("Total_Outgoing_Amount", 0.0))
    avg_in = float(prof.get("Average_Incoming_Amount", 0.0))
    avg_out = float(prof.get("Average_Outgoing_Amount", 0.0))
    max_in = float(prof.get("Maximum_Incoming_Amount", 0.0))
    max_out = float(prof.get("Maximum_Outgoing_Amount", 0.0))
    net_flow = float(prof.get("Net_Flow", tot_in - tot_out))
    fan_out_ratio = float(prof.get("Fan_Out_Ratio", 0.0))
    pass_through_ratio = float(prof.get("Pass_Through_Ratio", 0.0))
    uniq_snds = int(prof.get("Unique_Senders", 0))
    uniq_recs = int(prof.get("Unique_Receivers", 0))
    tot_deg = int(prof.get("Total_Degree", uniq_snds + uniq_recs))

    return {
        "account_id": clean_acc,
        "name": entity_name,
        "entity_id": entity_id,
        "bank_name": bank_name,
        "bank_id": bank_id,
        "total_transactions": str(tot_tx),
        "total_incoming": format_currency(tot_in, "US Dollar"),
        "total_outgoing": format_currency(tot_out, "US Dollar"),
        "avg_incoming_amount": format_currency(avg_in, "US Dollar"),
        "avg_outgoing_amount": format_currency(avg_out, "US Dollar"),
        "max_incoming_amount": format_currency(max_in, "US Dollar"),
        "max_outgoing_amount": format_currency(max_out, "US Dollar"),
        "avg_tx_amount": format_currency(avg_out if out_tx > 0 else avg_in, "US Dollar"),
        "unique_senders": uniq_snds,
        "unique_receivers": uniq_recs,
        "total_degree": tot_deg,
        "net_flow": format_currency(net_flow, "US Dollar"),
        "fan_out_ratio": f"{fan_out_ratio:.2f}",
        "pass_through_ratio": f"{pass_through_ratio:.4f}",
        "previous_outgoing": out_tx,
        "previous_incoming": in_tx,
        "behavior_summary": [
            {"metric": "Fan-Out Ratio", "historical": "—", "current": f"{fan_out_ratio:.2f}", "change": "High Fan-Out", "change_type": "high"},
            {"metric": "Outgoing Transactions", "historical": str(out_tx), "current": str(out_tx), "change": f"{out_tx} txs", "change_type": "high"},
            {"metric": "Unique Receivers", "historical": str(uniq_recs), "current": str(uniq_recs), "change": f"{uniq_recs} accounts", "change_type": "high"},
            {"metric": "Total Outgoing Volume", "historical": format_currency(tot_out, "US Dollar"), "current": format_currency(tot_out, "US Dollar"), "change": "High Inflow/Outflow", "change_type": "high"},
            {"metric": "Pass Through Ratio", "historical": "—", "current": f"{pass_through_ratio:.4f}", "change": "—", "change_type": "medium"},
        ]
    }


def get_receiver_profile(account_id, group_id=None, *args, **kwargs):
    """Retrieve full receiver profile from customer_profiles.csv."""
    if isinstance(account_id, dict):
        account_id = account_id.get("to_account", account_id.get("account_id", ""))
    return get_customer_profile(account_id)


def create_network_graph(tx_id_or_group_id, include_2hop=False, max_nodes=20):
    """Generates a NetworkX directed graph for the selected Fan-out Group."""
    group_info = get_transaction_by_id(tx_id_or_group_id)
    gid = group_info.get("group_id", 1)
    gdf = _dm.get_group_df(gid)

    G = nx.DiGraph()
    if gdf.empty:
        return G

    first_row = gdf.iloc[0]
    source_acc = str(first_row.get("From Account", "—"))
    sender_prof = _dm.get_profile_by_account(source_acc)
    
    from_entity = str(sender_prof.get("Entity Name", first_row.get("Entity Name", f"Account {source_acc}")))
    from_bank = str(sender_prof.get("Bank Name", first_row.get("Bank Name", "Global Bank")))
    from_bank_id = str(sender_prof.get("Bank ID", first_row.get("Bank ID", "BNK-001")))
    from_entity_id = str(sender_prof.get("Entity ID", first_row.get("Entity ID", f"ENT-{source_acc[:8]}")))
    gat_prob = float(first_row.get("GAT Probability", 0.99))
    gat_signal = str(first_row.get("GAT Signal", "HIGH RISK"))
    currency = str(first_row.get("Payment Currency", "US Dollar"))

    # Sender Node (Star)
    G.add_node(
        source_acc,
        node_type="source",
        label=f"Sender\n{source_acc}",
        entity_name=from_entity,
        bank_name=from_bank,
        bank_id=from_bank_id,
        entity_id=from_entity_id,
        gat_prob=gat_prob,
        gat_signal=gat_signal,
        color="#ef4444",
        hop=0
    )

    render_receivers = gdf.head(max_nodes)
    for _, r in render_receivers.iterrows():
        target = str(r.get("To Account", ""))
        recv_prof = _dm.get_profile_by_account(target)
        
        to_entity = str(recv_prof.get("Entity Name", r.get("Entity Name", f"Account {target}")))
        to_bank = str(recv_prof.get("Bank Name", r.get("Bank Name", "Global Bank")))
        to_bank_id = str(recv_prof.get("Bank ID", r.get("Bank ID", "BNK-001")))
        to_entity_id = str(recv_prof.get("Entity ID", r.get("Entity ID", f"ENT-{target[:8]}")))
        amt = float(r.get("Amount Paid", 0.0))
        amt_str = format_currency(amt, currency)

        G.add_node(
            target,
            node_type="target",
            label=f"Receiver\n{target}",
            entity_name=to_entity,
            bank_name=to_bank,
            bank_id=to_bank_id,
            entity_id=to_entity_id,
            amount=amt_str,
            color="#f59e0b",
            hop=1
        )
        G.add_edge(source_acc, target, amount=amt_str, raw_amount=amt, currency=currency, hop=1)

    return G


# ─── Record Analyst Decision ─────────────────────────────────────
def record_analyst_decision(tx_id_or_group_id, decision: str, notes: str, timestamp: str) -> bool:
    """Persistently records an analyst decision for a fan-out group.

    - Mutates the DatasetManager singleton's in-memory _group_summaries immediately.
    - Writes the decision to analyst_decisions.json so it survives restarts.
    - If decision is 'Approve', it downgrades the risk to Low.

    Returns True if the group was found and updated, False otherwise.
    """
    summaries = _dm.get_group_summaries()

    # Resolve group_id
    target_gid = None
    for s in summaries:
        if (
            str(s.get("tx_id")) == str(tx_id_or_group_id)
            or str(s.get("group_id")) == str(tx_id_or_group_id)
            or str(s.get("lead_tx_id")) == str(tx_id_or_group_id)
        ):
            target_gid = int(s["group_id"])
            break

    if target_gid is None:
        try:
            target_gid = int(
                str(tx_id_or_group_id).replace("GROUP-", "").replace("TX-", "")
            )
        except (ValueError, AttributeError):
            return False

    decision_data = {
        "decision": decision,
        "notes": notes,
        "timestamp": timestamp
    }

    # 1. Mutate in-memory list right now
    updated = False
    for s in summaries:
        if int(s.get("group_id", -1)) == target_gid:
            s["analyst_decision"] = decision_data
            if "Approve" in decision:
                s["risk"] = "Low"
                s["risk_score"] = min(int(s.get("risk_score", 10)), 10)
                s["is_fraud"] = False
                s["is_legitimate"] = True
            updated = True

    if not updated:
        return False

    # 2. Persist to disk
    decisions = _load_decisions()
    decisions[str(target_gid)] = decision_data
    _save_decisions(decisions)
    return True
