"""
Load the generated CSVs into PostgreSQL using COPY (fast: ~1M rows in seconds).

Usage:
    docker compose up -d db
    python -m data_gen.load_to_postgres --reset
Env:
    DATABASE_URL (default postgresql+psycopg2://finance:finance@localhost:5432/finance_advisor)
"""
from __future__ import annotations

import argparse
import io
import os
from pathlib import Path

import pandas as pd

from . import config as C

DEFAULT_URL = "postgresql+psycopg2://finance:finance@localhost:5432/finance_advisor"

# load order respects foreign keys
TABLES = {
    "customers": ["customer_id", "name", "age", "city", "city_tier", "occupation", "income_type", "monthly_income",
                  "dependents", "has_loan", "owns_home", "has_vehicle", "persona"],
    "accounts": ["account_id", "customer_id", "bank", "account_last4", "opening_balance", "opening_reserve",
                 "emi_amount", "sip_amount"],
    "injected_events": ["event_id", "customer_id", "event_type", "category", "start_month", "end_month",
                        "magnitude", "description"],
    "transactions": ["txn_id", "customer_id", "txn_date", "txn_ts", "amount", "type", "raw_description", "merchant",
                     "category", "true_category", "predicted_category", "confidence", "channel", "balance_after"],
    "sms_raw": ["sms_id", "customer_id", "sender", "body", "received_at", "is_transaction", "true_txn_id",
                "parsed_txn_id"],
}


def dim_category_frame() -> pd.DataFrame:
    rows = []   # (category, kind, is_essential, is_expense, is_primary_income)
    for cat, spec in C.CATEGORIES.items():
        rows.append((cat, "expense", spec["essential"], True, False))
    rows += [("EMI", "debt", True, True, False), ("Investments", "investment", False, False, False),
             ("Salary", "income", False, False, True), ("Freelance Income", "income", False, False, True),
             ("Pension", "income", False, False, True), ("Allowance", "income", False, False, True),
             ("Other Income", "income", False, False, False), ("Self Transfer", "transfer", False, False, False)]
    return pd.DataFrame(rows, columns=["category", "kind", "is_essential", "is_expense", "is_primary_income"])


def copy_df(raw_conn, df: pd.DataFrame, table: str, cols: list[str]) -> None:
    buf = io.StringIO()
    df[cols].to_csv(buf, index=False, header=False)     # NaN/None -> empty field -> NULL
    buf.seek(0)
    with raw_conn.cursor() as cur:
        cur.copy_expert(f"COPY {table} ({', '.join(cols)}) FROM STDIN WITH (FORMAT csv)", buf)


def main() -> None:
    from sqlalchemy import create_engine   # lazy: only needed when actually loading

    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data/raw")
    ap.add_argument("--reset", action="store_true", help="truncate tables before loading")
    a = ap.parse_args()

    engine = create_engine(os.getenv("DATABASE_URL", DEFAULT_URL))
    raw = engine.raw_connection()
    try:
        with raw.cursor() as cur:
            cur.execute(Path("sql/schema.sql").read_text(encoding="utf-8"))     # idempotent (IF NOT EXISTS)
            if a.reset:
                cur.execute("TRUNCATE sms_raw, transactions, injected_events, accounts, customers, "
                            "alerts, forecasts, health_scores, dim_category RESTART IDENTITY CASCADE")
        raw.commit()

        copy_df(raw, dim_category_frame(), "dim_category", ["category", "kind", "is_essential", "is_expense", "is_primary_income"])
        for table, cols in TABLES.items():
            df = pd.read_csv(Path(a.data_dir) / f"{table}.csv")
            for c in ("true_txn_id", "parsed_txn_id", "confidence"):
                if c in df.columns and df[c].dtype == "float64" and c != "confidence":
                    df[c] = df[c].astype("Int64")
            copy_df(raw, df, table, cols)
            print(f"loaded {table:16s} {len(df):>9,d} rows")
        raw.commit()

        with raw.cursor() as cur:                       # analytics views depend on the loaded tables
            cur.execute(Path("sql/views.sql").read_text(encoding="utf-8"))
        raw.commit()
        print("created analytics views (sql/views.sql)")
    finally:
        raw.close()


if __name__ == "__main__":
    main()
