"""
Makes the SQLite test runner return what psycopg2 / PostgreSQL would return, so the service layer can be tested
against Postgres-style types without a Postgres server:
  * NUMERIC / SUM / AVG / ROUND / EXTRACT  -> decimal.Decimal   (SQLite gives float / int)
  * DATE columns                           -> datetime.date     (SQLite gives 'YYYY-MM-DD' strings)
  * NULL                                   -> None              (not NaN)
COUNT() and plain integer columns stay int, as in Postgres.
"""
import datetime as dt
from decimal import Decimal

import pandas as pd

DATE_COLUMNS = {"month", "txn_date", "first_month", "last_month", "alert_month", "start_month", "end_month", "target_month"}
DECIMAL_INT_COLUMNS = {"isodow", "moy"}                     # EXTRACT(...) returns numeric in PostgreSQL 14+
INT_COLUMNS = {"txns", "txn_count", "n6", "n_hist", "alert_count", "age", "dependents", "city_tier", "n_alerts"}


def _dec(v):
    return None if pd.isna(v) else Decimal(repr(float(v)))


def pg_like(run):
    def wrapped(name, **params):
        df = run(name, **params)
        out = df.astype(object).copy()
        for c in out.columns:
            if c in DATE_COLUMNS:
                out[c] = [None if pd.isna(v) else pd.Timestamp(v).date() for v in df[c]]
            elif c in DECIMAL_INT_COLUMNS:                           # EXTRACT(...) -> Decimal even though the value is whole
                out[c] = [_dec(v) for v in df[c]]
            elif c in INT_COLUMNS or df[c].dtype.kind in "iub":
                out[c] = [None if pd.isna(v) else int(v) for v in df[c]]
            elif df[c].dtype.kind == "f":
                out[c] = [_dec(v) for v in df[c]]
            else:
                out[c] = [None if (not isinstance(v, str) and pd.isna(v)) else v for v in df[c]]
        return out
    return wrapped
