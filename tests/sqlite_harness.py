"""
Dev-time harness: run the Postgres analytics SQL under SQLite so the logic can be
verified without a database server.  It is a SMOKE TEST, not a substitute for
running against PostgreSQL (the loader does that in real use).

Translated constructs: ::casts, CREATE OR REPLACE VIEW, EXTRACT(x FROM y), date_trunc.
"""
from __future__ import annotations

import math
import re
import sqlite3
from datetime import date
from pathlib import Path

import pandas as pd

from core.queries import load_queries
from data_gen.load_to_postgres import dim_category_frame

ROOT = Path(__file__).resolve().parents[1]


def translate(sql: str) -> str:
    sql = re.sub(r"::(date|numeric|int|bigint)\b", "", sql)
    sql = sql.replace("CREATE OR REPLACE VIEW", "CREATE VIEW")
    sql = re.sub(r"EXTRACT\((\w+)\s+FROM\s+([^)]+)\)", r"extract_part('\1', \2)", sql, flags=re.I)
    return sql


def _date_trunc(unit, x):
    d = date.fromisoformat(str(x)[:10])
    return d.replace(day=1).isoformat() if unit == "month" else d.isoformat()


def _extract_part(part, x):
    d = date.fromisoformat(str(x)[:10])
    return {"isodow": d.isoweekday(), "month": d.month, "year": d.year}[part]


def _strip_comments(sql: str) -> str:
    return "\n".join(l for l in sql.splitlines() if not l.strip().startswith("--"))


def connect(data: dict[str, pd.DataFrame]) -> sqlite3.Connection:
    con = sqlite3.connect(":memory:")
    con.create_function("date_trunc", 2, _date_trunc)
    con.create_function("extract_part", 2, _extract_part)
    con.create_function("sqrt", 1, lambda v: None if v is None else math.sqrt(v))
    for name, df in data.items():
        df = df.copy()
        for c in df.columns:
            if "date" in c or c.endswith("_ts") or c in ("start_month", "end_month", "received_at"):
                df[c] = df[c].astype(str)
        df.to_sql(name, con, index=False)
    dim_category_frame().to_sql("dim_category", con, index=False)
    script = translate(_strip_comments((ROOT / "sql" / "views.sql").read_text(encoding="utf-8")))
    for stmt in filter(str.strip, script.split(";")):
        con.execute(stmt)
    return con


def run(con: sqlite3.Connection, name: str, **params) -> pd.DataFrame:
    sql = translate(load_queries()[name])
    return pd.read_sql_query(sql, con, params=params)
