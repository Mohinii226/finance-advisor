"""
Framework-free data access for the Streamlit app. Every function takes `run(query_name, **params) -> DataFrame`
and returns plain pandas objects with clean numeric dtypes (PostgreSQL returns Decimal / date objects), so it can be
tested without Streamlit or PostgreSQL.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

import pandas as pd

from core.service import AdvisorService, ToolError

DAY_NAMES = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
Run = Callable[..., pd.DataFrame]


def _num(df: pd.DataFrame, cols) -> pd.DataFrame:
    df = df.copy()
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def month_str(m) -> str:
    return pd.Timestamp(m).strftime("%Y-%m-%d")


def safe(fn, *args, **kwargs) -> tuple[bool, object]:
    """Run a service call; return (True, value) or (False, user-facing message). The UI never crashes on a missing table."""
    try:
        return True, fn(*args, **kwargs)
    except ToolError as e:
        return False, str(e)
    except Exception as e:                      # database hiccup, missing table, etc.
        return False, f"Unavailable right now ({type(e).__name__})."


def available_months(svc: AdvisorService, min_history: int = 6) -> list[pd.Timestamp]:
    """Months a health score can be computed for (needs `min_history` earlier months), newest first."""
    if pd.isna(svc.first_month) or pd.isna(svc.last_month):      # a customer with no transactions (row-level security shows only their own months)
        return []
    first = svc.first_month + pd.DateOffset(months=min_history - 1)
    return list(pd.date_range(first, svc.last_month, freq="MS"))[::-1]


def spend_history(run: Run, cid: str, month, limit: int = 12) -> pd.DataFrame:
    df = run("monthly_summary_history", cid=cid, month=month_str(month), limit=limit)
    df = _num(df, ["income", "expenses", "emi", "discretionary", "investments", "savings_rate"])
    df["month"] = pd.to_datetime(df["month"])
    return df.sort_values("month").reset_index(drop=True)


def category_history(run: Run, cid: str, month, months: int = 12, top_n: int = 7) -> pd.DataFrame:
    """Long frame (month, category, spend) for the last `months`; categories beyond the top_n are grouped as 'Other'."""
    m = pd.Timestamp(month)
    df = run("category_history", cid=cid, month=month_str(m), start=month_str(m - pd.DateOffset(months=months - 1)))
    df = _num(df, ["spend"])
    df["month"] = pd.to_datetime(df["month"])
    top = df.groupby("category")["spend"].sum().sort_values(ascending=False).head(top_n).index
    df["category"] = df["category"].where(df["category"].isin(top), "Other")
    return df.groupby(["month", "category"], as_index=False)["spend"].sum()


def weekday_pattern(run: Run, cid: str, month, months: int = 6) -> pd.DataFrame:
    m = pd.Timestamp(month)
    df = run("weekday_pattern", cid=cid, start=month_str(m - pd.DateOffset(months=months - 1)), end=month_str(m + pd.DateOffset(months=1)))
    df = _num(df, ["isodow", "spend", "txns", "share_pct", "avg_ticket"])
    if df.empty:
        return df.assign(day=[])
    df["day"] = df["isodow"].astype(int).map(lambda i: DAY_NAMES[i - 1])
    return df.sort_values("isodow").reset_index(drop=True)


def seasonal_index(run: Run, cid: str) -> pd.DataFrame:
    df = _num(run("seasonal_index", cid=cid), ["moy", "avg_expenses", "avg_discretionary", "expense_index", "discretionary_index"])
    df["moy"] = df["moy"].astype(int)
    df["month_name"] = df["moy"].map(lambda i: pd.Timestamp(2000, i, 1).strftime("%b"))
    return df.sort_values("moy").reset_index(drop=True)


def top_merchants(run: Run, cid: str, month, months: int = 6, limit: int = 8) -> pd.DataFrame:
    m = pd.Timestamp(month)
    df = run("top_unnecessary_spend", cid=cid, start=month_str(m - pd.DateOffset(months=months - 1)),
             end=month_str(m + pd.DateOffset(months=1)), limit=limit)
    return _num(df, ["spend", "txns"])


def largest_transactions(run: Run, cid: str, month, limit: int = 10) -> pd.DataFrame:
    df = _num(run("recent_transactions", cid=cid, month=month_str(month), limit=limit), ["amount"])
    if not df.empty:
        df["txn_date"] = pd.to_datetime(df["txn_date"]).dt.strftime("%d %b")
    return df


def slider_bounds(income: float, step: int = 500) -> tuple[int, int, int]:
    """(min, max, default) for the savings-target slider. Streamlit raises if the default lies outside [min, max],
    so this is always valid, even for very low incomes."""
    lo = step
    hi = max(int(max(income * 0.5, 2 * step) // step * step), lo)
    default = int(round(min(10000, income * 0.2) / step) * step)
    return lo, hi, min(max(default, lo), hi)


def money(x) -> str:
    return "n/a" if x is None or (isinstance(x, float) and pd.isna(x)) else f"Rs {float(x):,.0f}"


def load_model_lab(root: Path) -> dict:
    """Reads the evaluation artefacts written by the training / backtest scripts (missing ones are None)."""
    def read(p):
        return p.read_text(encoding="utf-8") if p.exists() else None
    docs = root / "docs"
    cat = docs / "categorizer"
    metrics = None
    if (cat / "metrics.json").exists():
        metrics = json.loads((cat / "metrics.json").read_text(encoding="utf-8"))
    return dict(overspend=read(docs / "overspend_results.md"), forecast=read(docs / "forecast_results.md"),
                core=read(docs / "core_results.md"), categorizer_metrics=metrics,
                categorizer_per_class=pd.read_csv(cat / "per_class.csv", index_col=0) if (cat / "per_class.csv").exists() else None,
                confusion_png=str(cat / "confusion_matrix.png") if (cat / "confusion_matrix.png").exists() else None)
