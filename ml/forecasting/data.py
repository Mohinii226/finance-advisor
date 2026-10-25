"""Monthly spending panels for forecasting."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

NON_SPEND = ("Investments", "Self Transfer")
INCOME_TYPES = ["salaried", "govt", "freelance", "student", "pension"]


@dataclass
class Panel:
    Y: np.ndarray                 # [n_series, T] monthly spend in rupees
    months: pd.DatetimeIndex      # length T (month starts)
    meta: pd.DataFrame            # one row per series: customer_id, category, monthly_income, income_type_code


def monthly_total_panel(txns: pd.DataFrame, customers: pd.DataFrame, category: str | None = None) -> Panel:
    """Total monthly spending per customer (debits excluding investments and self-transfers), or one category.

    Uses the bank/user `category` label (blank -> 'Uncategorized') exactly as the SQL views do, so totals do not
    depend on the categoriser; category-level panels do.
    """
    t = txns[txns["type"] == "debit"].copy()
    t["cat"] = t["category"].fillna("Uncategorized")
    t = t[~t["cat"].isin(NON_SPEND)]
    if category is not None:
        t = t[t["cat"] == category]
    t["month"] = pd.to_datetime(t["txn_ts"]).dt.to_period("M").dt.to_timestamp()
    wide = t.pivot_table(index="customer_id", columns="month", values="amount", aggfunc="sum", fill_value=0.0)
    months = pd.date_range(wide.columns.min(), wide.columns.max(), freq="MS")
    wide = wide.reindex(columns=months, fill_value=0.0)
    cust = customers.set_index("customer_id").reindex(wide.index)
    meta = pd.DataFrame({"customer_id": wide.index, "category": category or "TOTAL",
                         "monthly_income": cust["monthly_income"].to_numpy(float),
                         "income_type_code": cust["income_type"].map({k: i for i, k in enumerate(INCOME_TYPES)}).fillna(-1).to_numpy(int)})
    return Panel(wide.to_numpy(float), months, meta.reset_index(drop=True))
