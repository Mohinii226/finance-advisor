"""
Produce next-month spending forecasts (with calibrated intervals) for every customer.

    python -m ml.forecasting.run_forecast --source csv --data-dir data/raw      # writes data/raw/forecasts.csv (no database needed)
    python -m ml.forecasting.run_forecast --source db                           # reads transactions, writes the `forecasts` table

Model: global gradient boosting (L1). Interval: relative-error quantiles (10th / 90th percentile) from a rolling-origin
backtest of the same model on the same history, so the stated 80% range is empirical, not assumed.
The median-of-6-months baseline is stored alongside for transparency.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd

from .backtest import backtest
from .data import Panel, monthly_total_panel
from .models import EPS, GlobalGBM, MedianK

TABLE_COLUMNS = ["customer_id", "category", "target_month", "model_name", "predicted_amount", "lower_bound", "upper_bound"]


def calibrate_interval(panel: Panel, model, n_origins: int = 12, level: float = 0.8) -> tuple[float, float]:
    """Pooled relative errors (actual - pred) / pred over the last `n_origins` rolling origins."""
    res = backtest([model], panel, n_origins, verbose=False)
    rel = (res["actual"] - res["pred"]) / np.maximum(res["pred"], EPS)
    lo, hi = np.quantile(rel, [(1 - level) / 2, 1 - (1 - level) / 2])
    return float(lo), float(hi)


def forecast_next_month(panel: Panel, n_origins: int = 12, level: float = 0.8) -> pd.DataFrame:
    """Rows for the `forecasts` table: the GBM forecast with an empirical interval, plus the median-6 baseline."""
    target = panel.months[-1] + pd.DateOffset(months=1)
    gbm = GlobalGBM("absolute_error")
    lo, hi = calibrate_interval(panel, GlobalGBM("absolute_error"), n_origins, level)
    pred = gbm.forecast(panel.Y, panel.months, panel.meta)
    base = MedianK(6).forecast(panel.Y, panel.months, panel.meta)
    rows = []
    for name, p, l, h in (("gbm_l1", pred, lo, hi), ("median6", base, None, None)):
        rows.append(pd.DataFrame({
            "customer_id": panel.meta["customer_id"], "category": None, "target_month": target.date(), "model_name": name,
            "predicted_amount": np.round(p, 2),
            "lower_bound": np.round(np.maximum(p * (1 + l), 0), 2) if l is not None else None,
            "upper_bound": np.round(p * (1 + h), 2) if h is not None else None}))
    return pd.concat(rows, ignore_index=True)[TABLE_COLUMNS]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["csv", "db"], default="csv")
    ap.add_argument("--data-dir", default="data/raw")
    a = ap.parse_args()
    if a.source == "csv":
        d = Path(a.data_dir)
        t = pd.read_csv(d / "transactions.csv", usecols=["customer_id", "txn_ts", "amount", "type", "category"], parse_dates=["txn_ts"])
        panel = monthly_total_panel(t, pd.read_csv(d / "customers.csv"))
        out = forecast_next_month(panel)
        out.to_csv(d / "forecasts.csv", index=False)
        print(f"wrote {len(out):,} forecasts to {d / 'forecasts.csv'}")
        return
    from sqlalchemy import create_engine, text
    eng = create_engine(os.getenv("DATABASE_URL", "postgresql+psycopg2://finance:finance@localhost:5432/finance_advisor"))
    t = pd.read_sql("SELECT customer_id, txn_ts, amount::float AS amount, type, category FROM transactions WHERE type = 'debit'",
                    eng, parse_dates=["txn_ts"])
    cust = pd.read_sql("SELECT customer_id, monthly_income::float AS monthly_income, income_type FROM customers", eng)
    out = forecast_next_month(monthly_total_panel(t, cust))
    with eng.begin() as con:
        con.execute(text("DELETE FROM forecasts WHERE category IS NULL AND target_month = :m"), {"m": out["target_month"].iloc[0]})
        out.rename(columns={"lower_bound": "lower_bound", "upper_bound": "upper_bound"}).to_sql(
            "forecasts", con, if_exists="append", index=False, method="multi", chunksize=5000)
    print(f"wrote {len(out):,} forecasts for {out['target_month'].iloc[0]}")


if __name__ == "__main__":
    main()
