"""
Run the overspend detector on the database and write results to the `alerts` table.

    python -m ml.anomaly.run_detector --preset balanced            # all months, replaces previous alerts
    python -m ml.anomaly.run_detector --preset low_noise --latest   # only the most recent month (production-style)
"""
from __future__ import annotations

import argparse
import os

import pandas as pd

from ml.anomaly.overspend import PRESETS, detect, score_cells

DEFAULT_URL = "postgresql+psycopg2://finance:finance@localhost:5432/finance_advisor"
ALERT_COLUMNS = ["customer_id", "alert_month", "category", "alert_type", "current_spend",
                 "baseline_mean", "pct_change", "z_score", "message"]


def alerts_to_frame(alerts: pd.DataFrame, cfg=None) -> pd.DataFrame:
    """Map detector output onto the `alerts` table columns."""
    if alerts.empty:
        return pd.DataFrame(columns=ALERT_COLUMNS)
    z = alerts["z"].where(alerts["severity"] == "spike", alerts["z_shift"].fillna(alerts["z"]))
    return pd.DataFrame({
        "customer_id": alerts["customer_id"], "alert_month": alerts["month"].dt.date, "category": alerts["category"],
        "alert_type": alerts["severity"], "current_spend": alerts["spend"].round(2),
        "baseline_mean": alerts["avg_ref"].round(2),
        "pct_change": ((alerts["spend"] / alerts["avg_ref"] - 1) * 100).round(2),
        "z_score": z.round(2), "message": alerts["message"],
    })[ALERT_COLUMNS]


def main() -> None:
    from sqlalchemy import create_engine, text   # lazy import

    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", choices=list(PRESETS), default="balanced")
    ap.add_argument("--latest", action="store_true", help="only alerts for the latest month in the data")
    a = ap.parse_args()
    cfg = PRESETS[a.preset]

    eng = create_engine(os.getenv("DATABASE_URL", DEFAULT_URL))
    grid = pd.read_sql("SELECT customer_id, category, month, spend, txn_count FROM v_category_month_grid", eng,
                       parse_dates=["month"])
    grid[["spend", "txn_count"]] = grid[["spend", "txn_count"]].astype(float)
    alerts = detect(grid, cfg, cells=score_cells(grid, cfg))
    if a.latest:
        alerts = alerts[alerts["month"] == grid["month"].max()]
    out = alerts_to_frame(alerts, cfg)
    with eng.begin() as con:
        con.execute(text("DELETE FROM alerts"))
        out.to_sql("alerts", con, if_exists="append", index=False, method="multi", chunksize=5000)
    print(f"wrote {len(out):,} alerts ({a.preset}); sample:")
    print(out["message"].head(5).to_string(index=False))


if __name__ == "__main__":
    main()
