"""
Compare monthly-spend forecasters with a rolling-origin backtest.

    python -m ml.forecasting.compare --data-dir data/raw                      # baselines + gradient boosting (+ any installed optional models)
    python -m ml.forecasting.compare --max-series 40 --origins 6              # cap slow per-series models (ARIMA, Prophet) to 40 customers
    python -m ml.forecasting.compare --no-optional                            # skip ARIMA / Prophet / LSTM

Optional models run only if their library is installed: pip install statsmodels prophet torch
Writes docs/forecast_results.md
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from .backtest import backtest, interval_calibration, paired_bootstrap, summarise
from .data import monthly_total_panel
from .models import default_models


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data/raw")
    ap.add_argument("--origins", type=int, default=12, help="number of rolling forecast origins (the last N months)")
    ap.add_argument("--max-series", type=int, default=40, help="cap for slow per-series models (arima, prophet)")
    ap.add_argument("--no-optional", action="store_true")
    ap.add_argument("--baseline", default="naive_last")
    a = ap.parse_args()

    d = Path(a.data_dir)
    t = pd.read_csv(d / "transactions.csv", usecols=["customer_id", "txn_ts", "amount", "type", "category"], parse_dates=["txn_ts"])
    panel = monthly_total_panel(t, pd.read_csv(d / "customers.csv"))
    print(f"panel: {panel.Y.shape[0]} customers x {panel.Y.shape[1]} months; {a.origins} rolling origins", flush=True)

    models = default_models(include_optional=not a.no_optional)
    print("models:", ", ".join(m.name for m in models), flush=True)
    caps = {"arima": a.max_series, "prophet": a.max_series}
    res = backtest(models, panel, a.origins, caps)
    table = summarise(res, a.baseline)
    pd.set_option("display.width", 200)
    print("\n" + table.round(3).to_string())

    names = [m for m in table.index if m != a.baseline]
    boots = pd.DataFrame([paired_bootstrap(res, m, a.baseline) for m in names]).round(4)
    print("\nWAPE difference vs", a.baseline, "(negative = better; 95% bootstrap CI over customers):")
    print(boots.to_string(index=False))
    simple = [m for m in ("naive_last", "mean3", "mean6", "median6", "seasonal_naive", "ses") if m in table.index]
    strongest = table.loc[simple, "WAPE"].idxmin()
    vs_strong = pd.DataFrame([paired_bootstrap(res, m, strongest) for m in table.index if m != strongest]).round(4)
    print(f"\nWAPE difference vs the STRONGEST simple baseline ({strongest}); negative = better:")
    print(vs_strong.to_string(index=False))
    best = table.index[0]
    cal = pd.DataFrame([interval_calibration(res, best, 0.8)]).round(3)
    print("\nPrediction-interval calibration for the best model (80% target):\n" + cal.to_string(index=False))

    Path("docs").mkdir(exist_ok=True)
    md = ["# Forecast backtest results", "",
          f"{panel.Y.shape[0]} simulated customers, {panel.Y.shape[1]} months, rolling-origin backtest over the last {a.origins} months. "
          "Slow per-series models (ARIMA, Prophet) are run on a subset; every comparison uses only the customers all models share.", "",
          table.round(3).reset_index().rename(columns={"index": "model"}).to_markdown(index=False) if _has_tabulate()
          else "```\n" + table.round(3).to_string() + "\n```", "",
          "WAPE difference vs " + a.baseline + " (negative = better):", "", "```\n" + boots.to_string(index=False) + "\n```", "",
          f"WAPE difference vs the strongest simple baseline ({strongest}):", "", "```\n" + vs_strong.to_string(index=False) + "\n```", "",
          "Interval calibration:", "", "```\n" + cal.to_string(index=False) + "\n```"]
    Path("docs/forecast_results.md").write_text("\n".join(md) + "\n", encoding="utf-8")


def _has_tabulate() -> bool:
    try:
        import tabulate  # noqa: F401
        return True
    except Exception:
        return False


if __name__ == "__main__":
    main()
