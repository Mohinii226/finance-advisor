"""
Compare overspend detectors on DEV (tuning world) and TEST (unseen world) datasets.

    python -m ml.anomaly.compare --customers 150        # ~2 minutes
Writes docs/overspend_results.md
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from data_gen.generate import build_dataset
from ml.anomaly.evaluate import average_precision, evaluate, f1, grid_from_transactions
from ml.anomaly.overspend import PRESETS, detect, score_cells

COLS = ["world", "detector", "alerts_per_cust_month", "precision", "event_recall", "recall_creep", "recall_spike",
        "recall_medical", "ttd_creep", "f1", "avg_precision", "base_rate"]


def run_world(seed: int, n_customers: int, worlds: dict[str, str]) -> list[dict]:
    d = build_dataset(n_customers, 36, seed=seed, sms_frac=0.0)       # generated once, scored under each labelling
    rows = []
    for world_name, label in worlds.items():
        grid = grid_from_transactions(d["transactions"], label=label)
        months = pd.date_range(grid["month"].min(), grid["month"].max(), freq="MS")
        for name, cfg in PRESETS.items():
            cells = score_cells(grid, cfg)
            m = evaluate(detect(grid, cfg, cells=cells), d["injected_events"], months, n_customers)
            ap, base = average_precision(cells, d["injected_events"], months, "z_shift" if cfg.shift_len > 1 else "z")
            rows.append(dict(world=world_name, detector=name, **m, f1=f1(m), avg_precision=ap, base_rate=base))
    return rows


def to_markdown(df: pd.DataFrame) -> str:
    lines = ["| " + " | ".join(df.columns) + " |", "|" + "---|" * len(df.columns)]
    lines += ["| " + " | ".join(str(v) for v in r) + " |" for r in df.itertuples(index=False)]
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--customers", type=int, default=150)
    ap.add_argument("--dev-seed", type=int, default=11)
    ap.add_argument("--test-seed", type=int, default=22)
    a = ap.parse_args()

    rows = run_world(a.dev_seed, a.customers, {"DEV (tuned on)": "category"}) \
        + run_world(a.test_seed, a.customers, {"TEST (unseen)": "category", "TEST, perfect categoriser": "true_category"})
    show = pd.DataFrame(rows)[COLS].round(3)
    pd.set_option("display.width", 250, "display.max_columns", 30)
    print(show.to_string(index=False))
    Path("docs").mkdir(exist_ok=True)
    Path("docs/overspend_results.md").write_text(to_markdown(show) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
