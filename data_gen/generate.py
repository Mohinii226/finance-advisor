"""
Generate the full synthetic dataset and write CSVs to data/raw/.

Usage:
    python -m data_gen.generate --customers 300 --months 36 --seed 42
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

from . import config as C
from .profiles import build_customer, build_events, customers_to_frames, events_to_frame
from .ambiguity import make_ambiguous
from .sms_generator import generate_sms
from .transactions import apply_ledger, apply_label_noise, generate_customer_transactions


def build_dataset(n_customers: int = 300, n_months: int = 36, seed: int = 42, sms_frac: float = 0.25,
                  ambiguity: bool = True) -> dict[str, pd.DataFrame]:
    rng = np.random.default_rng(seed)

    customers = [build_customer(i + 1, rng) for i in range(n_customers)]
    cmap = {c.customer_id: c for c in customers}
    all_events, rows = [], []
    for c in customers:
        ev = build_events(c, n_months, rng)
        all_events.extend(ev)
        rows.extend(generate_customer_transactions(c, ev, n_months, rng))

    raw = pd.DataFrame(rows, columns=["customer_id", "txn_ts", "amount", "type", "raw_description",
                                      "merchant", "true_category", "channel"])
    txns = apply_ledger(raw, cmap, rng)
    txns = apply_label_noise(txns, rng)
    txns = txns.sort_values(["txn_ts", "customer_id"]).reset_index(drop=True)
    txns.insert(0, "txn_id", np.arange(1, len(txns) + 1))
    txns["txn_date"] = txns["txn_ts"].dt.date
    txns["predicted_category"] = None
    txns["confidence"] = np.nan
    txns = txns[["txn_id", "customer_id", "txn_date", "txn_ts", "amount", "type", "raw_description", "merchant",
                 "category", "true_category", "predicted_category", "confidence", "channel", "balance_after"]]

    cust_df, acc_df = customers_to_frames(customers)
    if ambiguity:       # realistic, partly ambiguous descriptions (own RNG; amounts/categories untouched)
        txns = make_ambiguous(txns, cust_df, seed=seed)
    events_df = events_to_frame(all_events)
    sms_df = generate_sms(txns, cust_df, acc_df, customer_frac=sms_frac, seed=seed)
    return dict(customers=cust_df, accounts=acc_df, transactions=txns, injected_events=events_df, sms_raw=sms_df)


def summarize(d: dict[str, pd.DataFrame]) -> None:
    t = d["transactions"]
    print("\n=== Dataset summary ===")
    for k, v in d.items():
        print(f"{k:16s} {len(v):>9,d} rows")
    deb = t[(t.type == "debit") & (~t.true_category.isin(["Self Transfer"]))]
    print(f"\nDate range        : {t.txn_date.min()} -> {t.txn_date.max()}")
    print(f"Unlabelled debits : {t[(t.type=='debit') & t.category.isna()].shape[0] / (t.type=='debit').sum():.1%}")
    print("\nPersona mix:\n", d["customers"].persona.value_counts().to_string())
    print("\nInjected events:\n", d["injected_events"].event_type.value_counts().to_string())
    print("\nSpend share by category (debits, excl. self-transfers):")
    print((deb.groupby("true_category").amount.sum() / deb.amount.sum()).sort_values(ascending=False).map("{:.1%}".format).to_string())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--customers", type=int, default=300)
    ap.add_argument("--months", type=int, default=36)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--sms-frac", type=float, default=0.25)
    ap.add_argument("--no-ambiguity", action="store_true", help="easy mode: every description names its merchant")
    ap.add_argument("--out", type=str, default="data/raw")
    a = ap.parse_args()

    t0 = time.time()
    data = build_dataset(a.customers, a.months, a.seed, a.sms_frac, ambiguity=not a.no_ambiguity)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for name, df in data.items():
        df.to_csv(out / f"{name}.csv", index=False)
    summarize(data)
    print(f"\nWrote CSVs to {out.resolve()} in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
