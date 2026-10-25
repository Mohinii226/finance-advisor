"""
Score overspend detectors against injected ground-truth events.

Fairness rules
  * Only months with index >= ELIGIBLE_FROM are scored, and only events starting then count, so every
    config (with a different history requirement) is judged on the same cells and the same events.
  * Alert precision = share of alerts that fall inside a real event window (same customer + category).
    "False" alerts may be genuine but unlabeled spikes (e.g. a big one-off trip), so treat precision as a lower bound.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

ELIGIBLE_FROM = 8                     # month index (0-based): window 6 + gap 2
EVENT_TYPES = ("category_creep", "category_spike", "medical_emergency")
NON_SPEND = ("Investments", "Self Transfer")


def grid_from_transactions(txns: pd.DataFrame, label: str = "category") -> pd.DataFrame:
    """Pandas twin of v_category_month_grid. label='true_category' gives the perfect-categoriser ceiling."""
    t = txns[txns["type"] == "debit"].copy()
    if label == "category":
        t["cat"] = t["category"].fillna(t.get("predicted_category")).fillna("Uncategorized")
    else:
        t["cat"] = t[label].fillna("Uncategorized")
    t = t[~t["cat"].isin(NON_SPEND)]
    t["month"] = pd.to_datetime(t["txn_ts"]).dt.to_period("M").dt.to_timestamp()
    g = t.groupby(["customer_id", "cat", "month"], as_index=False).agg(spend=("amount", "sum"), txn_count=("amount", "size"))
    return g.rename(columns={"cat": "category"})


def _events(events: pd.DataFrame, months: pd.DatetimeIndex) -> pd.DataFrame:
    e = events[events["event_type"].isin(EVENT_TYPES) & events["category"].notna()].copy()
    e["start_month"], e["end_month"] = pd.to_datetime(e["start_month"]), pd.to_datetime(e["end_month"])
    idx = {m: i for i, m in enumerate(months)}
    e["start_idx"] = e["start_month"].map(idx)
    return e[e["start_idx"] >= ELIGIBLE_FROM]


def evaluate(alerts: pd.DataFrame, events: pd.DataFrame, months: pd.DatetimeIndex, n_customers: int) -> dict:
    ev = _events(events, months)
    midx = {m: i for i, m in enumerate(months)}
    a = alerts[alerts["month"].map(midx) >= ELIGIBLE_FROM].copy()
    n_cm = n_customers * (len(months) - ELIGIBLE_FROM)

    if a.empty:
        return dict(alerts=0, alerts_per_cust_month=0.0, precision=np.nan, event_recall=0.0, recall_creep=0.0,
                    recall_spike=0.0, recall_medical=0.0, ttd_creep=np.nan, n_events=len(ev))
    j = a.reset_index(drop=True).reset_index().merge(
        ev[["event_id", "customer_id", "category", "start_month", "end_month", "event_type", "start_idx"]],
        on=["customer_id", "category"], how="left")
    j["hit"] = (j["month"] >= j["start_month"]) & (j["month"] <= j["end_month"])
    hits = j[j["hit"]]
    first = hits.assign(midx=hits["month"].map(midx)).groupby("event_id").agg(first_idx=("midx", "min"))
    ev2 = ev.merge(first, left_on="event_id", right_index=True, how="left")
    ev2["detected"] = ev2["first_idx"].notna()
    ev2["ttd"] = ev2["first_idx"] - ev2["start_idx"]
    rec = ev2.groupby("event_type")["detected"].mean()
    return dict(
        alerts=len(a), alerts_per_cust_month=len(a) / n_cm,
        precision=hits["index"].nunique() / len(a),
        event_recall=ev2["detected"].mean(),
        recall_creep=rec.get("category_creep", np.nan), recall_spike=rec.get("category_spike", np.nan),
        recall_medical=rec.get("medical_emergency", np.nan),
        ttd_creep=ev2.loc[(ev2.event_type == "category_creep") & ev2.detected, "ttd"].median(),
        n_events=len(ev2),
    )


def f1(m: dict) -> float:
    p, r = m["precision"], m["event_recall"]
    return 0.0 if (p is None or np.isnan(p) or p + r == 0) else 2 * p * r / (p + r)


def average_precision(cells: pd.DataFrame, events: pd.DataFrame, months: pd.DatetimeIndex, score_col: str):
    """Threshold-free ranking quality of `score_col` over all eligible cells vs event windows.
    Returns (average_precision, base_rate); a random scorer gets ~ the base rate."""
    midx = {m: i for i, m in enumerate(months)}
    c = cells[cells["month"].map(midx) >= ELIGIBLE_FROM][["customer_id", "category", "month", score_col]].copy()
    c["label"] = False
    for e in _events(events, months).itertuples():
        m = (c["customer_id"] == e.customer_id) & (c["category"] == e.category) & \
            (c["month"] >= e.start_month) & (c["month"] <= e.end_month)
        c.loc[m, "label"] = True
    c = c.dropna(subset=[score_col]).sort_values(score_col, ascending=False)
    y = c["label"].to_numpy()
    if y.sum() == 0:
        return float("nan"), 0.0
    prec_at_k = np.cumsum(y) / (np.arange(len(y)) + 1)
    return float(prec_at_k[y].mean()), float(y.mean())
