"""
Explainable insights: sentences generated from SQL results, never from model guesses.

Every function takes the DataFrame returned by a named query in sql/analytics_queries.sql and returns text whose
numbers can be traced back to that DataFrame. `generate_insights` wires the queries together; it needs a
`run(name, **params) -> DataFrame` callable (see `make_runner`).
"""
from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd

from .queries import get_query

DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
MONTHS = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October",
          "November", "December"]


def _rs(x: float) -> str:
    return f"Rs {float(x):,.0f}"


# Fixed / periodic bills: a rise here is a bill revision or a premium falling due, not "more purchases".
PERIODIC_BILLS = {"Rent": "a rent revision", "Insurance": "a periodic premium falling due this month",
                  "EMI": "a change in your loan payment", "Education": "a fee payment",
                  "Mobile & Internet": "a plan or bill change", "Subscriptions": "a new or repriced subscription"}


def _plural(n: int, word: str) -> str:
    return f"{n} {word}" if n == 1 else f"{n} {word}s"


def _merchant(name: str) -> str:
    return "ATM withdrawals" if str(name).upper() == "ATM" else str(name).title()


def make_runner(engine) -> Callable[..., pd.DataFrame]:
    """Runner for a SQLAlchemy engine (PostgreSQL): run('top_categories_month', cid=..., month=...)."""
    from sqlalchemy import text

    def run(name: str, **params) -> pd.DataFrame:
        return pd.read_sql(text(get_query(name)), engine, params=params)
    return run


# ---------------------------------------------------------------- single-purpose insights
def largest_category(df: pd.DataFrame, month_label: str) -> str | None:
    """df = top_categories_month"""
    if df.empty:
        return None
    r = df.iloc[0]
    return (f"Your largest spending category in {month_label} was {r['category']} at {_rs(r['spend'])} "
            f"({float(r['share_pct']):.0f}% of the month's spending across {_plural(int(r['txns']), 'transaction')}).")


def top_unnecessary(df: pd.DataFrame, n: int = 5, period: str = "the last 6 months") -> str | None:
    """df = top_unnecessary_spend (merchants inside non-essential categories)"""
    if df.empty:
        return None
    top = df.head(n)
    items = "; ".join(f"{_merchant(r['merchant'])} ({r['category']}, {_rs(r['spend'])})" for _, r in top.iterrows())
    total = float(top["spend"].sum())
    return (f"Your top {len(top)} discretionary expenses over {period} were: {items}. Together that is {_rs(total)}, "
            f"or about {_rs(total / 6)} a month, spending you could trim without touching essentials.")


def weekday_trend(df: pd.DataFrame) -> str | None:
    """df = weekday_pattern (variable spending only; fixed bills excluded)"""
    if df.empty or float(df["spend"].sum()) <= 0:
        return None
    d = df.assign(day=lambda x: x["isodow"].astype(int).map(lambda i: DAYS[i - 1]), spend=lambda x: x["spend"].astype(float))
    total = d["spend"].sum()
    peak = d.sort_values("spend", ascending=False).iloc[0]
    weekend = d[d["isodow"].astype(int) >= 6]["spend"].sum() / total * 100
    lowest = d.sort_values("spend").iloc[0]
    skew = ("You spend noticeably more on weekends" if weekend > 36 else
            "Your spending is spread fairly evenly across the week" if weekend > 24 else "You spend mostly on weekdays")
    return (f"{skew}: {weekend:.0f}% of your day-to-day spending happens on Saturday and Sunday (an even week would be 29%). "
            f"{peak['day']} is your biggest day ({_rs(peak['spend'])} over the last 6 months) and {lowest['day']} your quietest.")


def seasonal_pattern(df: pd.DataFrame) -> str | None:
    """df = seasonal_index (average discretionary spend by calendar month vs the customer's overall average)"""
    if df.empty or len(df) < 6:
        return None
    d = df.assign(moy=lambda x: x["moy"].astype(int), idx=lambda x: x["discretionary_index"].astype(float))
    hi, lo = d.sort_values("idx", ascending=False).iloc[0], d.sort_values("idx").iloc[0]
    if hi["idx"] < 1.15:
        return "Your discretionary spending is steady through the year, with no strong seasonal peak."
    peaks = d[d["idx"] >= 1.15].sort_values("idx", ascending=False)
    names = ", ".join(f"{MONTHS[m - 1]} ({i:.2f}x)" for m, i in zip(peaks["moy"].head(3), peaks["idx"].head(3)))
    return (f"Your discretionary spending runs above your average in {names}; "
            f"{MONTHS[int(lo['moy']) - 1]} is your leanest month ({lo['idx']:.2f}x).")


def spend_story(month_label: str, drivers: pd.DataFrame, largest: Callable[[str], pd.DataFrame]) -> str | None:
    """'Why did I spend more this month?'  drivers = spend_drivers; largest(category) -> largest_txns_in_category."""
    if drivers.empty:
        return None
    d = drivers.assign(spend=lambda x: x["spend"].astype(float), baseline_mean=lambda x: x["baseline_mean"].astype(float),
                       delta=lambda x: x["delta"].astype(float))
    total, base = d["spend"].sum(), d["baseline_mean"].sum()
    diff = total - base
    if abs(diff) < 0.05 * max(base, 1):
        return f"Your spending in {month_label} ({_rs(total)}) was in line with your usual {_rs(base)}."
    direction = "more" if diff > 0 else "less"
    behavioural = d[~d["category"].isin(PERIODIC_BILLS)]
    bills_up = d[d["category"].isin(PERIODIC_BILLS) & (d["delta"] > 0)]
    movers = behavioural[behavioural["delta"] > 0].head(3) if diff > 0 else behavioural.sort_values("delta").head(3)
    parts = []
    for _, r in movers.iterrows():
        txt = f"{r['category']} ({'+' if r['delta'] > 0 else '-'}{_rs(abs(r['delta']))}"
        if r["delta"] > 0:
            big = largest(r["category"])
            if len(big):
                top, typ = big.iloc[0], big["typical_ticket"].iloc[0]
                if float(top["amount"]) >= 0.6 * float(r["delta"]) and (pd.isna(typ) or float(top["amount"]) >= 2.5 * float(typ)):
                    txt += f", mostly one {_rs(top['amount'])} purchase at {_merchant(top['merchant'])}"
                else:
                    txt += ", from more or larger purchases than usual"
        parts.append(txt + ")")
    text = (f"You spent {_rs(total)} in {month_label}, {_rs(abs(diff))} {direction} than your usual {_rs(base)} "
            f"({abs(diff) / max(base, 1) * 100:.0f}%).")
    if parts:
        text += f" The biggest {'increases' if diff > 0 else 'decreases'} in your habits: {'; '.join(parts)}."
    if diff > 0 and len(bills_up):
        text += (f" Fixed and periodic bills added another {_rs(bills_up['delta'].sum())} "
                 f"(likely {'; '.join(PERIODIC_BILLS[c] for c in bills_up['category'])}), which is not a change in your habits.")
    return text


def robust_drivers(hist: pd.DataFrame, month: str, min_history: int = 3) -> pd.DataFrame:
    """Category deltas vs the customer's usual level, where "usual" ignores one-off spikes.

    hist = category_history rows (category, month, spend). Usual = mean of the previous (up to 6) months after dropping
    the single largest, so a one-off medical bill or trip does not inflate the baseline. Same idea as the detector's trimmed mean.
    """
    h = hist.assign(month=pd.to_datetime(hist["month"]), spend=hist["spend"].astype(float))
    m = pd.Timestamp(month)
    rows = []
    for cat, g in h.groupby("category"):
        if cat == "Uncategorized":
            continue
        prev = g[g["month"] < m].sort_values("month").tail(6)["spend"].to_numpy()
        if len(prev) < min_history:
            continue
        usual = float(prev.mean())
        if len(prev) >= 4:                       # drop the top month ONLY if it is a genuine one-off (avoids a downward bias)
            srt = np.sort(prev)
            rest = srt[:-1]
            if srt[-1] > 3 * max(float(np.median(rest)), 1.0) and srt[-1] - rest.mean() > 1500:
                usual = float(rest.mean())
        spend = float(g.loc[g["month"] == m, "spend"].sum())
        if spend == 0 and usual == 0:
            continue
        rows.append(dict(category=cat, spend=spend, baseline_mean=usual, delta=spend - usual,
                         pct_vs_baseline=(spend / usual - 1) * 100 if usual > 0 else None))
    out = pd.DataFrame(rows, columns=["category", "spend", "baseline_mean", "delta", "pct_vs_baseline"])
    return out.sort_values("delta", ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------- orchestration
def generate_insights(run: Callable[..., pd.DataFrame], cid: str, month: str) -> list[dict]:
    """month = first day of the month, 'YYYY-MM-01'. Returns [{'topic', 'text'}, ...] (empty ones are skipped)."""
    m = pd.Timestamp(month)
    start, end = (m - pd.DateOffset(months=6)).strftime("%Y-%m-%d"), (m + pd.DateOffset(months=1)).strftime("%Y-%m-%d")
    label = f"{MONTHS[m.month - 1]} {m.year}"
    out = [
        ("largest_category", largest_category(run("top_categories_month", cid=cid, month=month), label)),
        ("why_spend_changed", spend_story(label, robust_drivers(run("category_history", cid=cid, month=month,
                                                                    start=(m - pd.DateOffset(months=6)).strftime("%Y-%m-%d")), month),
                                          lambda c: run("largest_txns_in_category", cid=cid, month=month, category=c, limit=3))),
        ("top_unnecessary", top_unnecessary(run("top_unnecessary_spend", cid=cid, start=start, end=end, limit=5))),
        ("weekday_trend", weekday_trend(run("weekday_pattern", cid=cid, start=start, end=end))),
        ("seasonal_pattern", seasonal_pattern(run("seasonal_index", cid=cid))),
    ]
    return [dict(topic=t, text=x) for t, x in out if x]
