"""
Personalised savings planner.

Problem: the customer wants to save T rupees per month. Choose how much to cut in each category so that
total cuts >= T while minimising *lifestyle pain*.

    minimise   sum_i pain_i * cut_i
    subject to sum_i cut_i >= T ,   0 <= cut_i <= capacity_i

With a single linear constraint this LP is solved EXACTLY by taking the cheapest-pain rupees first (a fractional
knapsack). tests/test_core.py checks the greedy solution against scipy's `linprog`.

Realism rules (assumptions, tune in CUT_RULES):
  * Essentials (rent, EMI, insurance, education, health) are never cut; groceries, transport, fuel and utilities
    can only be trimmed by 10%.
  * Discretionary categories can be cut by 15-50%, with different pain per rupee (a streaming subscription is
    easier to drop than dining out with friends).
  * Money a customer is spending ABOVE their own normal level (6-month baseline) is the cheapest to cut:
    returning to the usual habit costs 40% of the normal pain.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# category: (max share of current spend that can be cut, pain per rupee cut)
CUT_RULES: dict[str, tuple[float, float]] = {
    "Subscriptions": (0.50, 1.0), "Food Delivery": (0.40, 1.0), "Entertainment": (0.40, 1.3),
    "Shopping": (0.35, 1.5), "Dining Out": (0.30, 1.8), "Travel": (0.30, 2.0), "Cash Withdrawal": (0.15, 2.0),
    "Personal Care": (0.20, 2.5), "Gifts & Donations": (0.15, 3.0), "Transfers": (0.10, 3.0),
    "Groceries": (0.10, 4.0), "Utilities": (0.10, 4.0), "Mobile & Internet": (0.10, 4.0),
    "Transport": (0.10, 4.5), "Fuel": (0.10, 4.5),
}   # anything not listed (Rent, EMI, Insurance, Education, Health, ...) has no cut capacity
EXCESS_PAIN_DISCOUNT = 0.4


def _segments(cands: pd.DataFrame, rules=CUT_RULES, excess_discount=EXCESS_PAIN_DISCOUNT) -> pd.DataFrame:
    rows = []
    for r in cands.itertuples():
        cap_pct, pain = rules.get(r.category, (0.0, np.inf))
        capacity = cap_pct * float(r.avg_3m_spend)
        if capacity <= 0:
            continue
        base = float(r.baseline_mean) if pd.notna(getattr(r, "baseline_mean", np.nan)) else float(r.avg_3m_spend)
        excess = min(capacity, max(float(r.avg_3m_spend) - base, 0.0))
        if excess > 0:
            rows.append((r.category, "above your usual level", excess, pain * excess_discount))
        if capacity - excess > 1e-9:
            rows.append((r.category, "flexible trim", capacity - excess, pain))
    return pd.DataFrame(rows, columns=["category", "kind", "capacity", "pain"])


def plan(candidates: pd.DataFrame, target: float, rules=CUT_RULES, excess_discount=EXCESS_PAIN_DISCOUNT) -> dict:
    """candidates columns: category, avg_3m_spend, baseline_mean (optional). Returns a plan dict."""
    seg = _segments(candidates, rules, excess_discount).sort_values("pain", kind="stable")
    remaining, take = float(target), []
    for r in seg.itertuples():
        if remaining <= 1e-9:
            break
        x = min(r.capacity, remaining)
        take.append((r.category, r.kind, x, r.pain)); remaining -= x
    cuts = pd.DataFrame(take, columns=["category", "kind", "cut", "pain"])
    cur = candidates.set_index("category")["avg_3m_spend"].astype(float)
    base = candidates.set_index("category")["baseline_mean"].astype(float) if "baseline_mean" in candidates else cur
    items = []
    for cat, g in cuts.groupby("category", sort=False):
        c = float(g["cut"].sum())
        items.append(dict(category=cat, current=float(cur[cat]), cut=c, new_level=float(cur[cat]) - c,
                          pct_cut=100 * c / float(cur[cat]), annual_saving=12 * c,
                          includes_excess=bool((g["kind"] == "above your usual level").any()),
                          above_usual=float(max(cur[cat] - (base[cat] if pd.notna(base[cat]) else cur[cat]), 0.0)),
                          baseline=float(base[cat]) if pd.notna(base[cat]) else float(cur[cat]),
                          pain=float((g["cut"] * g["pain"]).sum())))
    achieved = float(cuts["cut"].sum()) if len(cuts) else 0.0
    return dict(target=float(target), achieved=achieved, feasible=remaining <= 1e-6, shortfall=max(remaining, 0.0),
                max_possible=float(seg["capacity"].sum()) if len(seg) else 0.0,
                total_pain=float(sum(i["pain"] for i in items)), items=items)


def _r(x: float, to: int = 10) -> float:
    return float(round(x / to) * to)


MATERIAL_ABOVE_USUAL = 500.0      # only call out "above your usual level" when it is at least this (and >= 10% of normal)
MIN_LINE = 100.0                  # trims smaller than this are rolled into one line
MAX_LINES = 6


def format_plan(p: dict) -> str:
    """Human-readable plan, in the style of the product spec."""
    if not p["items"]:
        return "I could not find flexible spending to trim; most of your spending is on essentials."
    items = sorted(p["items"], key=lambda i: -i["cut"])
    shown = [i for i in items if i["cut"] >= MIN_LINE][:MAX_LINES]
    rest = [i for i in items if i not in shown]
    lines = []
    for i in shown:
        why = ""
        if i["above_usual"] >= max(MATERIAL_ABOVE_USUAL, 0.10 * i["baseline"]):
            why = (f" Your {i['category']} spending is Rs {_r(i['above_usual']):,.0f} above your usual level, "
                   f"so this is the easiest place to start.")
        lines.append(f"- Reduce {i['category']} by Rs {_r(i['cut']):,.0f}/month (Rs {_r(i['current']):,.0f} -> Rs {_r(i['new_level']):,.0f}); "
                     f"that adds Rs {_r(i['annual_saving'], 100):,.0f} to your yearly savings.{why}")
    if rest:
        tot = sum(i["cut"] for i in rest)
        lines.append(f"- Small trims across {', '.join(i['category'] for i in rest)} add another Rs {_r(tot):,.0f}/month.")
    head = (f"To save Rs {p['target']:,.0f} a month, here is the lowest-impact plan:" if p["feasible"] else
            f"Rs {p['target']:,.0f} a month is more than your flexible spending allows. The most I can find is "
            f"Rs {_r(p['max_possible']):,.0f}/month (short by Rs {_r(p['shortfall']):,.0f}); beyond that you would need to "
            f"look at essentials such as rent, EMIs or income.")
    return head + "\n" + "\n".join(lines)


def score_impact(p: dict, feature_row: pd.Series, income_type: str | None = None) -> dict:
    """How much would following the plan lift the health score? (savings + emergency components only)"""
    from .health_score import project_after_cut
    return project_after_cut(feature_row, p["achieved"], income_type)
