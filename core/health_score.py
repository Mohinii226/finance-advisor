"""
Financial Health Score (0-100): transparent, rule-based and explainable.

Inputs come from the SQL view `v_health_features` (trailing 6-month windows):
    savings_rate_6m, debt_ratio, emergency_months, expense_cv_6m, income_cv_6m

Each component is mapped linearly onto 0..1 between a "worst" and a "best" anchor and multiplied by its weight.
The anchors are design choices (documented below), not learned from data; change them in COMPONENTS.

    component            weight   0 points at        full points at
    Savings rate           30     <= 0% saved         >= 20% saved
    Debt ratio (EMI/inc)   20     >= 50% of income    <= 10% of income
    Emergency fund         20     0 months expenses   >= 6 months
    Expense consistency    15     CV >= 0.50          CV <= 0.10
    Income stability       15     CV >= 0.50          CV <= 0.05   (80%) + income-type prior (20%)

Explainability: every score returns per-component points, the component losing the most points ("top drag"),
and a one-sentence explanation. `project_after_cut` shows what a savings plan would do to the score.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

COMPONENTS = {
    # name: (weight, worst, best, label, column in v_health_features)
    "savings":     (30, 0.00, 0.20, "Savings rate", "savings_rate_6m"),
    "debt":        (20, 0.50, 0.10, "Debt ratio", "debt_ratio"),
    "emergency":   (20, 0.0, 6.0, "Emergency fund", "emergency_months"),
    "consistency": (15, 0.50, 0.10, "Expense consistency", "expense_cv_6m"),
    "income":      (15, 0.50, 0.05, "Income stability", "income_cv_6m"),
}
INCOME_TYPE_PRIOR = {"govt": 1.0, "pension": 1.0, "salaried": 0.9, "freelance": 0.5, "student": 0.4}
BANDS = [(80, "Excellent"), (65, "Good"), (50, "Fair"), (0, "Needs attention")]
assert sum(v[0] for v in COMPONENTS.values()) == 100


def _lin(x, worst, best):
    """0 at `worst`, 1 at `best`, linear in between (works for either direction)."""
    return np.clip((np.asarray(x, dtype=float) - worst) / (best - worst), 0.0, 1.0)


def band(total: float) -> str:
    return next(name for cut, name in BANDS if total >= cut)


def score(features: pd.DataFrame, income_type: pd.Series | None = None) -> pd.DataFrame:
    """Score rows of v_health_features. Returns component POINTS, total, band, top drag and a sentence.

    Missing inputs are handled conservatively: no income -> 0 savings points; no EMI -> full debt points."""
    f = features.copy()
    pts = pd.DataFrame(index=f.index)

    sav = f["savings_rate_6m"].fillna(-1.0)                                  # no income at all -> worst
    pts["savings"] = _lin(sav, 0.00, 0.20) * COMPONENTS["savings"][0]
    debt = f["debt_ratio"].copy()
    debt = debt.where(debt.notna(), np.where(f["emi"] > 0, 1.0, 0.0))        # EMI but no income -> worst
    pts["debt"] = _lin(debt, 0.50, 0.10) * COMPONENTS["debt"][0]
    emer = f["emergency_months"].fillna(6.0)                                 # no spending history -> not penalised
    pts["emergency"] = _lin(emer, 0.0, 6.0) * COMPONENTS["emergency"][0]
    pts["consistency"] = _lin(f["expense_cv_6m"].fillna(0.5), 0.50, 0.10) * COMPONENTS["consistency"][0]

    inc_cv = _lin(f["income_cv_6m"].fillna(0.5), 0.50, 0.05)
    prior = income_type.reindex(f["customer_id"]).map(INCOME_TYPE_PRIOR).fillna(0.7).to_numpy() \
        if income_type is not None else np.full(len(f), 0.7)
    pts["income"] = (0.8 * inc_cv + 0.2 * prior) * COMPONENTS["income"][0]

    out = pd.concat([f[["customer_id", "month"]], pts.round(2)], axis=1)
    out["total"] = pts.sum(axis=1).round(1)
    out["band"] = out["total"].map(band)
    out["peer_percentile"] = (out.groupby("month")["total"].rank(pct=True) * 100).round(0)   # rank among customers that month
    lost = pd.DataFrame({k: COMPONENTS[k][0] - pts[k] for k in COMPONENTS})
    out["top_drag"] = lost.idxmax(axis=1)
    out["points_lost_top_drag"] = lost.max(axis=1).round(1)
    out["explanation"] = [_explain(r, f.loc[i]) for i, r in out.iterrows()]
    return out


def _explain(r, feat) -> str:
    k = r["top_drag"]
    label = COMPONENTS[k][3]
    detail = {
        "savings": lambda: ("no income was recorded in the last 6 months" if pd.isna(feat["savings_rate_6m"]) else
                            f"you spent {abs(feat['savings_rate_6m']) * 100:.0f}% more than you earned over the last 6 months"
                            if feat["savings_rate_6m"] < 0 else
                            f"you saved {feat['savings_rate_6m'] * 100:.0f}% of income over 6 months (target 20%)"),
        "debt": lambda: f"EMIs take {feat['debt_ratio'] * 100:.0f}% of income (comfortable is under 10-30%)"
                        if pd.notna(feat["debt_ratio"]) else "EMIs are paid with no recorded income",
        "emergency": lambda: f"you hold {feat['emergency_months']:.1f} months of expenses in cash and savings (target 6)",
        "consistency": lambda: f"monthly spending swings by {feat['expense_cv_6m'] * 100:.0f}% around its average",
        "income": lambda: f"monthly income varies by {feat['income_cv_6m'] * 100:.0f}% around its average",
    }[k]()
    return (f"Your financial health score is {r['total']:.0f}/100 ({r['band']}). "
            f"Biggest drag: {label}, because {detail} (-{r['points_lost_top_drag']:.0f} points).")


def project_after_cut(feature_row: pd.Series, monthly_cut: float, income_type: str | None = None) -> dict:
    """What would the score be if the customer cut `monthly_cut` Rs of spending every month?
    Re-computes ONLY the savings and emergency components (other inputs are assumed unchanged)."""
    f = feature_row.copy()
    inc, exp = f["avg_income_6m"], f["avg_expense_6m"]
    new_exp = max(exp - monthly_cut, 0.0)
    f["savings_rate_6m"] = (inc - new_exp) / inc if inc and inc > 0 else np.nan
    cash = f["month_end_balance"] + f["reserve_balance"]
    f["emergency_months"] = cash / new_exp if new_exp > 0 else 6.0
    row = pd.DataFrame([f])
    before = score(pd.DataFrame([feature_row]), pd.Series({feature_row["customer_id"]: income_type}) if income_type else None).iloc[0]
    after = score(row, pd.Series({f["customer_id"]: income_type}) if income_type else None).iloc[0]
    return dict(before=float(before["total"]), after=float(after["total"]), gain=float(after["total"] - before["total"]),
                band_before=before["band"], band_after=after["band"])


# ---------------------------------------------------------------- database output
def to_table_frame(scored: pd.DataFrame) -> pd.DataFrame:
    """Map to the `health_scores` table (component columns hold POINTS, not 0-1 scores)."""
    return pd.DataFrame({
        "customer_id": scored["customer_id"], "score_month": pd.to_datetime(scored["month"]).dt.date,
        "total_score": scored["total"], "savings_score": scored["savings"], "debt_score": scored["debt"],
        "emergency_score": scored["emergency"], "consistency_score": scored["consistency"],
        "income_score": scored["income"], "top_drag": scored["top_drag"],
    })


def main() -> None:
    """python -m core.health_score  -> compute scores for every customer-month and write `health_scores`."""
    import os
    from sqlalchemy import create_engine, text

    eng = create_engine(os.getenv("DATABASE_URL", "postgresql+psycopg2://finance:finance@localhost:5432/finance_advisor"))
    feats = pd.read_sql("SELECT * FROM v_health_features WHERE n6 = 6", eng, parse_dates=["month"])
    itype = pd.read_sql("SELECT customer_id, income_type FROM customers", eng).set_index("customer_id")["income_type"]
    for c in feats.columns.difference(["customer_id", "month"]):
        feats[c] = pd.to_numeric(feats[c], errors="coerce")
    out = to_table_frame(score(feats, itype))
    with eng.begin() as con:
        con.execute(text("DELETE FROM health_scores"))
        out.to_sql("health_scores", con, if_exists="append", index=False, method="multi", chunksize=5000)
    print(f"wrote {len(out):,} health scores; mean {out.total_score.mean():.1f}")


if __name__ == "__main__":
    main()
