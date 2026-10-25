"""Run: python -m tests.run_sql_checks  — cross-checks SQL views against pandas on a generated dataset."""
import sqlite3
import numpy as np
import pandas as pd

from data_gen.generate import build_dataset
from tests.sqlite_harness import connect, run

print("sqlite", sqlite3.sqlite_version)
data = build_dataset(n_customers=80, n_months=36, seed=2024)
con = connect(data)
t = data["transactions"].copy()
t["month"] = pd.to_datetime(t["txn_ts"]).dt.to_period("M").dt.to_timestamp()

# ---- 1. expense ledger excludes non-spending categories -------------------------------
exp_sql = pd.read_sql_query("SELECT category, SUM(amount) a FROM v_expense_txn GROUP BY category", con).set_index("category").a
assert "Investments" not in exp_sql.index and "Self Transfer" not in exp_sql.index
fin = t["category"].fillna("Uncategorized")
mask = (t.type == "debit") & ~fin.isin(["Investments", "Self Transfer"])
exp_pd = t[mask].groupby(fin[mask]).amount.sum()
assert np.allclose(exp_sql.sort_index().values, exp_pd.sort_index().values), "expense totals differ"
print("PASS expense ledger matches pandas (%d categories)" % len(exp_sql))

# ---- 2. rolling baseline equals pandas rolling (zero-filled, excl. current month) -----------
cid, cat = t[t.true_category == "Groceries"].customer_id.iloc[0], "Groceries"
months = pd.date_range(t.month.min(), t.month.max(), freq="MS")
s = (t[(t.customer_id == cid) & mask & (fin == cat)].groupby("month").amount.sum().reindex(months, fill_value=0))
exp_mean = s.shift(1).rolling(6, min_periods=1).mean()
exp_std = s.shift(1).rolling(6, min_periods=2).std()
b = pd.read_sql_query("SELECT month, baseline_mean, baseline_std, n_hist FROM v_category_baseline WHERE customer_id=? AND category=? ORDER BY month", con, params=(cid, cat))
b["month"] = pd.to_datetime(b["month"])
b = b.set_index("month").reindex(months)
ok = slice(7, None)   # skip the warm-up rows
assert np.allclose(b.baseline_mean.values[ok], exp_mean.values[ok], rtol=1e-6)
assert np.allclose(b.baseline_std.values[ok], exp_std.values[ok], rtol=1e-4)
assert (b.n_hist.values[ok] == 6).all()
print("PASS 6-month rolling mean/std match pandas for", cid, cat)

# ---- 3. monthly summary: income + reserve sanity ---------------------------------------------
ms = pd.read_sql_query("SELECT * FROM v_monthly_summary", con)
assert (ms.reserve_balance >= -1).all(), "reserve should never go negative"
sal = t[(t.type == "credit") & t.category.isin(["Salary", "Freelance Income", "Pension", "Allowance"])].groupby(["customer_id", "month"]).amount.sum()
chk = ms.assign(month=pd.to_datetime(ms.month)).set_index(["customer_id", "month"]).income
common = sal.index.intersection(chk.index)
assert np.allclose(chk.loc[common].values, sal.loc[common].values)
print("PASS monthly income matches pandas; reserve >= 0; rows:", len(ms))

# ---- 4. health features ------------------------------------------------------------------------
hf = pd.read_sql_query("SELECT * FROM v_health_features WHERE n6 = 6", con)
print("health feature quantiles (n6=6 rows):")
print(hf[["savings_rate_6m", "debt_ratio", "emergency_months", "expense_cv_6m", "income_cv_6m"]].describe().loc[["25%", "50%", "75%"]].round(2).to_string())

# ---- 5. every parametrised query executes -------------------------------------------------------
cid = ms.customer_id.iloc[0]
P = dict(cid=cid, month="2025-06-01", start="2025-01-01", end="2026-01-01", m1="2025-05-01", m2="2025-06-01",
         category="Shopping", limit=5)
for q in ["top_categories_month", "category_total", "month_compare", "weekday_pattern", "seasonal_index",
          "top_unnecessary_spend", "merchant_concentration", "spend_drivers", "largest_txns_in_category",
          "savings_candidates", "health_inputs", "alerts_for_month"]:
    r = run(con, q, **P)
    print(f"PASS query {q:26s} -> {len(r):3d} rows")

# ---- 6. detector benchmark against injected ground truth ------------------------------------------
prec = run(con, "detector_precision").iloc[0]
rec = run(con, "detector_recall")
print(f"\nSQL rule detector (>=25% & z>=2 & >=Rs500 vs 6-mo baseline)")
print(f"  alerts: {int(prec.n_alerts)}  inside a true event window: {int(prec.n_true_alerts)}  -> precision {prec.n_true_alerts / max(prec.n_alerts, 1):.1%}")
rec["recall"] = (rec.n_detected / rec.n_events).round(3)
print(rec.to_string(index=False))
print("\nTop false-positive sources (category, month-of-year):")
print(run(con, "false_positive_breakdown").head(8).to_string(index=False))
