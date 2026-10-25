import numpy as np
import pandas as pd
from scipy.optimize import linprog

from core import health_score as hs
from core import insights as ins
from core import savings_planner as sp
from data_gen.generate import build_dataset
from tests.sqlite_harness import connect, run

_CACHE = {}


def data():
    if "d" not in _CACHE:
        d = build_dataset(100, 36, seed=2024, sms_frac=0.0)
        con = connect(d)
        f = pd.read_sql_query("SELECT * FROM v_health_features WHERE n6 = 6", con, parse_dates=["month"])
        _CACHE.update(d=d, con=con, f=f, itype=d["customers"].set_index("customer_id")["income_type"])
    return _CACHE


# ------------------------------------------------------------------ health score
def _row(**kw):
    base = dict(customer_id="X", month=pd.Timestamp("2025-06-01"), savings_rate_6m=0.10, debt_ratio=0.20, emergency_months=3.0,
                expense_cv_6m=0.25, income_cv_6m=0.10, emi=1.0, avg_income_6m=50000.0, avg_expense_6m=40000.0,
                month_end_balance=60000.0, reserve_balance=60000.0)
    base.update(kw)
    return pd.DataFrame([base])


def test_weights_and_bounds():
    assert sum(v[0] for v in hs.COMPONENTS.values()) == 100
    s = hs.score(data()["f"], data()["itype"])
    for k, v in hs.COMPONENTS.items():
        assert s[k].between(0, v[0]).all(), k
    assert s["total"].between(0, 100).all()
    assert (s["total"] - s[list(hs.COMPONENTS)].sum(axis=1)).abs().max() < 0.2


def test_score_is_monotone_in_each_input():
    t = lambda **kw: hs.score(_row(**kw)).total.iloc[0]
    assert t(savings_rate_6m=0.02) < t(savings_rate_6m=0.10) < t(savings_rate_6m=0.25)
    assert t(debt_ratio=0.45) < t(debt_ratio=0.25) < t(debt_ratio=0.05)
    assert t(emergency_months=0.5) < t(emergency_months=3) < t(emergency_months=8)
    assert t(expense_cv_6m=0.6) < t(expense_cv_6m=0.3) < t(expense_cv_6m=0.05)
    assert t(income_cv_6m=0.6) < t(income_cv_6m=0.2) < t(income_cv_6m=0.0)


def test_perfect_and_worst_profiles():
    best = hs.score(_row(savings_rate_6m=0.4, debt_ratio=0.0, emergency_months=12, expense_cv_6m=0.0, income_cv_6m=0.0, emi=0.0),
                    pd.Series({"X": "govt"})).iloc[0]
    worst = hs.score(_row(savings_rate_6m=-0.3, debt_ratio=0.9, emergency_months=0.0, expense_cv_6m=0.9, income_cv_6m=0.9),
                     pd.Series({"X": "student"})).iloc[0]
    assert best.total == 100 and best.band == "Excellent"
    assert worst.total < 10 and worst.band == "Needs attention"


def test_top_drag_is_the_component_losing_most_points_and_is_explained():
    r = hs.score(_row(emergency_months=0.2, savings_rate_6m=0.25, debt_ratio=0.05, expense_cv_6m=0.1, income_cv_6m=0.0),
                 pd.Series({"X": "govt"})).iloc[0]
    assert r.top_drag == "emergency" and "Emergency fund" in r.explanation and "0.2 months" in r.explanation


def test_projection_never_lowers_score():
    p = hs.project_after_cut(_row().iloc[0], 5000.0)
    assert p["after"] >= p["before"] and p["gain"] >= 0


def test_scores_fall_when_income_is_lost():
    """Validity vs injected ground truth: after a job loss the score should drop."""
    d, f = data(), data()["f"]
    s = hs.score(f, data()["itype"])
    drops = []
    for e in d["d"]["injected_events"].query("event_type == 'job_loss'").itertuples():
        x = s[s.customer_id == e.customer_id].set_index("month").total
        st, en = pd.Timestamp(e.start_month), pd.Timestamp(e.end_month)
        before = x[(x.index >= st - pd.DateOffset(months=3)) & (x.index < st)]
        after = x[(x.index >= en) & (x.index <= en + pd.DateOffset(months=2))]
        if len(before) and len(after):
            drops.append(after.mean() < before.mean())
    assert len(drops) >= 3 and np.mean(drops) >= 0.75, drops


def test_personas_rank_sensibly():
    s = hs.score(data()["f"], data()["itype"])
    persona = data()["d"]["customers"].set_index("customer_id")["persona"]
    m = s.assign(p=s.customer_id.map(persona)).groupby("p").total.mean()
    assert m["student_parttime"] < m["frugal_saver"] and m["student_parttime"] < m["govt_stable"]
    inc = s.assign(p=s.customer_id.map(persona)).groupby("p").income.mean()
    assert inc["freelancer"] < inc["govt_stable"], "irregular freelance income should score lower on stability"


# ------------------------------------------------------------------ savings planner
def _cands(seed=0, n=12):
    rng = np.random.default_rng(seed)
    cats = list(sp.CUT_RULES) + ["Rent", "EMI", "Insurance", "Education", "Health"]
    pick = rng.choice(cats, size=n, replace=False)
    avg = rng.uniform(500, 15000, n)
    return pd.DataFrame({"category": pick, "avg_3m_spend": avg, "baseline_mean": avg * rng.uniform(0.7, 1.1, n)})


def _lp_optimum(cands, target):
    seg = sp._segments(cands)
    if seg.empty:
        return None
    res = linprog(c=seg["pain"].to_numpy(), A_ub=-np.ones((1, len(seg))), b_ub=[-target],
                  bounds=list(zip([0] * len(seg), seg["capacity"])), method="highs")
    return res.fun if res.success else None


def test_greedy_matches_linprog_optimum():
    for seed in range(8):
        c = _cands(seed)
        cap = sp.plan(c, 1e12)["max_possible"]
        for frac in (0.1, 0.5, 0.9):
            T = frac * cap
            p = sp.plan(c, T)
            assert p["feasible"] and abs(p["achieved"] - T) < 1e-6
            assert abs(p["total_pain"] - _lp_optimum(c, T)) < 1e-4 * max(1, p["total_pain"]), (seed, frac)


def test_plan_respects_rules_and_reports_infeasible_targets():
    c = _cands(3, n=15)
    p = sp.plan(c, 1e12)
    assert not p["feasible"] and p["shortfall"] > 0 and abs(p["achieved"] - p["max_possible"]) < 1e-6
    for i in p["items"]:
        assert i["category"] in sp.CUT_RULES, "essentials like Rent/EMI/Health must never be cut"
        assert i["cut"] <= sp.CUT_RULES[i["category"]][0] * i["current"] + 1e-6
    assert "more than your flexible spending" in sp.format_plan(p)


def test_overspent_category_is_cut_first():
    c = pd.DataFrame({"category": ["Shopping", "Shopping".replace("Shopping", "Food Delivery")], "avg_3m_spend": [10000.0, 10000.0],
                      "baseline_mean": [6000.0, 10000.0]})
    sp_rules = {"Shopping": (0.4, 1.5), "Food Delivery": (0.4, 1.5)}
    p = sp.plan(c, 3000.0, rules=sp_rules)
    cuts = {i["category"]: i["cut"] for i in p["items"]}
    assert cuts["Shopping"] > 2900 and cuts.get("Food Delivery", 0) < 100, cuts


def test_plan_from_real_customer_and_score_impact():
    d = data()
    cid = d["f"].customer_id.iloc[0]
    c = run(d["con"], "savings_candidates", cid=cid, month="2025-06-01")
    p = sp.plan(c, 5000.0)
    assert p["achieved"] <= 5000.0 + 1e-6 and "Reduce" in sp.format_plan(p)
    row = d["f"][(d["f"].customer_id == cid) & (d["f"].month == pd.Timestamp("2025-06-01"))].iloc[0]
    assert sp.score_impact(p, row)["gain"] >= 0


# ------------------------------------------------------------------ insights
def test_insights_are_consistent_with_query_results():
    d = data()
    cid = d["f"].customer_id.iloc[0]
    out = ins.generate_insights(lambda name, **kw: run(d["con"], name, **kw), cid, "2025-06-01")
    topics = {o["topic"] for o in out}
    assert {"largest_category", "top_unnecessary", "weekday_trend", "seasonal_pattern"} <= topics
    top = run(d["con"], "top_categories_month", cid=cid, month="2025-06-01").iloc[0]
    text = next(o["text"] for o in out if o["topic"] == "largest_category")
    assert top["category"] in text and f"{top['spend']:,.0f}" in text


def test_spend_story_detects_one_off_purchase_and_weekend_skew():
    drivers = pd.DataFrame({"category": ["Shopping", "Groceries"], "spend": [18000.0, 6000.0], "baseline_mean": [6000.0, 6000.0],
                            "delta": [12000.0, 0.0], "pct_vs_baseline": [200.0, 0.0], "z_score": [4.0, 0.0]})
    big = pd.DataFrame({"txn_date": ["2025-06-12"], "merchant": ["AMAZON"], "amount": [11000.0], "channel": ["Card"], "typical_ticket": [1500.0]})
    story = ins.spend_story("June 2025", drivers, lambda c: big)
    assert "Rs 12,000 more than your usual" in story, story
    assert "mostly one Rs 11,000 purchase at Amazon" in story, story
    wk = pd.DataFrame({"isodow": range(1, 8), "spend": [100, 100, 100, 100, 200, 500, 450]})
    assert "more on weekends" in ins.weekday_trend(wk)
    seas = pd.DataFrame({"moy": range(1, 13), "avg_expenses": 1.0, "avg_discretionary": 1.0,
                         "expense_index": 1.0, "discretionary_index": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1.5, 0.5]})
    assert "November" in ins.seasonal_pattern(seas)


def test_wording_fixes_periodic_bills_plurals_and_atm():
    drivers = pd.DataFrame({"category": ["Rent", "Insurance"], "spend": [20000.0, 6000.0], "baseline_mean": [16000.0, 3000.0],
                            "delta": [4000.0, 3000.0], "pct_vs_baseline": [25.0, 100.0], "z_score": [2.0, 2.0]})
    story = ins.spend_story("June 2025", drivers, lambda c: pd.DataFrame())
    assert "Fixed and periodic bills added another Rs 7,000" in story, story
    assert "a rent revision" in story and "a periodic premium falling due" in story
    assert "more or larger purchases" not in story and "biggest increases in your habits" not in story
    one = pd.DataFrame({"category": ["Rent"], "spend": [21910.0], "txns": [1], "share_pct": [33.0]})
    assert "1 transaction)" in ins.largest_category(one, "June 2025") and "1 transactions" not in ins.largest_category(one, "June 2025")
    top = pd.DataFrame({"merchant": ["ATM"], "category": ["Cash Withdrawal"], "spend": [6500.0], "txns": [4]})
    assert "ATM withdrawals" in ins.top_unnecessary(top) and "Atm" not in ins.top_unnecessary(top)


def test_story_separates_habit_changes_from_periodic_bills():
    drivers = pd.DataFrame({"category": ["Shopping", "Rent"], "spend": [18000.0, 20000.0], "baseline_mean": [6000.0, 16000.0],
                            "delta": [12000.0, 4000.0], "pct_vs_baseline": [200.0, 25.0], "z_score": [4.0, 2.0]})
    story = ins.spend_story("June 2025", drivers, lambda c: pd.DataFrame())
    assert "increases in your habits: Shopping (+Rs 12,000" in story and "Rent (" not in story
    assert "Fixed and periodic bills added another Rs 4,000 (likely a rent revision)" in story


def test_explanation_says_overspending_not_negative_savings():
    r = hs.score(_row(savings_rate_6m=-0.62, debt_ratio=0.0, emergency_months=6, expense_cv_6m=0.1, income_cv_6m=0.0, emi=0.0),
                 pd.Series({"X": "govt"})).iloc[0]
    assert "spent 62% more than you earned" in r.explanation and "-62%" not in r.explanation
