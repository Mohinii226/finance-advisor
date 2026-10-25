import numpy as np
import pandas as pd

from data_gen.generate import build_dataset
from ml.anomaly.evaluate import grid_from_transactions
from ml.anomaly.overspend import BALANCED, NAIVE, detect, with_
from ml.anomaly.run_detector import ALERT_COLUMNS, alerts_to_frame


def _synthetic_grid(n_cust=40, T=30, creep_cust="C0000", creep_start=16, mult=2.0, seed=0):
    rng = np.random.default_rng(seed)
    months = pd.date_range("2023-01-01", periods=T, freq="MS")
    rows = []
    for i in range(n_cust):
        cid, base = f"C{i:04d}", rng.uniform(3000, 9000)
        for t, m in enumerate(months):
            lvl = base * (mult if (cid == creep_cust and t >= creep_start) else 1.0)
            spend = lvl * rng.lognormal(0, 0.12)
            rows.append((cid, "Food Delivery", m, spend, max(1, round(spend / 400))))
    return pd.DataFrame(rows, columns=["customer_id", "category", "month", "spend", "txn_count"])


def test_planted_creep_is_detected_and_quiet_elsewhere():
    a = detect(_synthetic_grid(), BALANCED)
    mine = a[a.customer_id == "C0000"]
    start = pd.Timestamp("2023-01-01")
    assert len(mine) > 0 and mine.month.min() >= start + pd.DateOffset(months=16)
    assert mine.month.min() <= start + pd.DateOffset(months=21), "should catch within ~5 months"
    assert len(a[a.customer_id != "C0000"]) <= 2, "stable customers should not alert"


def test_alert_budget_is_respected():
    g = pd.concat([_synthetic_grid(), _synthetic_grid().assign(category="Shopping")])
    a = detect(g, with_(NAIVE, max_alerts=1))
    assert a.groupby(["customer_id", "month"]).size().max() <= 1


def test_message_is_human_readable():
    msg = detect(_synthetic_grid(), BALANCED).message.iloc[0]
    assert "Food Delivery spending is" in msg and "higher than your 6-month average" in msg


def test_naive_matches_sql_benchmark_view_exactly():
    """The pandas NAIVE config must reproduce v_overspend_alerts alert-for-alert."""
    from tests.sqlite_harness import connect
    d = build_dataset(30, 36, seed=5, sms_frac=0.0)
    con = connect(d)
    sql = pd.read_sql_query("SELECT customer_id, category, month FROM v_overspend_alerts", con)
    py = detect(grid_from_transactions(d["transactions"]), NAIVE)[["customer_id", "category", "month"]]
    a = set(zip(sql.customer_id, sql.category, pd.to_datetime(sql.month).dt.strftime("%Y-%m-%d")))
    b = set(zip(py.customer_id, py.category, py.month.dt.strftime("%Y-%m-%d")))
    assert len(a) > 0 and a == b, f"sql-only={len(a - b)} pandas-only={len(b - a)}"


def test_alerts_to_frame_matches_table_columns():
    f = alerts_to_frame(detect(_synthetic_grid(), BALANCED), BALANCED)
    assert list(f.columns) == ALERT_COLUMNS and len(f) > 0
    assert f.alert_type.isin(["spike", "sustained"]).all()
