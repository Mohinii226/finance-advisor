import os
from pathlib import Path

import numpy as np
import pandas as pd

from app import data_access as da
from app.chat import SAMPLE_QUESTIONS, chat_turn, history_for_model
from core.service import AdvisorService, ToolError
from tests.chat_fixture import chat_env
from tests.fake_streamlit import install_fake_plotly_if_missing, run_app
from tests.pg_mimic import pg_like
from tests.test_forecasting import _with_forecast_table

ROOT = Path(__file__).resolve().parents[1]


def _env():
    e = _with_forecast_table()                       # alerts + forecasts tables present
    return e, e["cid"]


def _runners():
    e, cid = _env()
    return [("sqlite types", e["run"]), ("postgres-style types", pg_like(e["run"]))], cid


# ------------------------------------------------------------------ data access (same assertions under both type systems)
def test_data_access_functions_under_both_type_systems():
    runners, cid = _runners()
    for label, run in runners:
        svc = AdvisorService(run, cid)
        month = svc.last_month
        cust = da.list_customers(run)
        assert all(l.startswith(i) for l, i in zip(cust["label"], cust["customer_id"])) and cid in set(cust["customer_id"]), label
        months = da.available_months(svc)
        assert months[0] == svc.last_month and months[-1] == svc.first_month + pd.DateOffset(months=5) and months == sorted(months, reverse=True)
        h = da.spend_history(run, cid, month, 12)
        assert len(h) == 12 and h["month"].is_monotonic_increasing and h["expenses"].dtype.kind == "f", label
        cat = da.category_history(run, cid, month, 12, top_n=5)
        assert cat["category"].nunique() <= 6 and cat["spend"].dtype.kind == "f"
        full = pd.to_numeric(run("category_history", cid=cid, month=da.month_str(month),
                                 start=da.month_str(month - pd.DateOffset(months=11)))["spend"]).sum()
        assert abs(cat["spend"].sum() - full) < 0.05, "grouping into 'Other' must not lose money"
        wd = da.weekday_pattern(run, cid, month)
        assert list(wd["day"]) == da.DAY_NAMES and wd["spend"].dtype.kind == "f"
        seas = da.seasonal_index(run, cid)
        assert len(seas) == 12 and list(seas["month_name"][:2]) == ["Jan", "Feb"]
        tm = da.top_merchants(run, cid, month)
        assert {"merchant", "category", "spend", "txns"} <= set(tm.columns) and tm["spend"].is_monotonic_decreasing
        lt = da.largest_transactions(run, cid, month)
        assert lt["amount"].is_monotonic_decreasing and lt["txn_date"].str.match(r"^\d{2} [A-Z][a-z]{2}$").all()


def test_safe_wrapper_and_money():
    ok, v = da.safe(lambda: 5)
    assert ok and v == 5
    ok, msg = da.safe(lambda: (_ for _ in ()).throw(ToolError("nothing yet")))
    assert not ok and msg == "nothing yet"
    ok, msg = da.safe(lambda: 1 / 0)
    assert not ok and "ZeroDivisionError" in msg and "division" not in msg.lower().replace("zerodivisionerror", "")
    assert da.money(12345.6) == "Rs 12,346" and da.money(None) == "n/a" and da.money(float("nan")) == "n/a"


def test_model_lab_loader_reads_repo_docs_and_tolerates_missing():
    lab = da.load_model_lab(ROOT)
    assert lab["forecast"] and "WAPE" in lab["forecast"] and lab["categorizer_metrics"] and lab["categorizer_per_class"] is not None
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        empty = da.load_model_lab(Path(d))
    assert empty["forecast"] is None and empty["categorizer_metrics"] is None and empty["confusion_png"] is None


# ------------------------------------------------------------------ charts
def test_charts_build_with_expected_traces():
    install_fake_plotly_if_missing()
    from app import charts
    e, cid = _env()
    svc, run = AdvisorService(e["run"], cid), e["run"]
    m = svc.last_month
    hist = da.spend_history(run, cid, m, 12)
    fc = svc.get_spending_forecast()
    assert len(charts.spend_trend(hist, fc).data) == 3 and len(charts.spend_trend(hist, None).data) == 2
    assert len(charts.spend_trend(hist.iloc[0:0], None).data) == 1
    top = svc.get_top_categories(da.month_str(m), 6)["top_categories"]
    assert len(charts.category_donut(top).data) == 1
    cat = da.category_history(run, cid, m)
    assert len(charts.category_stack(cat).data) == cat["category"].nunique()
    assert len(charts.weekday_bar(da.weekday_pattern(run, cid, m)).data) == 1
    assert len(charts.seasonal_line(da.seasonal_index(run, cid)).data) == 2
    h = svc.get_health_score(da.month_str(m))
    assert len(charts.health_gauge(h["total"], h["band"]).data) == 1 and len(charts.components_bar(h["components"]).data) == 2


# ------------------------------------------------------------------ chat
def test_chat_turn_and_history_trimming():
    e, cid = _env()
    svc = AdvisorService(e["run"], cid)
    r = chat_turn(svc, [], SAMPLE_QUESTIONS[0])
    assert r["grounded"] and r["tool_calls"][0]["name"] == "get_top_categories" and r["client"] == "RuleBasedClient" and "Rs" in r["answer"]
    for q in SAMPLE_QUESTIONS:
        out = chat_turn(svc, [], q)
        assert out["tool_calls"] and out["tool_calls"][0]["ok"] and out["grounded"], (q, out["answer"])
    msgs = [{"role": "user" if i % 2 == 0 else "assistant", "content": f"m{i}", "tools": ["x"]} for i in range(40)]
    h = history_for_model(msgs, max_turns=3)
    assert len(h) == 6 and set(h[0]) == {"role", "content"} and h[-1]["content"] == "m39"


# ------------------------------------------------------------------ the Streamlit script, end to end against the fake
def test_app_renders_every_tab_without_errors():
    e, cid = _env()
    st, _ = run_app(e["run"])
    assert not st.named("error"), st.named("error")
    labels = [a[0] for a, _ in st.named("metric")]
    assert any(l.startswith("Health score") for l in labels) and "Expected next month" in labels and "Overspending alerts" in labels
    assert len(st.named("plotly_chart")) == 7, "overview(2) + spending(3) + health(2) charts expected"
    assert st.named("set_page_config")[0][1]["layout"] == "wide"
    assert any("Savings planner" in t for t in st.texts()) and any("Evaluation artefacts" in t for t in st.texts())
    tabs = [a[0] for a, _ in st.named("tabs")]
    assert ["Overview", "Spending", "Health & savings", "Alerts & insights", "Chat", "Model lab"] in tabs
    assert not any("n/a" == str(a[1]) for a, _ in st.named("metric")), "with all tables present no metric should be n/a"


def test_app_runs_under_postgres_style_types():
    e, _ = _env()
    st, _ = run_app(pg_like(e["run"]))
    assert not st.named("error") and len(st.named("plotly_chart")) == 7


def test_app_chat_typed_question_and_sample_button():
    e, _ = _env()
    cid = da.list_customers(e["run"])["customer_id"].iloc[0]          # the sidebar selects the first customer by default
    st, _ = run_app(e["run"], inputs={"chat_input": ["Where did I spend the most last month?"]})
    msgs = st.session_state["chats"][cid]
    assert [m["role"] for m in msgs] == ["user", "assistant"] and "Rs" in msgs[1]["content"] and msgs[1]["tools"] == ["get_top_categories"]
    st2, _ = run_app(e["run"], inputs={"buttons": {SAMPLE_QUESTIONS[5]: True}})        # click -> rerun -> answered
    m2 = st2.session_state["chats"][cid]
    assert m2[0]["content"] == SAMPLE_QUESTIONS[5] and m2[1]["tools"] == ["get_health_score"]


def test_app_degrades_gracefully_when_tables_are_missing():
    e, cid = _env()

    def flaky(name, **kw):
        if name in ("alerts_table_for_month", "forecast_for_customer", "health_inputs"):
            raise RuntimeError("relation does not exist")
        return e["run"](name, **kw)
    st, _ = run_app(flaky)
    assert not st.named("error")
    na = [a[0] for a, _ in st.named("metric") if str(a[1]) == "n/a"]
    assert {"Health score", "Expected next month", "Overspending alerts"} <= set(na), na
    assert any(c for c in st.named("info")), "missing pieces should be explained, not hidden"


def test_app_shows_friendly_message_when_database_is_down():
    def dead(name, **kw):
        raise ConnectionError("could not connect to server")
    st, _ = run_app(dead)
    errs = [a[0] for a, _ in st.named("error")]
    assert errs and "docker compose up -d db" in errs[0]
    assert not st.named("tabs"), "nothing else should render when the database is unreachable"


# ------------------------------------------------------------------ real Streamlit (runs only where it is installed)
def test_real_streamlit_app_starts_and_reports_unreachable_database():
    try:
        from streamlit.testing.v1 import AppTest
    except ImportError:
        import pytest
        pytest.skip("streamlit not installed in this environment; this check runs where it is")
    os.environ["DATABASE_URL"] = "postgresql+psycopg2://nobody:nopass@127.0.0.1:1/none"
    try:
        at = AppTest.from_file(str(ROOT / "app" / "streamlit_app.py"), default_timeout=30).run()
    finally:
        os.environ.pop("DATABASE_URL", None)
    assert not at.exception, at.exception
    assert any("docker compose up -d db" in e.value for e in at.error)


def test_slider_default_is_always_valid_for_any_income():
    """Streamlit raises if a slider default is outside its range, so check the real function for extreme incomes."""
    for income in (0.0, 1500.0, 3000.0, 8000.0, 14000.0, 55000.0, 400000.0):
        lo, hi, default = da.slider_bounds(income)
        assert lo == 500 and lo <= default <= hi and default % 500 == 0 and hi % 500 == 0, (income, lo, default, hi)
