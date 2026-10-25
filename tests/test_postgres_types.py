"""The same questions must give the same answers whether the database returns SQLite types or PostgreSQL types."""
import datetime as dt
import json
from decimal import Decimal

import numpy as np
import pandas as pd

from core import insights as ins
from core.service import AdvisorService
from llm.agent import AdvisorAgent
from llm.rulebased import RuleBasedClient
from llm.tools import ToolExecutor
from tests.chat_fixture import chat_env
from tests.pg_mimic import pg_like


def test_mimic_really_returns_postgres_style_types():
    e = chat_env()
    df = pg_like(e["run"])("top_categories_month", cid=e["cid"], month="2025-06-01")
    assert isinstance(df["spend"].iloc[0], Decimal) and isinstance(df["txns"].iloc[0], (int, np.integer))
    rng = pg_like(e["run"])("month_range")
    assert isinstance(rng["first_month"].iloc[0], dt.date) and not isinstance(rng["first_month"].iloc[0], pd.Timestamp)
    wd = pg_like(e["run"])("weekday_pattern", cid=e["cid"], start="2025-01-01", end="2026-01-01")
    assert isinstance(wd["isodow"].iloc[0], Decimal)


def _answers(runner, cid):
    s = AdvisorService(runner, cid)
    out = {
        "top": s.get_top_categories("last_month"), "total": s.get_category_total("food", "last_3_months"),
        "compare": s.compare_months("2025-10", "2025-11"), "why": s.explain_spending_change("this_month"),
        "alerts": s.get_alerts(), "health": s.get_health_score(), "savings": s.suggest_savings(3000),
        "insights": s.get_insights(),
    }
    return json.loads(json.dumps(out, default=str))


def test_service_answers_identical_with_postgres_style_types():
    e = chat_env()
    plain = _answers(e["run"], e["cid"])
    pg = _answers(pg_like(e["run"]), e["cid"])
    assert pg == plain


def test_full_chat_stack_works_with_postgres_style_types():
    e = chat_env()
    s = AdvisorService(pg_like(e["run"]), e["cid"])
    agent = AdvisorAgent(RuleBasedClient(s.as_of), ToolExecutor(s))
    for q in ["Where did I spend the most last month?", "Why did I spend more this month?", "Suggest ways to save Rs 3,000",
              "What is my financial health score?", "Which weekday do I spend the most on?"]:
        r = agent.chat(q)
        assert r.tool_calls and r.tool_calls[0]["ok"] and r.grounding.ok, (q, r.answer)


def test_health_score_and_planner_accept_decimal_inputs():
    e = chat_env()
    s = AdvisorService(pg_like(e["run"]), e["cid"])
    row = s._features(s.last_month)
    assert all(isinstance(row[c], (float, int)) or pd.isna(row[c]) for c in ("savings_rate_6m", "debt_ratio", "emergency_months"))
