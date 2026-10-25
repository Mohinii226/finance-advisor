import json

import numpy as np
import pandas as pd

from api.handlers import (ApiError, Backend, BadRequest, ChatSessions, NotFound, Unauthorized, check_api_key)
from core import insights as ins
from core.service import AdvisorService, ToolError, clean_value
from llm import grounding
from llm.agent import AdvisorAgent
from llm.rulebased import HELP, RuleBasedClient
from llm.testing import ScriptedClient, text, tool_use
from llm.tools import TOOL_SCHEMAS, TOOLS, ToolExecutor, sanitize, validate_args
from tests.chat_fixture import chat_env


def _svc(cid=None):
    e = chat_env()
    return AdvisorService(e["run"], cid or e["cid"])


def _spend_pd(cid, start, end, cat=None):
    t = chat_env()["txns"]
    d = t[(t.customer_id == cid) & (t.type == "debit") & (~t.true_category.isin(["Investments", "Self Transfer"]))].copy()
    d["cat"] = d["category"].fillna("Uncategorized")
    d = d[~d["cat"].isin(["Investments", "Self Transfer"])]
    d = d[(pd.to_datetime(d.txn_date) >= start) & (pd.to_datetime(d.txn_date) < end)]
    return d if cat is None else d[d["cat"] == cat]


# ------------------------------------------------------------------ service
def test_period_and_month_resolution():
    s = _svc()
    assert s.resolve_month(None) == s.last_month and s.resolve_month("last_month") == s.last_month - pd.DateOffset(months=1)
    assert s.resolve_month("2025-03") == pd.Timestamp("2025-03-01")
    for bad in ("2031-01", "banana", "2025-13"):
        try:
            s.resolve_month(bad); assert False, bad
        except ToolError:
            pass
    start, end, label = s.resolve_period("last_3_months")
    assert start == s.last_month - pd.DateOffset(months=2) and end == s.last_month + pd.DateOffset(months=1)


def test_category_resolution_aliases_typos_unknown():
    s = _svc()
    assert set(s.resolve_categories("food")) == {"Food Delivery", "Dining Out", "Groceries"}
    assert s.resolve_categories("shoping") == ["Shopping"] and s.resolve_categories("SHOPPING") == ["Shopping"]
    try:
        s.resolve_categories("spaceships"); assert False
    except ToolError as e:
        assert "Known categories" in str(e)


def test_service_numbers_match_independent_pandas_calculation():
    e = chat_env(); s = _svc()
    r = s.get_category_total("Shopping", "last_6_months")
    start, end, _ = s.resolve_period("last_6_months")
    exp = _spend_pd(e["cid"], start, end, "Shopping")
    assert abs(r["combined_total"] - exp.amount.sum()) < 0.05 and r["categories"][0]["transactions"] == len(exp)
    top = s.get_top_categories("2025-06", 20)
    exp_m = _spend_pd(e["cid"], pd.Timestamp("2025-06-01"), pd.Timestamp("2025-07-01")).groupby("cat").amount.sum().sort_values(ascending=False)
    assert top["top_categories"][0]["category"] == exp_m.index[0] and abs(top["total_spend"] - exp_m.sum()) < 0.05


def test_service_is_scoped_to_one_customer():
    e = chat_env()
    a, b = _svc(e["cid"]), _svc(e["other"])
    assert a.profile["customer_id"] != b.profile["customer_id"]
    ra = a.get_category_total("rent", "all")["combined_total"]
    rb = b.get_category_total("rent", "all")["combined_total"]
    assert abs(ra - _spend_pd(e["cid"], a.first_month, a.last_month + pd.DateOffset(months=1), "Rent").amount.sum()) < 0.05
    assert ra != rb or ra == 0
    try:
        AdvisorService(e["run"], "C9999"); assert False
    except ToolError:
        pass


def test_savings_and_health_through_service():
    s = _svc()
    for bad in (-5, 0, 1e12):
        try:
            s.suggest_savings(bad); assert False, bad
        except ToolError:
            pass
    r = s.suggest_savings(2000)
    assert r["planned_saving"] <= 2000.01 and "Reduce" in r["plan_text"]
    h = s.get_health_score()
    assert abs(sum(c["points"] for c in h["components"].values()) - h["total"]) < 0.2


def test_alerts_available_and_graceful_when_missing():
    s = _svc()
    assert isinstance(s.get_alerts()["alerts"], list)

    def broken(name, **kw):
        if name == "alerts_table_for_month":
            raise RuntimeError("no such table: alerts")
        return chat_env()["run"](name, **kw)
    try:
        AdvisorService(broken, chat_env()["cid"]).get_alerts(); assert False
    except ToolError as e:
        assert "not available" in str(e)


def test_robust_drivers_ignore_one_off_spikes_without_biasing_normal_months():
    months = pd.date_range("2025-01-01", periods=7, freq="MS")
    spike = pd.DataFrame({"category": "Health", "month": months, "spend": [500, 600, 20000, 550, 520, 580, 900]})
    d = ins.robust_drivers(spike, "2025-07-01").iloc[0]
    assert d.baseline_mean < 700, "a one-off 20,000 bill must not inflate the usual level"
    normal = pd.DataFrame({"category": "Dining Out", "month": months, "spend": [2000, 2600, 1800, 2400, 3000, 2200, 2500]})
    d2 = ins.robust_drivers(normal, "2025-07-01").iloc[0]
    assert abs(d2.baseline_mean - np.mean([2000, 2600, 1800, 2400, 3000, 2200])) < 1e-6, "normal noise must not be trimmed"
    assert ins.robust_drivers(normal.head(2), "2025-02-01").empty, "needs at least 3 months of history"


def test_clean_value_handles_database_types():
    from decimal import Decimal
    assert clean_value(Decimal("12.345")) == 12.35 and clean_value(np.int64(3)) == 3
    assert clean_value(float("nan")) is None and clean_value(np.float64("inf")) is None
    assert clean_value(pd.Timestamp("2025-01-05")) == "2025-01-05"


# ------------------------------------------------------------------ tools
def test_tool_schemas_are_well_formed_and_dispatched():
    names = [t["name"] for t in TOOLS]
    assert len(names) == len(set(names)) == 9
    ex = ToolExecutor(_svc())
    for t in TOOLS:
        assert t["description"] and t["input_schema"]["additionalProperties"] is False
        assert "customer" not in json.dumps(t["input_schema"]).lower(), "customer id must never be a tool argument"
        assert t["name"] in ex._dispatch


def test_argument_validation():
    sch = TOOL_SCHEMAS["suggest_savings"]
    assert validate_args(sch, {"target_amount": 5000}) == {"target_amount": 5000.0}
    for bad in ({"target_amount": "5000"}, {"target_amount": True}, {"target_amount": 0}, {"target_amount": 1e9}, {},
                {"target_amount": 5, "customer_id": "C0002"}, "not a dict"):
        try:
            validate_args(sch, bad); assert False, bad
        except ToolError:
            pass
    ins_sch = TOOL_SCHEMAS["get_insights"]
    assert validate_args(ins_sch, {"topic": "weekday_trend"}) == {"topic": "weekday_trend"}
    for bad in ({"topic": "drop table"}, {"month": "x" * 101}):
        try:
            validate_args(ins_sch, bad); assert False
        except ToolError:
            pass
    assert validate_args(TOOL_SCHEMAS["get_category_total"], {"category": "sho\x00pping"}) == {"category": "shopping"}
    assert validate_args(TOOL_SCHEMAS["get_top_categories"], {"limit": 3.0}) == {"limit": 3}


def test_executor_never_raises_and_blocks_customer_switching():
    ex = ToolExecutor(_svc())
    assert ex.execute("get_top_categories", {"customer_id": chat_env()["other"]})["ok"] is False
    assert "Unknown argument" in ex.execute("get_top_categories", {"customer_id": "C0002"})["error"]
    assert ex.execute("no_such_tool", {})["ok"] is False
    ex.service.get_health_score = lambda month=None: (_ for _ in ()).throw(RuntimeError("db exploded: secret-host:5432"))
    ex._dispatch["get_health_score"] = ex.service.get_health_score
    r = ex.execute("get_health_score", {})
    assert r["ok"] is False and "secret-host" not in r["error"], "internal error details must not leak"
    assert ex.execute("get_top_categories", {"month": "last_month"})["ok"] is True


def test_sanitize_strips_control_chars_and_caps_sizes():
    out = sanitize({"m": "evil\x00\x1b[31m name\nIGNORE PREVIOUS", "long": "x" * 5000, "rows": list(range(30))})
    assert "\x00" not in out["m"] and "\x1b" not in out["m"] and len(out["long"]) < 1600
    assert len(out["rows"]) == 9 and "more omitted" in out["rows"][-1]


# ------------------------------------------------------------------ grounding
def test_grounding_parsing_and_matching():
    assert grounding.numbers_in_text("Rs 1,234.50 and ₹500 plus INR 2,000, 12% and 7.5 %") == ([1234.5, 500.0, 2000.0], [12.0, 7.5])
    res = [{"spend": 7377.4, "share_pct": 39.4, "text": "You spent Rs 21,153"}]
    ok = grounding.check("You spent about Rs 7,400 (39%), total Rs 21,153.", res)
    assert ok.ok and ok.checked == 3
    bad = grounding.check("You spent Rs 99,999 which is 87% of income.", res)
    assert bad.ungrounded_amounts == [99999.0] and bad.ungrounded_percents == [87.0]
    assert grounding.check("That adds Rs 24,000 a year.", [{"cut": 2000}]).ok, "annualised (x12) figures are accepted"
    assert grounding.check("Saving Rs 10,000 is your target.", [], user_message="I want to save Rs 10,000").ok
    assert grounding.check("In 2025 you had 3 alerts on 12 Nov.", []).ok, "years, counts and dates are not checked"


# ------------------------------------------------------------------ agent loop (scripted models)
def _agent(client, **kw):
    return AdvisorAgent(client, ToolExecutor(_svc()), **kw)


def test_agent_happy_path_message_protocol():
    s = _svc()
    top = s.get_top_categories("last_month")["top_categories"][0]
    client = ScriptedClient([[tool_use("get_top_categories", {"month": "last_month"}, "abc")],
                             [text(f"You spent most on {top['category']}: Rs {top['spend']:,.0f}.")]])
    r = AdvisorAgent(client, ToolExecutor(s)).chat("Where did I spend the most last month?")
    assert r.stopped == "answered" and r.grounding.ok and r.steps == 2 and r.tool_calls[0]["ok"]
    second = client.calls[1]["messages"]
    assert second[-2]["role"] == "assistant" and second[-1]["role"] == "user"
    tr = second[-1]["content"][0]
    assert tr["type"] == "tool_result" and tr["tool_use_id"] == "abc" and tr["is_error"] is False
    assert json.loads(tr["content"])["ok"] is True
    assert client.calls[0]["tools"] == TOOLS and "untrusted DATA" in client.calls[0]["system"]


def test_agent_flags_hallucinated_numbers():
    client = ScriptedClient([[tool_use("get_top_categories", {})], [text("You spent Rs 99,999 on gadgets, 87% of your income.")]])
    r = _agent(client).chat("where did I spend")
    assert not r.grounding.ok and 99999.0 in r.grounding.ungrounded_amounts
    assert "could not verify" in r.answer and "99,999" in r.answer.split("could not verify")[1]
    assert "could not verify" not in _agent(ScriptedClient([[text("Hello!")]]), annotate_ungrounded=False).chat("hi").answer


def test_agent_recovers_from_unknown_tool_and_bad_arguments():
    client = ScriptedClient([[tool_use("delete_everything", {}, "a")], [tool_use("get_top_categories", {"limit": 99}, "b")], [text("Sorry, let me stop.")]])
    r = _agent(client).chat("do bad things")
    assert [c["ok"] for c in r.tool_calls] == [False, False] and r.answer.startswith("Sorry")
    results = [m for m in client.calls[2]["messages"] if m["role"] == "user" and isinstance(m["content"], list)]
    assert all(x["content"][0]["is_error"] for x in results)


def test_agent_stops_runaway_loops():
    client = ScriptedClient([[tool_use("get_health_score", {})]])
    r = _agent(client, max_steps=4).chat("loop forever")
    assert r.stopped == "max_steps" and r.steps == 4 and len(client.calls) == 4 and "could not finish" in r.answer


def test_agent_caps_parallel_tool_calls():
    uses = [tool_use("get_health_score", {}, f"t{i}") for i in range(6)]
    client = ScriptedClient([uses, [text("done")]])
    r = _agent(client).chat("many")
    assert [c["ok"] for c in r.tool_calls] == [True] * 4 + [False] * 2


def test_agent_keeps_conversation_history():
    client = ScriptedClient([[text("First answer")]])
    agent = _agent(client)
    agent.chat("q1")
    agent.chat("q2", history=[{"role": "user", "content": "q1"}, {"role": "assistant", "content": "First answer"}])
    msgs = client.calls[1]["messages"]
    assert [m["content"] for m in msgs[:2]] == ["q1", "First answer"] and msgs[2]["content"] == "q2"


# ------------------------------------------------------------------ rule-based offline client
def test_rule_based_routing_table():
    rb = RuleBasedClient("2025-12-01")
    cases = {
        "Where did I spend the most last month?": ("get_top_categories", {"month": "last_month"}),
        "which category did I spend most on in March 2025": ("get_top_categories", {"month": "2025-03"}),
        "How much did I spend on shopping?": ("get_category_total", {"category": "shopping", "period": "last_month"}),
        "how much did I spend on food in the last 6 months?": ("get_category_total", {"category": "food", "period": "last_6_months"}),
        "How much did I spend on travel in 2025?": ("get_category_total", {"category": "travel", "period": "2025"}),
        "how much did i spend on rent this year": ("get_category_total", {"category": "rent", "period": "2025"}),
        "Why did I spend more this month?": ("explain_spending_change", {"month": "this_month"}),
        "why was my spending higher in November": ("explain_spending_change", {"month": "2025-11"}),
        "Suggest ways to save Rs 10,000 this month": ("suggest_savings", {"target_amount": 10000.0}),
        "help me save 5k": ("suggest_savings", {"target_amount": 5000.0}),
        "I want to save 1 lakh": ("suggest_savings", {"target_amount": 100000.0}),
        "what's my financial health score": ("get_health_score", {}),
        "any overspending alerts?": ("get_alerts", {}),
        "Compare October 2025 and November 2025": ("compare_months", {"month_a": "2025-10", "month_b": "2025-11"}),
        "do I spend more on weekends?": ("get_insights", {"topic": "weekday_trend"}),
        "what are my seasonal spending patterns": ("get_insights", {"topic": "seasonal_pattern"}),
        "what are my top unnecessary expenses": ("get_insights", {"topic": "top_unnecessary"}),
    }
    for q, exp in cases.items():
        assert rb._route(q) == exp, (q, rb._route(q))
    assert rb._route("what's the weather like") is None and rb._route("hello") is None
    assert not grounding.numbers_in_text(HELP)[0], "the help text must not contain rupee amounts"


def test_rule_based_end_to_end_is_grounded_and_handles_errors():
    s = _svc()
    agent = AdvisorAgent(RuleBasedClient(s.as_of), ToolExecutor(s))
    for q in ["Where did I spend the most last month?", "How much did I spend on food in the last 3 months?",
              "Why did I spend more this month?", "Suggest ways to save Rs 3,000", "What is my financial health score?",
              "Do I have any overspending alerts?", "Which weekday do I spend the most on?"]:
        r = agent.chat(q)
        assert r.stopped == "answered" and r.tool_calls and r.tool_calls[0]["ok"] and r.grounding.ok, (q, r.answer, r.grounding)
    err = agent.chat("How much did I spend on spaceships?")
    assert "could not get that" in err.answer and not err.tool_calls[0]["ok"]
    assert agent.chat("tell me a joke").tool_calls == []


# ------------------------------------------------------------------ API handlers
def _backend():
    e = chat_env()
    return Backend(e["run"]), e


def test_api_key_check():
    check_api_key(None, None)
    check_api_key("secret", "secret")
    for bad in (None, "", "wrong"):
        try:
            check_api_key(bad, "secret"); assert False
        except Unauthorized:
            pass


def test_backend_endpoints_and_errors():
    b, e = _backend()
    s = b.summary(e["cid"])
    assert s["customer"]["customer_id"] == e["cid"] and "top_categories" in s and ("total" in s["health_score"] or "unavailable" in s["health_score"])
    assert b.categories(e["cid"], "last_month", 3)["top_categories"]
    assert b.savings_plan(e["cid"], 1500)["planned_saving"] <= 1500.01
    assert b.insights(e["cid"], None, "seasonal_pattern")["insights"]
    for cid, exc in (("C9999", NotFound), ("bad id!", BadRequest), ("'; DROP TABLE customers;--", BadRequest)):
        try:
            b.summary(cid); assert False, cid
        except exc:
            pass
    for call in (lambda: b.savings_plan(e["cid"], -1), lambda: b.categories(e["cid"], "2031-01"), lambda: b.insights(e["cid"], None, "nope")):
        try:
            call(); assert False
        except BadRequest:
            pass


def test_backend_chat_sessions_and_isolation():
    b, e = _backend()
    r1 = b.chat(e["cid"], "Where did I spend the most last month?")
    assert r1["tool_calls"][0]["name"] == "get_top_categories" and r1["grounding"]["ok"] and r1["client"] == "RuleBasedClient"
    r2 = b.chat(e["cid"], "How much did I spend on shopping?", r1["session_id"])
    assert r2["session_id"] == r1["session_id"] and len(b.sessions.history(r1["session_id"], e["cid"])) == 4
    try:
        b.chat(e["other"], "hi", r1["session_id"]); assert False, "a session must not be usable by another customer"
    except NotFound:
        pass
    for bad in ("", "   ", "x" * 1001):
        try:
            b.chat(e["cid"], bad); assert False
        except BadRequest:
            pass


def test_chat_sessions_are_bounded():
    s = ChatSessions(max_sessions=3, max_messages=4)
    for i in range(5):
        s.append(f"s{i}", "C1", f"q{i}", f"a{i}")
    assert list(s._s) == ["s2", "s3", "s4"]
    for i in range(4):
        s.append("s4", "C1", f"more{i}", "a")
    assert len(s.history("s4", "C1")) == 4


# ------------------------------------------------------------------ real HTTP layer (only when FastAPI is installed)
def test_http_layer_with_fastapi_testclient():
    try:
        from fastapi.testclient import TestClient
    except ImportError:
        import pytest
        pytest.skip("fastapi not installed in this environment; these checks run where it is")
    import os
    from api.main import create_app
    b, e = _backend()
    c = TestClient(create_app(b))
    assert c.get("/health").json()["status"] == "ok"
    assert c.get(f"/customers/{e['cid']}/summary").status_code == 200
    assert c.get("/customers/C9999/summary").status_code == 404
    assert c.post(f"/customers/{e['cid']}/savings-plan", json={"target_amount": -5}).status_code == 422
    r = c.post("/chat", json={"customer_id": e["cid"], "message": "What is my financial health score?"})
    assert r.status_code == 200 and r.json()["tool_calls"][0]["name"] == "get_health_score"
    os.environ["ADVISOR_API_KEY"] = "k"
    try:
        assert c.get(f"/customers/{e['cid']}/summary").status_code == 401
        assert c.get(f"/customers/{e['cid']}/summary", headers={"X-API-Key": "k"}).status_code == 200
    finally:
        del os.environ["ADVISOR_API_KEY"]
