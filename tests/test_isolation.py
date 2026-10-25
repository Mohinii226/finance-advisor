"""Static guards for 'a customer can never reach another customer's data or names'. They fail the build if someone adds a
query without a customer filter, a chat tool with a customer argument, or an API route with a customer id."""
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
# Queries that are allowed to have no :cid. Each one returns reference data or evaluation counts, never rows about a person.
GLOBAL_QUERIES = {"month_range", "expense_categories", "detector_precision", "detector_recall", "false_positive_breakdown"}


def _queries() -> dict:
    text = (ROOT / "sql" / "analytics_queries.sql").read_text(encoding="utf-8")
    return {b.split("\n", 1)[0].strip(): b for b in re.split(r"(?m)^-- name: ", text)[1:]}


def test_every_named_query_is_scoped_to_a_customer_or_explicitly_global():
    unscoped = {n for n, body in _queries().items() if ":cid" not in body}
    assert unscoped <= GLOBAL_QUERIES, f"queries without a customer filter: {sorted(unscoped - GLOBAL_QUERIES)}"


def test_global_queries_never_read_the_customers_table():
    for name in GLOBAL_QUERIES:
        body = _queries().get(name)
        assert body is None or not re.search(r"\b(FROM|JOIN)\s+customers\b", body, re.I), name


def test_there_is_no_query_that_lists_customers():
    assert "customer_list" not in _queries()
    assert "list_customers" not in (ROOT / "app" / "data_access.py").read_text(encoding="utf-8")


def test_chat_tools_never_take_a_customer_argument():
    tools = pytest.importorskip("llm.tools")
    forbidden = {"customer_id", "cid", "customer", "user", "user_id", "account", "account_id"}
    for name, schema in tools.TOOL_SCHEMAS.items():
        assert not forbidden & set(schema.get("properties", {})), name
        assert schema.get("additionalProperties") is False, f"{name} accepts unknown arguments"


def test_rls_policy_covers_every_table_with_a_customer_id():
    schema = (ROOT / "sql" / "schema.sql").read_text(encoding="utf-8")
    rls = (ROOT / "sql" / "rls.sql").read_text(encoding="utf-8")
    with_customer = set(re.findall(r"CREATE TABLE IF NOT EXISTS (\w+) \((?:(?!\n\);).)*?\bcustomer_id\b", schema, re.S))
    covered = set(re.findall(r"'(\w+)'", rls.split("FOREACH t IN ARRAY ARRAY[")[1].split("]")[0]))
    # users / auth_tokens / web_sessions / rate_events / consents hold login data, not customer data: they are not RLS tables
    login_tables = {"users"}
    assert (with_customer - login_tables) <= covered, sorted((with_customer - login_tables) - covered)
