"""
Proves row-level security on a REAL PostgreSQL (SQLite cannot do RLS). Skipped unless both URLs are set:

  RLS_TEST_ADMIN_URL  the owner / superuser (finance)      postgresql+psycopg2://finance:finance@localhost:5432/finance_advisor
  RLS_TEST_APP_URL    the limited role (finsight_app)       postgresql+psycopg2://finsight_app:<password>@localhost:5432/finance_advisor

Apply sql/rls.sql first. The test inserts two throw-away customers (ZRLS_A / ZRLS_B) and deletes them afterwards.
PowerShell:  $env:RLS_TEST_ADMIN_URL="..."; $env:RLS_TEST_APP_URL="..."; python -m pytest tests/test_rls_postgres.py -v
"""
import os

import pytest

ADMIN, APP = os.getenv("RLS_TEST_ADMIN_URL"), os.getenv("RLS_TEST_APP_URL")
pytestmark = pytest.mark.skipif(not (ADMIN and APP), reason="set RLS_TEST_ADMIN_URL and RLS_TEST_APP_URL (see the file docstring)")

if ADMIN and APP:
    from sqlalchemy import create_engine, text

    from core import rls
    from core.auth import SqlAlchemyDb


@pytest.fixture(scope="module")
def engines():
    os.environ["RLS"] = "on"
    admin, app = create_engine(ADMIN), create_engine(APP)
    rls.prepare_engine(app)            # itself fails if the role is a superuser / BYPASSRLS or any view is not security_invoker
    with admin.begin() as c:
        for cid, name in (("ZRLS_A", "Alice RLS"), ("ZRLS_B", "Bob RLS")):
            c.execute(text("INSERT INTO customers (customer_id, name) VALUES (:c, :n) ON CONFLICT DO NOTHING"), {"c": cid, "n": name})
        for i, cid in enumerate(("ZRLS_A", "ZRLS_B")):
            c.execute(text("INSERT INTO transactions (txn_id, customer_id, txn_date, txn_ts, amount, type, raw_description, category) "
                           "VALUES (:i, :c, '2025-10-01', '2025-10-01 00:00:00', 10, 'debit', 'RLS TEST', 'Shopping') ON CONFLICT DO NOTHING"),
                      {"i": 900_000_001 + i, "c": cid})
    yield admin, app
    with admin.begin() as c:
        c.execute(text("DELETE FROM transactions WHERE customer_id LIKE 'ZRLS%'"))
        c.execute(text("DELETE FROM customers WHERE customer_id LIKE 'ZRLS%'"))


def _col(engine, sql):
    with engine.connect() as c:
        return sorted(r[0] for r in c.execute(text(sql)))


def test_a_scoped_customer_sees_only_their_own_rows_and_name(engines):
    _, app = engines
    with rls.customer_scope("ZRLS_A"):
        assert _col(app, "SELECT customer_id FROM transactions WHERE customer_id LIKE 'ZRLS%'") == ["ZRLS_A"]
        assert _col(app, "SELECT name FROM customers WHERE customer_id LIKE 'ZRLS%'") == ["Alice RLS"]


def test_nothing_is_visible_without_a_customer_scope(engines):
    _, app = engines
    assert _col(app, "SELECT customer_id FROM transactions WHERE customer_id LIKE 'ZRLS%'") == []
    assert _col(app, "SELECT name FROM customers WHERE customer_id LIKE 'ZRLS%'") == []


def test_the_scope_does_not_leak_to_the_next_transaction_on_a_pooled_connection(engines):
    _, app = engines
    with rls.customer_scope("ZRLS_A"):
        assert _col(app, "SELECT customer_id FROM transactions WHERE customer_id LIKE 'ZRLS%'") == ["ZRLS_A"]
    for _ in range(5):
        assert _col(app, "SELECT customer_id FROM transactions WHERE customer_id LIKE 'ZRLS%'") == []


def test_asking_for_another_customers_rows_by_id_returns_nothing(engines):
    _, app = engines
    with rls.customer_scope("ZRLS_A"):
        assert _col(app, "SELECT customer_id FROM transactions WHERE customer_id = 'ZRLS_B'") == []
        assert _col(app, "SELECT name FROM customers WHERE customer_id = 'ZRLS_B'") == []


def test_views_are_scoped_too(engines):
    _, app = engines
    with rls.customer_scope("ZRLS_A"):
        assert _col(app, "SELECT DISTINCT customer_id FROM v_expense_txn WHERE customer_id LIKE 'ZRLS%'") == ["ZRLS_A"]


def test_a_customer_cannot_write_or_delete_another_customers_rows(engines):
    admin, app = engines
    with rls.customer_scope("ZRLS_A"):
        with pytest.raises(Exception):
            with app.begin() as c:
                c.execute(text("INSERT INTO transactions (txn_id, customer_id, txn_date, txn_ts, amount, type, raw_description) "
                               "VALUES (900000099, 'ZRLS_B', '2025-10-02', '2025-10-02 00:00:00', 5, 'debit', 'forged')"))
        with app.begin() as c:
            deleted = c.execute(text("DELETE FROM transactions WHERE customer_id = 'ZRLS_B'")).rowcount
        assert deleted == 0
    assert _col(admin, "SELECT customer_id FROM transactions WHERE customer_id = 'ZRLS_B'") == ["ZRLS_B"]


def test_auth_transactions_can_look_across_customers_but_customer_transactions_cannot(engines):
    _, app = engines
    db = SqlAlchemyDb(lambda: app)
    with db.transaction() as q:                                        # login / sign-up code path
        assert sorted(r["customer_id"] for r in q("SELECT customer_id FROM customers WHERE customer_id LIKE 'ZRLS%'")) == ["ZRLS_A", "ZRLS_B"]
    with db.transaction(customer_id="ZRLS_B") as q:                    # statement import / delete-my-data path
        assert [r["customer_id"] for r in q("SELECT customer_id FROM customers WHERE customer_id LIKE 'ZRLS%'")] == ["ZRLS_B"]


def test_the_app_role_cannot_change_the_schema(engines):
    _, app = engines
    with pytest.raises(Exception):
        with app.begin() as c:
            c.execute(text("CREATE TABLE zrls_should_fail (x int)"))
