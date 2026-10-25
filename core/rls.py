"""
Optional PostgreSQL row-level security (RLS): a second wall behind the `WHERE customer_id = :cid` in every query.

How it works
  * sql/rls.sql creates a limited database role (finsight_app), switches RLS on for every table that has a customer_id, and
    makes the analytics views run with the caller's rights (security_invoker), so the policies apply through the views too.
  * Each transaction tells the database which customer it acts for (SET LOCAL app.customer_id) through an engine hook that reads
    CURRENT_CUSTOMER. No customer set = no rows visible (fails closed). The auth code (login, sign-up, erasure) sets
    `app.bypass` instead, because it must look users up before any customer is known.
  * It protects against a query that forgets its WHERE clause or a code path that uses the wrong id. It is NOT protection against
    SQL injection (we only use bound parameters), because an injected statement could also set `app.bypass`.

Turn it on with RLS=on, after applying sql/rls.sql, and connect the API / app with the finsight_app role through APP_DATABASE_URL.
Startup then refuses to run if the connection role could bypass RLS or any view is not security_invoker (misconfigured = loud, not silent).
"""
from __future__ import annotations

import os
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Optional

CURRENT_CUSTOMER: ContextVar[Optional[str]] = ContextVar("finsight_current_customer", default=None)


def enabled(env=os.environ) -> bool:
    return env.get("RLS", "off").strip().lower() in ("on", "1", "true", "yes")


def database_url(default: str, env=os.environ) -> str:
    """APP_DATABASE_URL (the limited role) wins over DATABASE_URL (the owner / superuser used by data loading and jobs)."""
    return env.get("APP_DATABASE_URL") or env.get("DATABASE_URL", default)


@contextmanager
def customer_scope(customer_id: Optional[str]):
    token = CURRENT_CUSTOMER.set(customer_id)
    try:
        yield
    finally:
        CURRENT_CUSTOMER.reset(token)


def prepare_engine(engine):
    """Install the per-transaction customer hook and verify the setup. A no-op when RLS is off."""
    if not enabled():
        return engine
    from sqlalchemy import event

    @event.listens_for(engine, "begin")
    def _scope_transaction(conn):
        conn.exec_driver_sql("SELECT set_config('app.customer_id', %s, true)", (CURRENT_CUSTOMER.get() or "",))

    with engine.connect() as c:
        bypass = c.exec_driver_sql("SELECT rolsuper OR rolbypassrls FROM pg_roles WHERE rolname = current_user").scalar()
        loose = [r[0] for r in c.exec_driver_sql(
            "SELECT c.relname FROM pg_class c JOIN pg_namespace n ON n.oid = c.relnamespace "
            "WHERE n.nspname = 'public' AND c.relkind = 'v' "
            "AND NOT COALESCE(c.reloptions @> ARRAY['security_invoker=true'], false)").fetchall()]
    if bypass:
        raise RuntimeError("RLS=on but this database role is a superuser or has BYPASSRLS, so row-level security would not apply. "
                           "Connect with the finsight_app role (APP_DATABASE_URL); see docs/DEPLOY.md.")
    if loose:
        raise RuntimeError("RLS=on but these views are not security_invoker (re-run sql/rls.sql after loading data): " + ", ".join(loose))
    return engine
