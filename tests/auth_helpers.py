"""Shared helpers for the auth tests: an in-memory SQLite stand-in for PostgreSQL behind the same Db interface."""
from __future__ import annotations

import sqlite3
import threading
from contextlib import contextmanager
from datetime import date, datetime

from core.auth import AuthService, DuplicateKey, UserStore

DDL = """
CREATE TABLE customers (
    customer_id VARCHAR(10) PRIMARY KEY, name TEXT NOT NULL, monthly_income NUMERIC, city TEXT, occupation TEXT);
CREATE TABLE accounts (
    account_id VARCHAR(10) PRIMARY KEY, customer_id VARCHAR(10) NOT NULL REFERENCES customers(customer_id),
    bank TEXT, opening_balance NUMERIC, opening_reserve NUMERIC, emi_amount NUMERIC, sip_amount NUMERIC);
CREATE TABLE users (
    user_id TEXT PRIMARY KEY,
    email TEXT NOT NULL UNIQUE CHECK (email = lower(email)),
    password_hash TEXT NOT NULL,
    customer_id VARCHAR(10) NOT NULL UNIQUE REFERENCES customers(customer_id),
    is_demo BOOLEAN NOT NULL DEFAULT 0,
    created_at TIMESTAMP NOT NULL,
    last_login_at TIMESTAMP,
    email_verified_at TIMESTAMP,
    token_version INTEGER NOT NULL DEFAULT 0);
CREATE TABLE auth_tokens (
    token_hash TEXT PRIMARY KEY, user_id TEXT NOT NULL REFERENCES users(user_id),
    purpose TEXT NOT NULL CHECK (purpose IN ('verify_email', 'reset_password')),
    expires_at TIMESTAMP NOT NULL, created_at TIMESTAMP NOT NULL);
CREATE TABLE web_sessions (
    session_hash TEXT PRIMARY KEY, user_id TEXT NOT NULL REFERENCES users(user_id),
    created_at TIMESTAMP NOT NULL, expires_at TIMESTAMP NOT NULL);
CREATE TABLE rate_events (rate_key TEXT NOT NULL, ts TIMESTAMP NOT NULL);
CREATE TABLE consents (
    consent_id TEXT PRIMARY KEY, user_id TEXT NOT NULL REFERENCES users(user_id), purpose TEXT NOT NULL,
    notice_version TEXT NOT NULL, given_at TIMESTAMP NOT NULL, withdrawn_at TIMESTAMP);
CREATE TABLE transactions (
    txn_id BIGINT PRIMARY KEY, customer_id VARCHAR(10) NOT NULL REFERENCES customers(customer_id),
    txn_date DATE NOT NULL, txn_ts TIMESTAMP NOT NULL, amount NUMERIC NOT NULL CHECK (amount > 0),
    type TEXT NOT NULL CHECK (type IN ('debit','credit')), raw_description TEXT NOT NULL, merchant TEXT,
    category TEXT, true_category TEXT, predicted_category TEXT, confidence REAL, channel TEXT, balance_after NUMERIC);
CREATE TABLE sms_raw (sms_id INTEGER PRIMARY KEY, customer_id VARCHAR(10));
CREATE TABLE alerts (alert_id INTEGER PRIMARY KEY, customer_id VARCHAR(10));
CREATE TABLE forecasts (forecast_id INTEGER PRIMARY KEY, customer_id VARCHAR(10));
CREATE TABLE health_scores (customer_id VARCHAR(10), score_month DATE);
CREATE TABLE injected_events (event_id INTEGER PRIMARY KEY, customer_id VARCHAR(10));
INSERT INTO customers (customer_id, name, monthly_income, city, occupation) VALUES
    ('C0001', 'Asha Rao', 60000, 'Pune', 'Engineer'), ('C0002', 'Vikram Shah', 45000, 'Mumbai', 'Analyst');
"""


def _adapt(v):
    if isinstance(v, datetime):
        return v.isoformat(sep=" ")
    if isinstance(v, date):
        return v.isoformat()
    return v


class SqliteDb:
    dialect = "sqlite"

    def __init__(self):
        self.conn = sqlite3.connect(":memory:", check_same_thread=False, isolation_level=None)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA foreign_keys = ON")
        self.conn.executescript(DDL)
        self._lock = threading.RLock()

    @contextmanager
    def transaction(self, customer_id=None):
        with self._lock:
            self.conn.execute("BEGIN")

            def execute(sql, params=None):
                try:
                    if isinstance(params, list):
                        self.conn.executemany(sql, [{k: _adapt(v) for k, v in p.items()} for p in params])
                        return []
                    cur = self.conn.execute(sql, {k: _adapt(v) for k, v in (params or {}).items()})
                except sqlite3.IntegrityError as e:
                    raise DuplicateKey(str(e)) from e
                return [dict(r) for r in cur.fetchall()] if cur.description else []

            try:
                yield execute
            except BaseException:
                self.conn.execute("ROLLBACK")
                raise
            else:
                self.conn.execute("COMMIT")

    def scalar(self, sql, params=()):
        return self.conn.execute(sql, params).fetchone()[0]


class RecordingMailer:
    """Collects outgoing mail instead of sending it."""

    def __init__(self):
        self.sent = []                                   # (to, subject, body)

    def send(self, to, subject, body):
        self.sent.append((to, subject, body))

    def last_token(self, kind):
        """The token in the most recent link of the form ...?verify=TOKEN or ...?reset=TOKEN."""
        import re
        for to, subject, body in reversed(self.sent):
            m = re.search(rf"[?&]{kind}=([A-Za-z0-9_-]+)", body)
            if m:
                return m.group(1)
        return None


class FakeClock:
    def __init__(self):
        self.t = 1000.0

    def __call__(self):
        return self.t


class FakeNow:
    """Controllable datetime clock for DbWindowLimiter."""

    def __init__(self):
        self.t = datetime(2026, 10, 23, 12, 0, 0)

    def __call__(self):
        return self.t


def make_auth(**kw):
    kw.setdefault("require_verification", False)
    kw.setdefault("mailer", RecordingMailer())
    db = SqliteDb()
    return AuthService(UserStore(db), **kw), db


def reg(svc, email, name="Asha", income=0, password=None, **flags):
    """Register with the consent boxes ticked (the normal case)."""
    flags.setdefault("accept_privacy_notice", True)
    flags.setdefault("confirm_adult", True)
    return svc.register(email, password or STRONG, name, income, **flags)


STRONG = "correct horse battery"
