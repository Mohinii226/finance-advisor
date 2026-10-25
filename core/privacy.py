"""
Consent records and data erasure (supports India's DPDP Act duties: notice + consent, withdrawal, erasure).

* Every consent is stored with the version of the privacy notice it was given against, so you can show what a user agreed to.
* Withdrawing a consent is as easy as giving it (one call / one button).
* Erasure deletes the login, its tokens and sessions, and (for accounts created by sign-up, whose customer id starts with 'U')
  all the customer's transactions and derived rows. Synthetic customers linked by an admin (C0001 ...) only lose the login.
Not legal advice: have the notice text (docs/privacy_notice_template.md) and your retention rules reviewed before you invite real users.
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Optional

NOTICE_VERSION = "2026-10-v1"          # bump whenever docs/privacy_notice_template.md changes materially
PURPOSE_ACCOUNT = "account"            # name, email, income: to run your account
PURPOSE_ADULT = "age_18_plus"          # DPDP treats under-18s as children (verifiable parental consent needed)
PURPOSE_STATEMENTS = "statement_processing"   # transactions you upload: to analyse your spending

_CUSTOMER_TABLES = ("transactions", "sms_raw", "alerts", "forecasts", "health_scores", "injected_events", "accounts")
_USER_TABLES = ("auth_tokens", "web_sessions", "consents")


def _now() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


def insert_consent(q, user_id: str, purpose: str, now: Optional[datetime] = None) -> None:
    q("INSERT INTO consents (consent_id, user_id, purpose, notice_version, given_at) VALUES (:id, :u, :p, :v, :t)",
      {"id": uuid.uuid4().hex, "u": user_id, "p": purpose, "v": NOTICE_VERSION, "t": now or _now()})


def record_consent(db, user_id: str, purpose: str) -> None:
    with db.transaction() as q:
        insert_consent(q, user_id, purpose)


def has_consent(db, user_id: str, purpose: str) -> bool:
    with db.transaction() as q:
        return bool(q("SELECT 1 AS ok FROM consents WHERE user_id = :u AND purpose = :p AND withdrawn_at IS NULL",
                      {"u": user_id, "p": purpose}))


def withdraw_consent(db, user_id: str, purpose: str) -> None:
    with db.transaction() as q:
        q("UPDATE consents SET withdrawn_at = :t WHERE user_id = :u AND purpose = :p AND withdrawn_at IS NULL",
          {"t": _now(), "u": user_id, "p": purpose})


def owns_customer(customer_id: str) -> bool:
    """Sign-up customers are 'U' + hex (core.auth.new_customer_id); synthetic ones are C0001 ... and are never erased here."""
    return customer_id.startswith("U")


def delete_transactions(db, customer_id: str) -> None:
    """Remove everything derived from the customer's uploaded statements (keeps the account)."""
    if not owns_customer(customer_id):
        raise ValueError("Refusing to delete data of a non-sign-up customer.")
    with db.transaction(customer_id=customer_id) as q:
        for t in ("transactions", "sms_raw", "alerts", "forecasts", "health_scores"):
            q(f"DELETE FROM {t} WHERE customer_id = :c", {"c": customer_id})


def erase_user(db, user_id: str, customer_id: str) -> None:
    with db.transaction() as q:
        for t in _USER_TABLES:
            q(f"DELETE FROM {t} WHERE user_id = :u", {"u": user_id})
        q("DELETE FROM users WHERE user_id = :u", {"u": user_id})
        if owns_customer(customer_id):
            for t in _CUSTOMER_TABLES:
                q(f"DELETE FROM {t} WHERE customer_id = :c", {"c": customer_id})
            q("DELETE FROM customers WHERE customer_id = :c", {"c": customer_id})
