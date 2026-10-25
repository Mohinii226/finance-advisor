"""HTTP-level auth tests. Skipped automatically when FastAPI / httpx / PyJWT are not installed."""
import os

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")
pytest.importorskip("jwt")
os.environ.setdefault("JWT_SECRET", "t" * 48)            # api.main builds `app` at import time and refuses to start without it

from fastapi.testclient import TestClient              # noqa: E402

from api.main import create_app                         # noqa: E402
from core.auth import TokenManager                      # noqa: E402
from tests.auth_helpers import STRONG, make_auth        # noqa: E402

SECRET = "s" * 48
CONSENT = {"accept_privacy_notice": True, "confirm_adult": True}


class SpyBackend:
    """Records which customer id every call was made for; returns it so tests can assert on responses."""

    def __init__(self):
        self.calls = []

    def _rec(self, name, cid):
        self.calls.append((name, cid))
        return {"customer_id": cid}

    def summary(self, cid, month=None): return self._rec("summary", cid)
    def categories(self, cid, month, limit=10): return self._rec("categories", cid)
    def alerts(self, cid, month): return self._rec("alerts", cid)
    def health_score(self, cid, month): return self._rec("health_score", cid)
    def savings_plan(self, cid, target): return self._rec("savings_plan", cid)
    def insights(self, cid, month, topic): return self._rec("insights", cid)
    def chat(self, cid, message, session_id=None): return self._rec("chat", cid)


def make_client(**auth_kw):
    auth, _ = make_auth(**auth_kw)
    spy = SpyBackend()
    client = TestClient(create_app(backend=spy, auth=auth, tokens=TokenManager(SECRET)))
    client.auth_service = auth                                   # tests read the recorded emails from client.auth.mailer
    return client, spy


def register(client, email, name="Asha"):
    r = client.post("/auth/register", json={"email": email, "password": STRONG, "name": name, **CONSENT})
    assert r.status_code == 201, r.text
    body = r.json()
    return body["user"]["customer_id"], {"Authorization": f"Bearer {body['access_token']}"}


def test_every_data_route_requires_a_token():
    c, spy = make_client()
    for method, path in (("get", "/me"), ("get", "/me/summary"), ("get", "/me/categories"), ("get", "/me/alerts"),
                         ("get", "/me/health-score"), ("get", "/me/insights")):
        assert getattr(c, method)(path).status_code == 401, path
    assert c.post("/me/savings-plan", json={"target_amount": 1000}).status_code == 401
    assert c.post("/chat", json={"message": "hi"}).status_code == 401
    assert spy.calls == []


def test_each_user_only_ever_reaches_their_own_customer():
    c, spy = make_client()
    cid_a, ha = register(c, "a@example.com")
    cid_b, hb = register(c, "b@example.com", "Bob")
    assert cid_a != cid_b
    assert c.get("/me/summary", headers=ha).json()["customer_id"] == cid_a
    assert c.get("/me/alerts", headers=hb).json()["customer_id"] == cid_b
    assert c.post("/chat", json={"message": "hi"}, headers=ha).json()["customer_id"] == cid_a
    assert c.post("/me/savings-plan", json={"target_amount": 5000}, headers=hb).json()["customer_id"] == cid_b
    assert {cid for _, cid in spy.calls} == {cid_a, cid_b}


def test_old_customer_id_routes_are_gone_and_chat_rejects_a_customer_id():
    c, spy = make_client()
    cid, h = register(c, "a@example.com")
    assert c.get("/customers/C0001/summary", headers=h).status_code == 404
    assert c.post("/chat", json={"customer_id": "C0001", "message": "hi"}, headers=h).status_code == 422
    assert spy.calls == []


def test_forged_or_foreign_tokens_are_refused():
    c, spy = make_client()
    cid, h = register(c, "a@example.com")
    from core.auth import User
    foreign = TokenManager("o" * 48).issue(User("u", "a@example.com", "C0001", "Evil"))
    for token in ("garbage", foreign, h["Authorization"].split()[1] + "x"):
        assert c.get("/me/summary", headers={"Authorization": f"Bearer {token}"}).status_code == 401
    assert spy.calls == []


def test_login_failures_look_identical_and_lock_after_five():
    c, _ = make_client()
    register(c, "a@example.com")
    ok = c.post("/auth/login", json={"email": "a@example.com", "password": STRONG})
    assert ok.status_code == 200 and ok.json()["token_type"] == "bearer"
    wrong = c.post("/auth/login", json={"email": "a@example.com", "password": "wrong password!"})
    ghost = c.post("/auth/login", json={"email": "ghost@example.com", "password": "wrong password!"})
    assert wrong.status_code == ghost.status_code == 401 and wrong.json() == ghost.json()
    for _ in range(4):
        c.post("/auth/login", json={"email": "a@example.com", "password": "wrong password!"})
    assert c.post("/auth/login", json={"email": "a@example.com", "password": STRONG}).status_code == 429


def test_duplicate_email_and_closed_signup():
    c, _ = make_client()
    register(c, "a@example.com")
    assert c.post("/auth/register", json={"email": "A@example.com", "password": STRONG, "name": "X", **CONSENT}).status_code == 409
    closed, _ = make_client(allow_signup=False)
    assert closed.post("/auth/register", json={"email": "a@example.com", "password": STRONG, "name": "X", **CONSENT}).status_code == 403


def test_demo_login_only_when_enabled_and_is_chat_limited():
    off, _ = make_client()
    assert off.post("/auth/demo").status_code == 404
    c, spy = make_client(demo_customer_id="C0001")
    r = c.post("/auth/demo")
    assert r.status_code == 200 and r.json()["user"]["is_demo"] and r.json()["user"]["customer_id"] == "C0001"
    h = {"Authorization": f"Bearer {r.json()['access_token']}"}
    codes = [c.post("/chat", json={"message": "hi"}, headers=h).status_code for _ in range(22)]
    assert codes[:20] == [200] * 20 and codes[20:] == [429, 429]    # default CHAT_PER_HOUR_DEMO = 20


def test_sign_up_needs_both_confirmations():
    c, _ = make_client()
    base = {"email": "a@example.com", "password": STRONG, "name": "Asha"}
    assert c.post("/auth/register", json=base).status_code == 422                                   # fields missing
    assert c.post("/auth/register", json={**base, "accept_privacy_notice": True, "confirm_adult": False}).status_code == 400
    assert c.post("/auth/register", json={**base, "accept_privacy_notice": False, "confirm_adult": True}).status_code == 400


def test_email_verification_flow_over_http():
    c, _ = make_client(require_verification=True)
    r = c.post("/auth/register", json={"email": "a@example.com", "password": STRONG, "name": "Asha", **CONSENT})
    assert r.status_code == 201 and r.json()["verification_required"] and "access_token" not in r.json() and "user" not in r.json()
    assert c.post("/auth/login", json={"email": "a@example.com", "password": STRONG}).status_code == 403
    token = c.auth_service.mailer.last_token("verify")
    assert c.post("/auth/verify-email", json={"token": "not-a-real-token-at-all"}).status_code == 400
    assert c.post("/auth/verify-email", json={"token": token}).json() == {"verified": True, "email": "a@example.com"}
    assert c.post("/auth/verify-email", json={"token": token}).status_code == 400                    # single use
    assert c.post("/auth/login", json={"email": "a@example.com", "password": STRONG}).status_code == 200


def test_forgot_password_answers_the_same_for_known_and_unknown_emails():
    c, _ = make_client()
    register(c, "a@example.com")
    known = c.post("/auth/forgot-password", json={"email": "a@example.com"})
    unknown = c.post("/auth/forgot-password", json={"email": "ghost@example.com"})
    assert known.status_code == unknown.status_code == 202 and known.json() == unknown.json()
    resend = c.post("/auth/resend-verification", json={"email": "ghost@example.com"})
    assert resend.status_code == 202
    assert len(c.auth_service.mailer.sent) == 1                                                              # only the real account got mail


def test_password_reset_revokes_old_tokens_over_http():
    c, spy = make_client()
    cid, old = register(c, "a@example.com")
    assert c.get("/me", headers=old).status_code == 200
    c.post("/auth/forgot-password", json={"email": "a@example.com"})
    token = c.auth_service.mailer.last_token("reset")
    assert c.post("/auth/reset-password", json={"token": token, "new_password": "short"}).status_code == 400
    ok = c.post("/auth/reset-password", json={"token": token, "new_password": "a brand new passphrase"})
    assert ok.status_code == 200 and ok.json()["reset"] is True
    assert c.get("/me/summary", headers=old).status_code == 401                                      # token from before the reset
    assert c.post("/auth/login", json={"email": "a@example.com", "password": STRONG}).status_code == 401
    fresh = c.post("/auth/login", json={"email": "a@example.com", "password": "a brand new passphrase"})
    assert fresh.status_code == 200
    assert c.get("/me", headers={"Authorization": f"Bearer {fresh.json()['access_token']}"}).status_code == 200


def test_delete_account_needs_the_password_and_kills_the_token():
    c, _ = make_client()
    cid, h = register(c, "a@example.com")
    assert c.post("/me/delete", json={"password": "wrong password!"}, headers=h).status_code == 401
    assert c.get("/me", headers=h).status_code == 200
    assert c.post("/me/delete", json={"password": STRONG}, headers=h).json() == {"deleted": True}
    assert c.get("/me", headers=h).status_code == 401
    assert c.post("/auth/login", json={"email": "a@example.com", "password": STRONG}).status_code == 401
    assert c.post("/me/delete", json={"password": STRONG}).status_code == 401                        # needs a token too


def test_sign_up_response_is_identical_for_new_and_registered_addresses():
    c, _ = make_client(require_verification=True)
    body = {"password": STRONG, "name": "Asha", **CONSENT}
    new = c.post("/auth/register", json={"email": "a@example.com", **body})
    taken = c.post("/auth/register", json={"email": "A@example.com", **body})
    assert new.status_code == taken.status_code == 201 and new.json() == taken.json()
    assert len(c.auth_service.mailer.sent) == 2                                   # the verification link, then the "you already have an account" notice


def test_no_route_takes_a_customer_id():
    c, _ = make_client()
    paths = [r.path for r in c.app.routes]
    assert not [p for p in paths if "customer" in p.lower() or "{cid}" in p], paths
    schema = c.get("/openapi.json").json()
    for path, ops in schema["paths"].items():
        for op in ops.values():
            for prm in op.get("parameters", []):
                assert prm["name"].lower() not in {"customer_id", "cid", "customer"}, (path, prm)


def test_login_is_throttled_per_ip_as_well_as_per_account():
    c, _ = make_client()
    for i in range(20):
        assert c.post("/auth/login", json={"email": f"v{i}@example.com", "password": "guess guess guess"}).status_code == 401
    assert c.post("/auth/login", json={"email": "w@example.com", "password": "guess guess guess"}).status_code == 429
