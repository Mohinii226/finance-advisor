"""Tests for core/auth.py (no FastAPI, Streamlit or PostgreSQL needed)."""
import base64
import json

import pytest

from core.auth import (EmailTaken, InvalidCredentials, InvalidInput, NotAvailable, SignupClosed, Throttled, TokenManager,
                       User, WindowLimiter, check_password_rules, hash_password, normalize_email, verify_password)
from tests.auth_helpers import STRONG, FakeClock, make_auth, reg

SECRET = "x" * 48


# ---------------------------------------------------------------- passwords
def test_hash_roundtrip_and_salting():
    h1, h2 = hash_password(STRONG), hash_password(STRONG)
    assert h1 != h2                                   # random salt
    assert verify_password(STRONG, h1) and verify_password(STRONG, h2)
    assert not verify_password(STRONG + "x", h1)
    assert STRONG not in h1


@pytest.mark.parametrize("bad", ["", "plain", "scrypt$1$2$3", "scrypt$a$b$c$d$e", "bcrypt$1$2$3$4$5", None])
def test_malformed_hash_never_verifies(bad):
    assert verify_password(STRONG, bad) is False


def test_input_rules():
    assert normalize_email("  Asha.Rao@Example.COM ") == "asha.rao@example.com"
    for bad in ("", "no-at-sign", "a@b", "a b@c.com", "x" * 260 + "@a.com"):
        with pytest.raises(InvalidInput):
            normalize_email(bad)
    with pytest.raises(InvalidInput):
        check_password_rules("short")
    with pytest.raises(InvalidInput):
        check_password_rules("x" * 129)
    check_password_rules(STRONG)


# ---------------------------------------------------------------- limiter
def test_limiter_blocks_expires_and_resets():
    clock = FakeClock()
    lim = WindowLimiter(max_events=3, window_s=100, clock=clock)
    for _ in range(3):
        assert not lim.blocked("a")
        lim.record("a")
    assert lim.blocked("a") and not lim.blocked("b")
    clock.t += 101
    assert not lim.blocked("a")                       # window passed
    for _ in range(3):
        lim.record("a")
    lim.reset("a")
    assert not lim.blocked("a")


def test_limiter_memory_is_bounded():
    lim = WindowLimiter(max_events=1, window_s=100, max_keys=50)
    for i in range(500):
        lim.record(f"k{i}")
    assert len(lim._events) <= 50


# ---------------------------------------------------------------- tokens
def test_token_roundtrip_carries_customer_id():
    tm = TokenManager(SECRET)
    u = User("u1", "a@b.co", "U1234ABCD", "Asha", False)
    assert tm.verify(tm.issue(u)) == u


def test_token_rejects_tampering_wrong_secret_expiry_and_alg_none():
    tm = TokenManager(SECRET)
    u = User("u1", "a@b.co", "U1234ABCD", "Asha")
    tok = tm.issue(u)
    head, body, sig = tok.split(".")
    # swap in another customer id without re-signing
    payload = json.loads(base64.urlsafe_b64decode(body + "=" * (-len(body) % 4)))
    payload["cid"] = "C0001"
    forged = base64.urlsafe_b64encode(json.dumps(payload).encode()).decode().rstrip("=")
    for bad in (f"{head}.{forged}.{sig}", tok + "x", "", "garbage", TokenManager("y" * 48).issue(u)):
        with pytest.raises(InvalidCredentials):
            tm.verify(bad)
    with pytest.raises(InvalidCredentials):
        TokenManager(SECRET, ttl_seconds=-60).verify(TokenManager(SECRET, ttl_seconds=-60).issue(u))
    none_head = base64.urlsafe_b64encode(b'{"alg":"none","typ":"JWT"}').decode().rstrip("=")
    unsigned = f"{none_head}.{body}."
    with pytest.raises(InvalidCredentials):
        tm.verify(unsigned)


def test_short_secret_refused():
    for bad in ("", "short", "x" * 31):
        with pytest.raises(RuntimeError):
            TokenManager(bad)


# ---------------------------------------------------------------- register / login
def test_register_creates_isolated_customer_and_account():
    svc, db = make_auth()
    a = reg(svc, "Asha@Example.com", "Asha", 50000)
    b = reg(svc, "bob@example.com", "Bob")
    assert a.customer_id != b.customer_id and a.customer_id.startswith("U") and len(a.customer_id) <= 10
    assert a.email == "asha@example.com" and not a.is_demo
    assert db.scalar("SELECT COUNT(*) FROM accounts WHERE customer_id = ?", (a.customer_id,)) == 1
    assert db.scalar("SELECT monthly_income FROM customers WHERE customer_id = ?", (a.customer_id,)) == 50000
    assert STRONG not in db.scalar("SELECT password_hash FROM users WHERE user_id = ?", (a.user_id,))


def test_duplicate_email_rejected_case_insensitively():
    svc, _ = make_auth()
    reg(svc, "asha@example.com", "Asha")
    with pytest.raises(EmailTaken):
        reg(svc, "ASHA@example.com", "Other")


@pytest.mark.parametrize("kwargs", [dict(name=""), dict(name="x" * 81), dict(password="short"),
                                    dict(monthly_income=-1), dict(monthly_income=float("nan")), dict(monthly_income="abc"),
                                    dict(email="nope")])
def test_register_validates_input(kwargs):
    svc, db = make_auth()
    args = dict(email="a@example.com", password=STRONG, name="Asha", monthly_income=0,
                accept_privacy_notice=True, confirm_adult=True)
    args.update(kwargs)
    with pytest.raises(InvalidInput):
        svc.register(**args)
    assert db.scalar("SELECT COUNT(*) FROM users") == 0


def test_signup_can_be_closed():
    svc, _ = make_auth(allow_signup=False)
    with pytest.raises(SignupClosed):
        reg(svc, "a@example.com", "Asha")


def test_login_success_and_identical_failures():
    svc, db = make_auth()
    u = reg(svc, "asha@example.com", "Asha")
    assert svc.login(" ASHA@example.com ", STRONG).customer_id == u.customer_id
    assert db.scalar("SELECT last_login_at FROM users WHERE user_id = ?", (u.user_id,)) is not None
    messages = set()
    for email, pw in (("asha@example.com", "wrong password!"), ("ghost@example.com", STRONG), ("not-an-email", STRONG),
                      ("asha@example.com", ""), ("asha@example.com", "x" * 500)):
        with pytest.raises(InvalidCredentials) as e:
            svc.login(email, pw)
        messages.add(str(e.value))
    assert len(messages) == 1                         # no hint about which part was wrong


def test_lockout_after_repeated_failures_even_with_correct_password():
    clock = FakeClock()
    svc, _ = make_auth(login_limiter=WindowLimiter(5, 900, clock=clock))
    reg(svc, "asha@example.com", "Asha")
    for _ in range(5):
        with pytest.raises(InvalidCredentials):
            svc.login("asha@example.com", "wrong password!")
    with pytest.raises(Throttled):
        svc.login("asha@example.com", STRONG)
    clock.t += 901
    assert svc.login("asha@example.com", STRONG).email == "asha@example.com"


def test_success_resets_failure_count():
    svc, _ = make_auth()
    reg(svc, "asha@example.com", "Asha")
    for _ in range(4):
        with pytest.raises(InvalidCredentials):
            svc.login("asha@example.com", "wrong password!")
    svc.login("asha@example.com", STRONG)
    for _ in range(4):                                # would be locked if the earlier failures still counted
        with pytest.raises(InvalidCredentials):
            svc.login("asha@example.com", "wrong password!")


# ---------------------------------------------------------------- demo + admin paths
def test_demo_disabled_by_default():
    svc, _ = make_auth()
    with pytest.raises(NotAvailable):
        svc.demo_user()


def test_demo_user_is_stable_shared_and_has_no_usable_password():
    svc, db = make_auth(demo_customer_id="C0001")
    d1, d2 = svc.demo_user(), svc.demo_user()
    assert d1 == d2 and d1.customer_id == "C0001" and d1.is_demo
    assert db.scalar("SELECT COUNT(*) FROM users") == 1
    for pw in (STRONG, "", "demo", "password123"):
        with pytest.raises(InvalidCredentials):
            svc.login(d1.email, pw)


def test_demo_with_unknown_customer_fails_cleanly():
    svc, _ = make_auth(demo_customer_id="C9999")
    with pytest.raises(NotAvailable):
        svc.demo_user()


def test_admin_can_link_login_to_existing_customer_once():
    svc, _ = make_auth(allow_signup=False)            # admin path works even when public sign-up is closed
    u = svc.admin_create_user("asha@example.com", STRONG, "Asha", customer_id="C0001")
    assert u.customer_id == "C0001"
    with pytest.raises(InvalidInput):
        svc.admin_create_user("other@example.com", STRONG, "Other", customer_id="C0001")
    with pytest.raises(NotAvailable):
        svc.admin_create_user("third@example.com", STRONG, "Third", customer_id="C4242")


def test_public_register_cannot_choose_a_customer():
    svc, _ = make_auth()
    with pytest.raises(TypeError):
        svc.register("a@example.com", STRONG, "Asha", 0, accept_privacy_notice=True, confirm_adult=True, customer_id="C0001")
