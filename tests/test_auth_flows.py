"""Email verification, password reset, web sessions, database-backed rate limiting, consents and erasure."""
import pytest

from core import privacy
from core.auth import (DbWindowLimiter, EmailNotVerified, InvalidCredentials, InvalidInput, Throttled)
from tests.auth_helpers import STRONG, FakeNow, RecordingMailer, make_auth, reg

NEW_PW = "a brand new passphrase"


# ---------------------------------------------------------------- email verification
def test_unverified_user_cannot_log_in_until_the_link_is_used():
    svc, db = make_auth(require_verification=True, app_base_url="https://app.example.com/")
    mail = svc.mailer
    u = reg(svc, "asha@example.com")
    assert db.scalar("SELECT email_verified_at FROM users WHERE user_id = ?", (u.user_id,)) is None
    to, subject, body = mail.sent[-1]
    assert to == "asha@example.com" and "https://app.example.com/?verify=" in body
    with pytest.raises(EmailNotVerified):
        svc.login("asha@example.com", STRONG)
    with pytest.raises(InvalidCredentials):                 # a wrong password must not reveal "unverified"
        svc.login("asha@example.com", "wrong password!")
    assert svc.verify_email(mail.last_token("verify")).email == "asha@example.com"
    assert svc.login("asha@example.com", STRONG).customer_id == u.customer_id


def test_verification_link_is_single_use_purpose_bound_and_expires():
    svc, db = make_auth(require_verification=True)
    reg(svc, "asha@example.com")
    tok = svc.mailer.last_token("verify")
    assert db.scalar("SELECT COUNT(*) FROM auth_tokens") == 1
    assert tok not in str(db.conn.execute("SELECT * FROM auth_tokens").fetchall())     # only a hash is stored
    svc.verify_email(tok)
    for bad in (tok, tok + "x", "", "short", "z" * 43):
        with pytest.raises(InvalidInput):
            svc.verify_email(bad)
    svc2, _ = make_auth(require_verification=True, verify_ttl_s=-5)
    reg(svc2, "b@example.com")
    with pytest.raises(InvalidInput):
        svc2.verify_email(svc2.mailer.last_token("verify"))
    svc3, _ = make_auth(require_verification=True)                                     # a reset link cannot verify an email
    reg(svc3, "c@example.com")
    svc3.verify_email(svc3.mailer.last_token("verify"))
    svc3.request_password_reset("c@example.com")
    with pytest.raises(InvalidInput):
        svc3.verify_email(svc3.mailer.last_token("reset"))


def test_resend_is_silent_for_unknown_and_verified_and_replaces_the_old_link():
    svc, _ = make_auth(require_verification=True)
    reg(svc, "asha@example.com")
    first = svc.mailer.last_token("verify")
    n = len(svc.mailer.sent)
    svc.resend_verification("ghost@example.com")
    svc.resend_verification("not an email")
    assert len(svc.mailer.sent) == n                        # nothing sent, nothing raised
    svc.resend_verification("asha@example.com")
    second = svc.mailer.last_token("verify")
    assert len(svc.mailer.sent) == n + 1 and second != first
    with pytest.raises(InvalidInput):
        svc.verify_email(first)
    svc.verify_email(second)
    n = len(svc.mailer.sent)
    svc.resend_verification("asha@example.com")             # already verified
    assert len(svc.mailer.sent) == n


def test_resend_is_throttled_silently():
    svc, _ = make_auth(require_verification=True)
    reg(svc, "asha@example.com")
    n = len(svc.mailer.sent)
    for _ in range(10):
        svc.resend_verification("asha@example.com")
    assert len(svc.mailer.sent) - n == 3


def test_a_broken_mail_server_does_not_break_sign_up():
    class Broken:
        def send(self, *a):
            raise OSError("smtp down")
    svc, _ = make_auth(require_verification=True, mailer=Broken())
    assert reg(svc, "asha@example.com").email == "asha@example.com"


# ---------------------------------------------------------------- password reset
def test_reset_flow_end_to_end():
    svc, _ = make_auth()
    reg(svc, "asha@example.com")
    svc.request_password_reset(" ASHA@example.com ")
    to, subject, body = svc.mailer.sent[-1]
    assert to == "asha@example.com" and "?reset=" in body
    tok = svc.mailer.last_token("reset")
    svc.reset_password(tok, NEW_PW)
    with pytest.raises(InvalidCredentials):
        svc.login("asha@example.com", STRONG)
    assert svc.login("asha@example.com", NEW_PW).email == "asha@example.com"
    with pytest.raises(InvalidInput):                       # single use
        svc.reset_password(tok, "another passphrase!")


def test_reset_request_reveals_nothing_about_which_emails_exist():
    svc, _ = make_auth()
    reg(svc, "asha@example.com")
    n = len(svc.mailer.sent)
    assert svc.request_password_reset("ghost@example.com") is None
    assert svc.request_password_reset("garbage") is None
    assert len(svc.mailer.sent) == n
    assert svc.request_password_reset("asha@example.com") is None
    assert len(svc.mailer.sent) == n + 1


def test_weak_new_password_does_not_burn_the_link_and_expired_links_fail():
    svc, _ = make_auth()
    reg(svc, "asha@example.com")
    svc.request_password_reset("asha@example.com")
    tok = svc.mailer.last_token("reset")
    with pytest.raises(InvalidInput):
        svc.reset_password(tok, "short")
    svc.reset_password(tok, NEW_PW)                         # the same link still works
    svc2, _ = make_auth(reset_ttl_s=-5)
    reg(svc2, "b@example.com")
    svc2.request_password_reset("b@example.com")
    with pytest.raises(InvalidInput):
        svc2.reset_password(svc2.mailer.last_token("reset"), NEW_PW)


def test_reset_revokes_old_api_tokens_and_web_sessions_and_lockout():
    svc, db = make_auth()
    u = reg(svc, "asha@example.com")
    old_user = svc.login("asha@example.com", STRONG)
    assert svc.ensure_current(old_user).customer_id == u.customer_id
    web = svc.start_web_session(old_user)
    assert svc.user_from_web_session(web) is not None
    for _ in range(5):
        with pytest.raises(InvalidCredentials):
            svc.login("asha@example.com", "wrong password!")
    svc.request_password_reset("asha@example.com")
    svc.reset_password(svc.mailer.last_token("reset"), NEW_PW)
    with pytest.raises(InvalidCredentials):                 # token issued before the reset
        svc.ensure_current(old_user)
    assert svc.user_from_web_session(web) is None
    assert svc.login("asha@example.com", NEW_PW).token_version == old_user.token_version + 1   # lockout cleared too


def test_reset_marks_email_verified_and_is_throttled_and_skips_the_demo_user():
    svc, db = make_auth(require_verification=True, demo_customer_id="C0001")
    reg(svc, "asha@example.com")                            # never clicked the verification link
    svc.request_password_reset("asha@example.com")
    svc.reset_password(svc.mailer.last_token("reset"), NEW_PW)
    assert svc.login("asha@example.com", NEW_PW).email == "asha@example.com"
    n = len(svc.mailer.sent)
    for _ in range(10):
        svc.request_password_reset("asha@example.com")
    assert len(svc.mailer.sent) - n == 2                    # 3 per hour, one already used above
    demo = svc.demo_user()
    svc.request_password_reset(demo.email)
    assert len(svc.mailer.sent) - n == 2


# ---------------------------------------------------------------- web sessions (Streamlit cookie)
def test_web_session_lifecycle():
    svc, db = make_auth()
    u = reg(svc, "asha@example.com")
    raw = svc.start_web_session(u)
    assert raw not in str(db.conn.execute("SELECT * FROM web_sessions").fetchall())     # only a hash is stored
    assert svc.user_from_web_session(raw).customer_id == u.customer_id
    for bad in (None, "", "garbage", raw + "x"):
        assert svc.user_from_web_session(bad) is None
    svc.end_web_session(raw)
    assert svc.user_from_web_session(raw) is None
    expired = make_auth(session_days=-1)[0]
    u2 = reg(expired, "b@example.com")
    assert expired.user_from_web_session(expired.start_web_session(u2)) is None


def test_unverified_user_has_no_web_session_access():
    svc, _ = make_auth(require_verification=True)
    u = reg(svc, "asha@example.com")
    assert svc.user_from_web_session(svc.start_web_session(u)) is None


# ---------------------------------------------------------------- shared (database) rate limiter
def test_db_limiter_blocks_expires_resets_and_hashes_keys():
    _, db = make_auth()
    now = FakeNow()
    lim = DbWindowLimiter(db, "login", 3, 600, now=now)
    for _ in range(3):
        assert not lim.blocked("a@example.com")
        lim.record("a@example.com")
    assert lim.blocked("a@example.com") and not lim.blocked("b@example.com")
    assert "a@example.com" not in str(db.conn.execute("SELECT * FROM rate_events").fetchall())
    now.t = now.t.replace(minute=30)                        # 30 minutes later: outside the 10-minute window
    assert not lim.blocked("a@example.com")
    for _ in range(3):
        lim.record("a@example.com")
    lim.reset("a@example.com")
    assert not lim.blocked("a@example.com")


def test_db_limiter_is_shared_between_workers_and_namespaced():
    _, db = make_auth()
    worker1, worker2 = DbWindowLimiter(db, "login", 2, 600), DbWindowLimiter(db, "login", 2, 600)
    other = DbWindowLimiter(db, "chat", 2, 600)
    worker1.record("k")
    worker2.record("k")
    assert worker1.blocked("k") and worker2.blocked("k") and not other.blocked("k")


def test_login_lockout_uses_the_shared_limiter():
    svc, db = make_auth()
    svc.login_limiter = DbWindowLimiter(db, "login", 5, 900)
    reg(svc, "asha@example.com")
    for _ in range(5):
        with pytest.raises(InvalidCredentials):
            svc.login("asha@example.com", "wrong password!")
    other_worker = DbWindowLimiter(db, "login", 5, 900)
    assert other_worker.blocked("asha@example.com")
    with pytest.raises(Throttled):
        svc.login("asha@example.com", STRONG)


# ---------------------------------------------------------------- consents and erasure
def test_sign_up_requires_both_confirmations_and_records_consents():
    svc, db = make_auth()
    for flags in (dict(accept_privacy_notice=False), dict(confirm_adult=False)):
        with pytest.raises(InvalidInput):
            reg(svc, "asha@example.com", **flags)
    assert db.scalar("SELECT COUNT(*) FROM users") == 0 and db.scalar("SELECT COUNT(*) FROM customers") == 2
    u = reg(svc, "asha@example.com")
    assert privacy.has_consent(db, u.user_id, privacy.PURPOSE_ACCOUNT)
    assert privacy.has_consent(db, u.user_id, privacy.PURPOSE_ADULT)
    assert not privacy.has_consent(db, u.user_id, privacy.PURPOSE_STATEMENTS)
    assert db.scalar("SELECT notice_version FROM consents LIMIT 1") == privacy.NOTICE_VERSION


def test_consent_can_be_given_and_withdrawn():
    svc, db = make_auth()
    u = reg(svc, "asha@example.com")
    privacy.record_consent(db, u.user_id, privacy.PURPOSE_STATEMENTS)
    assert privacy.has_consent(db, u.user_id, privacy.PURPOSE_STATEMENTS)
    privacy.withdraw_consent(db, u.user_id, privacy.PURPOSE_STATEMENTS)
    assert not privacy.has_consent(db, u.user_id, privacy.PURPOSE_STATEMENTS)


def _add_txn(db, cid, txn_id):
    db.conn.execute("INSERT INTO transactions (txn_id, customer_id, txn_date, txn_ts, amount, type, raw_description) "
                    "VALUES (?, ?, '2026-01-01', '2026-01-01 00:00:00', 10, 'debit', 'X')", (txn_id, cid))


def test_delete_account_erases_everything_of_that_user_only():
    svc, db = make_auth()
    a, b = reg(svc, "a@example.com"), reg(svc, "b@example.com", "Bob")
    _add_txn(db, a.customer_id, 1)
    _add_txn(db, b.customer_id, 2)
    svc.start_web_session(a)
    with pytest.raises(InvalidCredentials):
        svc.delete_account(a, "wrong password!")
    assert db.scalar("SELECT COUNT(*) FROM users WHERE user_id = ?", (a.user_id,)) == 1
    svc.delete_account(a, STRONG)
    for table, col, val in (("users", "user_id", a.user_id), ("customers", "customer_id", a.customer_id),
                            ("accounts", "customer_id", a.customer_id), ("transactions", "customer_id", a.customer_id),
                            ("web_sessions", "user_id", a.user_id), ("consents", "user_id", a.user_id)):
        assert db.scalar(f"SELECT COUNT(*) FROM {table} WHERE {col} = ?", (val,)) == 0, table
    assert db.scalar("SELECT COUNT(*) FROM transactions WHERE customer_id = ?", (b.customer_id,)) == 1
    assert svc.login("b@example.com", STRONG).customer_id == b.customer_id


def test_delete_account_never_deletes_demo_or_synthetic_customers():
    svc, db = make_auth(demo_customer_id="C0001")
    with pytest.raises(InvalidInput):
        svc.delete_account(svc.demo_user(), "anything at all")
    linked = svc.admin_create_user("admin-made@example.com", STRONG, "Linked", customer_id="C0002")
    svc.delete_account(linked, STRONG)
    assert db.scalar("SELECT COUNT(*) FROM users WHERE user_id = ?", (linked.user_id,)) == 0
    assert db.scalar("SELECT COUNT(*) FROM customers WHERE customer_id = 'C0002'") == 1   # synthetic data survives


def test_delete_transactions_keeps_the_account():
    svc, db = make_auth()
    a = reg(svc, "a@example.com")
    _add_txn(db, a.customer_id, 1)
    privacy.delete_transactions(db, a.customer_id)
    assert db.scalar("SELECT COUNT(*) FROM transactions") == 0
    assert db.scalar("SELECT COUNT(*) FROM users WHERE user_id = ?", (a.user_id,)) == 1
    with pytest.raises(ValueError):
        privacy.delete_transactions(db, "C0001")


# ---------------------------------------------------------------- sign-up must not reveal which emails exist
def test_signing_up_with_a_taken_email_looks_like_a_normal_sign_up_when_verification_is_on():
    svc, db = make_auth(require_verification=True)
    first = reg(svc, "asha@example.com")
    assert first is not None and first.email == "asha@example.com"
    n_users, n_customers, n_mail = (db.scalar("SELECT COUNT(*) FROM users"), db.scalar("SELECT COUNT(*) FROM customers"), len(svc.mailer.sent))
    again = reg(svc, "ASHA@example.com", "Someone Else")
    assert again is None                                              # no exception, no account object: the caller says the same thing
    assert db.scalar("SELECT COUNT(*) FROM users") == n_users and db.scalar("SELECT COUNT(*) FROM customers") == n_customers
    to, subject, body = svc.mailer.sent[-1]
    assert len(svc.mailer.sent) == n_mail + 1 and to == "asha@example.com" and "already have" in subject.lower()
    assert "?verify=" not in body and "?reset=" not in body           # the notice carries no link an attacker could use


def test_taken_email_notices_are_capped_so_nobody_can_mail_bomb_an_address():
    svc, _ = make_auth(require_verification=True)
    reg(svc, "asha@example.com")
    n = len(svc.mailer.sent)
    for _ in range(10):
        reg(svc, "asha@example.com", "Spammer")
    assert len(svc.mailer.sent) - n == 3


def test_taken_email_still_raises_when_verification_is_off_for_local_development():
    svc, _ = make_auth(require_verification=False)
    reg(svc, "asha@example.com")
    with pytest.raises(Exception) as e:
        reg(svc, "asha@example.com")
    assert type(e.value).__name__ == "EmailTaken"


def test_a_taken_email_does_not_skip_input_validation_or_consent_checks():
    svc, _ = make_auth(require_verification=True)
    reg(svc, "asha@example.com")
    with pytest.raises(InvalidInput):
        reg(svc, "asha@example.com", password="short")
    with pytest.raises(InvalidInput):
        reg(svc, "asha@example.com", confirm_adult=False)


# ---------------------------------------------------------------- per-IP throttling
def test_failed_logins_are_throttled_per_ip_across_different_accounts():
    svc, _ = make_auth()
    for i in range(20):
        with pytest.raises(InvalidCredentials):
            svc.login(f"victim{i}@example.com", "guess guess guess", ip="203.0.113.7")
    with pytest.raises(Throttled):
        svc.login("someone-else@example.com", "guess guess guess", ip="203.0.113.7")
    with pytest.raises(InvalidCredentials):                           # another address is unaffected
        svc.login("someone-else@example.com", "guess guess guess", ip="198.51.100.9")


def test_a_throttled_ip_cannot_log_in_even_with_the_right_password_until_the_window_passes():
    from core.auth import WindowLimiter
    from tests.auth_helpers import FakeClock
    clock = FakeClock()
    svc, _ = make_auth(ip_fail_limiter=WindowLimiter(3, 900, clock=clock))
    reg(svc, "asha@example.com")
    for _ in range(3):
        with pytest.raises(InvalidCredentials):
            svc.login("nobody@example.com", "guess guess guess", ip="203.0.113.7")
    with pytest.raises(Throttled):
        svc.login("asha@example.com", STRONG, ip="203.0.113.7")
    clock.t += 901
    assert svc.login("asha@example.com", STRONG, ip="203.0.113.7").email == "asha@example.com"


def test_successful_logins_do_not_count_against_the_ip():
    svc, _ = make_auth()
    reg(svc, "asha@example.com")
    for _ in range(30):
        svc.login("asha@example.com", STRONG, ip="203.0.113.7")


def test_email_sending_requests_are_capped_per_ip():
    svc, _ = make_auth(require_verification=True)
    for i in range(10):
        reg(svc, f"user{i}@example.com", ip="203.0.113.7")
    with pytest.raises(Throttled):
        reg(svc, "user11@example.com", ip="203.0.113.7")
    with pytest.raises(Throttled):
        svc.request_password_reset("user0@example.com", ip="203.0.113.7")
    with pytest.raises(Throttled):
        svc.resend_verification("user0@example.com", ip="203.0.113.7")
    reg(svc, "other@example.com", ip="198.51.100.9")                  # a different address is fine
    reg(svc, "no-ip@example.com")                                     # no ip given: no ip limiting
