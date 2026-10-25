"""
Authentication for FinSight: passwords, signed tokens, email verification, password reset, server-side web sessions,
shared (database-backed) rate limiting, and the register / login / demo flows.

Framework-free on purpose (no FastAPI or Streamlit imports), so the API and the Streamlit app share exactly the same
rules and everything here can be unit-tested without either of them.

THE RULE THIS MODULE EXISTS TO ENFORCE
  The customer a request may see comes from the verified login (a signed token, or a server-side web session),
  never from a URL, a request body or a chat message.

Design notes
  * Passwords: scrypt (stdlib), per-user salt, parameters stored in the hash string so they can be raised later.
    Unknown emails still cost one hash, so response time does not reveal which emails exist.
  * API tokens: HS256 JWT (PyJWT, JWT_SECRET >= 32 chars). A token carries the user's `token_version`; a password reset bumps
    it, so every older token stops working (the API checks this on each request).
  * Streamlit: a random session id in a cookie, stored hashed in `web_sessions` -> logins survive a browser refresh and can
    be revoked (password reset or logout deletes the rows).
  * Email verification / reset links: random 256-bit tokens, only their SHA-256 is stored, single use (deleted when used),
    24 h (verify) / 30 min (reset) lifetime. "Forgot password" and "resend" answer identically whether or not the email exists.
  * Rate limiting: sliding windows stored in PostgreSQL (`rate_events`, keys hashed), so limits hold across workers/containers.
  * Consents (core/privacy.py) are recorded at sign-up; the user can delete their account and data.

Command line (link a login to an existing synthetic customer, e.g. for a demo):
    python -m core.auth create-user --email you@example.com --name "Your Name" --customer C0001
"""
from __future__ import annotations

import base64
import hashlib
import hmac
import logging
import os
import random
import re
import secrets
import threading
import time
import uuid
from collections import OrderedDict, deque
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Callable, Iterator, Optional, Sequence

from core import privacy, rls

log = logging.getLogger("finsight.auth")
DEFAULT_DB_URL = "postgresql+psycopg2://finance:finance@localhost:5432/finance_advisor"
GENERIC_LOGIN_ERROR = "Incorrect email or password."


# ------------------------------------------------------------------------------------------------ errors
class AuthError(Exception):
    """Base class; `status` is the HTTP status the API maps it to. Messages are safe to show to the user."""
    status = 400


class InvalidInput(AuthError):
    status = 400


class InvalidCredentials(AuthError):
    status = 401


class EmailNotVerified(AuthError):
    status = 403


class ConsentRequired(AuthError):
    status = 403


class SignupClosed(AuthError):
    status = 403


class NotAvailable(AuthError):
    status = 404


class EmailTaken(AuthError):
    status = 409


class Throttled(AuthError):
    status = 429


class DuplicateKey(Exception):
    """Raised by a Db adapter when a unique / primary-key constraint is violated."""


def _utcnow() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


def sha256_hex(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


# ------------------------------------------------------------------------------------------------ passwords
_SCRYPT_N, _SCRYPT_R, _SCRYPT_P = 2**15, 8, 3        # ~32 MiB and ~0.25 s per hash; one of OWASP's equivalent scrypt settings
_MAXMEM = 256 * 1024 * 1024


def _scrypt(password: str, salt: bytes, n: int, r: int, p: int) -> bytes:
    return hashlib.scrypt(password.encode("utf-8"), salt=salt, n=n, r=r, p=p, maxmem=_MAXMEM, dklen=32)


def hash_password(password: str) -> str:
    salt = secrets.token_bytes(16)
    dk = _scrypt(password, salt, _SCRYPT_N, _SCRYPT_R, _SCRYPT_P)
    b64 = lambda b: base64.b64encode(b).decode("ascii")
    return f"scrypt${_SCRYPT_N}${_SCRYPT_R}${_SCRYPT_P}${b64(salt)}${b64(dk)}"


def verify_password(password: str, stored: str) -> bool:
    """Constant-time check. Any malformed hash simply fails."""
    try:
        scheme, n, r, p, salt, dk = stored.split("$")
        if scheme != "scrypt":
            return False
        actual = _scrypt(password, base64.b64decode(salt), int(n), int(r), int(p))
        expected = base64.b64decode(dk)
    except (ValueError, TypeError, AttributeError):        # binascii.Error is a ValueError
        return False
    return hmac.compare_digest(actual, expected)


_dummy: Optional[str] = None


def _dummy_hash() -> str:
    """A throwaway hash verified when the email is unknown, so unknown and known emails take the same time."""
    global _dummy
    if _dummy is None:
        _dummy = hash_password(secrets.token_urlsafe(16))
    return _dummy


# ------------------------------------------------------------------------------------------------ validation
_EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
MIN_PASSWORD, MAX_PASSWORD = 10, 128


def normalize_email(email: str) -> str:
    e = (email or "").strip().lower()
    if len(e) > 254 or not _EMAIL_RE.match(e):
        raise InvalidInput("Enter a valid email address.")
    return e


def check_password_rules(password: str) -> None:
    if not isinstance(password, str) or not (MIN_PASSWORD <= len(password) <= MAX_PASSWORD):
        raise InvalidInput(f"Choose a password between {MIN_PASSWORD} and {MAX_PASSWORD} characters.")


def _clean_name(name: str) -> str:
    name = (name or "").strip()
    if not 1 <= len(name) <= 80:
        raise InvalidInput("Enter your name (up to 80 characters).")
    return name


def _clean_income(value) -> float:
    try:
        v = float(value or 0)
    except (TypeError, ValueError):
        raise InvalidInput("Monthly income must be a number.") from None
    if not (0 <= v <= 10_000_000):          # also rejects NaN
        raise InvalidInput("Monthly income must be between 0 and 10,000,000.")
    return v


# ------------------------------------------------------------------------------------------------ rate limiting
class WindowLimiter:
    """In-memory sliding window (per process). Used in tests and as a fallback; production uses DbWindowLimiter."""

    def __init__(self, max_events: int, window_s: float, clock: Callable[[], float] = time.monotonic, max_keys: int = 10_000):
        self.max_events, self.window_s, self.clock, self.max_keys = max_events, window_s, clock, max_keys
        self._events: "OrderedDict[str, deque]" = OrderedDict()
        self._lock = threading.Lock()

    def _live(self, key: str, now: float) -> Optional[deque]:
        dq = self._events.get(key)
        while dq and now - dq[0] > self.window_s:
            dq.popleft()
        if dq is not None and not dq:
            del self._events[key]
            return None
        return dq

    def blocked(self, key: str) -> bool:
        with self._lock:
            dq = self._live(key, self.clock())
            return dq is not None and len(dq) >= self.max_events

    def record(self, key: str) -> None:
        with self._lock:
            now = self.clock()
            self._live(key, now)
            dq = self._events.setdefault(key, deque(maxlen=self.max_events))
            dq.append(now)
            self._events.move_to_end(key)
            while len(self._events) > self.max_keys:
                self._events.popitem(last=False)

    def reset(self, key: str) -> None:
        with self._lock:
            self._events.pop(key, None)


class DbWindowLimiter:
    """Same interface as WindowLimiter, stored in the `rate_events` table: shared by all workers and surviving restarts.
    Keys are SHA-256 hashed with the limiter name, so no emails or user ids sit in the table. Window must be <= 1 day."""

    def __init__(self, db, name: str, max_events: int, window_s: int, now: Callable[[], datetime] = _utcnow):
        if window_s > 86400:
            raise ValueError("window_s must be at most one day")
        self.db, self.name, self.max_events, self.window_s, self._now = db, name, max_events, window_s, now

    def _key(self, key: str) -> str:
        return sha256_hex(f"{self.name}:{key}")

    def blocked(self, key: str) -> bool:
        cutoff = self._now() - timedelta(seconds=self.window_s)
        with self.db.transaction() as q:
            rows = q("SELECT COUNT(*) AS n FROM rate_events WHERE rate_key = :k AND ts > :cutoff",
                     {"k": self._key(key), "cutoff": cutoff})
        return int(rows[0]["n"]) >= self.max_events

    def record(self, key: str) -> None:
        now = self._now()
        with self.db.transaction() as q:
            q("INSERT INTO rate_events (rate_key, ts) VALUES (:k, :now)", {"k": self._key(key), "now": now})
            if random.random() < 0.02:                         # opportunistic clean-up of old rows
                q("DELETE FROM rate_events WHERE ts < :old", {"old": now - timedelta(days=2)})

    def reset(self, key: str) -> None:
        with self.db.transaction() as q:
            q("DELETE FROM rate_events WHERE rate_key = :k", {"k": self._key(key)})


# ------------------------------------------------------------------------------------------------ users and tokens
@dataclass(frozen=True)
class User:
    user_id: str
    email: str
    customer_id: str
    name: str
    is_demo: bool = False
    token_version: int = 0

    def public(self) -> dict:
        return {"user_id": self.user_id, "email": self.email, "name": self.name,
                "customer_id": self.customer_id, "is_demo": self.is_demo}


class TokenManager:
    """Issues and verifies short-lived HS256 JWTs. The customer id lives inside the signed token (claim `cid`)."""
    ALGORITHM = "HS256"

    def __init__(self, secret: str, ttl_seconds: int = 3600, issuer: str = "finsight"):
        if not secret or len(secret) < 32:
            raise RuntimeError("JWT_SECRET must be set to a random string of at least 32 characters. Generate one with: "
                               "python -c \"import secrets; print(secrets.token_urlsafe(48))\"")
        import jwt                                   # PyJWT; imported here so the Streamlit app does not need it
        self._jwt, self._secret, self.ttl_seconds, self.issuer = jwt, secret, int(ttl_seconds), issuer

    @classmethod
    def from_env(cls, env=os.environ) -> "TokenManager":
        return cls(env.get("JWT_SECRET", ""), ttl_seconds=int(env.get("JWT_TTL_MINUTES", "60")) * 60)

    def issue(self, user: User) -> str:
        now = int(time.time())
        payload = {"iss": self.issuer, "sub": user.user_id, "cid": user.customer_id, "email": user.email,
                   "name": user.name, "demo": user.is_demo, "ver": user.token_version, "iat": now, "exp": now + self.ttl_seconds}
        return self._jwt.encode(payload, self._secret, algorithm=self.ALGORITHM)

    def verify(self, token: str) -> User:
        try:
            p = self._jwt.decode(token, self._secret, algorithms=[self.ALGORITHM], issuer=self.issuer, leeway=10,
                                 options={"require": ["exp", "iat", "iss", "sub", "cid"]})
            return User(user_id=str(p["sub"]), email=str(p.get("email", "")), customer_id=str(p["cid"]),
                        name=str(p.get("name", "")), is_demo=bool(p.get("demo", False)), token_version=int(p.get("ver", 0)))
        except (self._jwt.PyJWTError, ValueError, TypeError):
            raise InvalidCredentials("Invalid or expired token. Please log in again.") from None


# ------------------------------------------------------------------------------------------------ storage
class SqlAlchemyDb:
    """Production adapter. `get_engine` is a callable so importing the app never needs a database."""

    def __init__(self, get_engine: Callable[[], object]):
        self._get_engine = get_engine

    @property
    def dialect(self) -> str:
        return self._get_engine().dialect.name

    @contextmanager
    def transaction(self, customer_id: Optional[str] = None) -> Iterator[Callable[[str, object], list]]:
        """Yields execute(sql, params). `params` may be a dict, or a list of dicts for a bulk insert (executemany).
        With row-level security on, pass `customer_id` to act for one customer; without it the transaction is an auth-code
        transaction (login, sign-up, erasure) and may look across customers."""
        from sqlalchemy import text
        from sqlalchemy.exc import IntegrityError
        with self._get_engine().begin() as conn:
            if rls.enabled():
                if customer_id:
                    conn.exec_driver_sql("SELECT set_config('app.customer_id', %s, true)", (customer_id,))
                else:
                    conn.exec_driver_sql("SELECT set_config('app.bypass', 'on', true)")
            def execute(sql: str, params=None) -> list:
                try:
                    res = conn.execute(text(sql), params if params is not None else {})
                except IntegrityError as e:
                    raise DuplicateKey(str(e)) from e
                return [dict(r._mapping) for r in res] if res.returns_rows else []
            yield execute


def new_customer_id() -> str:
    """9 characters ('U' + 8 hex): fits VARCHAR(10), cannot be guessed or enumerated like C0001, C0002 ..."""
    return "U" + secrets.token_hex(4).upper()


def demo_email(customer_id: str) -> str:
    return f"demo-{customer_id.lower()}@demo.finsight.invalid"


_USER_SELECT = ("SELECT u.user_id, u.email, u.password_hash, u.customer_id, u.is_demo, u.email_verified_at, u.token_version, c.name "
                "FROM users u JOIN customers c ON c.customer_id = u.customer_id ")


def _row_to_user(row: dict) -> User:
    return User(user_id=row["user_id"], email=row["email"], customer_id=row["customer_id"], name=row["name"],
                is_demo=bool(row["is_demo"]), token_version=int(row.get("token_version") or 0))


class UserStore:
    """All SQL for users, email tokens and web sessions. Portable SQL (named :params): PostgreSQL and SQLite."""

    def __init__(self, db):
        self.db = db

    # -- lookups
    def _one(self, where: str, params: dict) -> Optional[dict]:
        with self.db.transaction() as q:
            rows = q(_USER_SELECT + where, params)
        return rows[0] if rows else None

    def find_by_email(self, email: str) -> Optional[dict]:
        return self._one("WHERE u.email = :email", {"email": email})

    def find_by_id(self, user_id: str) -> Optional[dict]:
        return self._one("WHERE u.user_id = :uid", {"uid": user_id})

    def customer_exists(self, customer_id: str) -> bool:
        with self.db.transaction() as q:
            return bool(q("SELECT 1 AS ok FROM customers WHERE customer_id = :cid", {"cid": customer_id}))

    def touch_login(self, user_id: str) -> None:
        with self.db.transaction() as q:
            q("UPDATE users SET last_login_at = :now WHERE user_id = :uid", {"now": _utcnow(), "uid": user_id})

    # -- creation
    def create_user(self, *, email: str, password_hash: str, name: str, monthly_income: float = 0.0,
                    customer_id: Optional[str] = None, is_demo: bool = False, verified: bool = True,
                    consents: Sequence[str] = ()) -> User:
        """Create a login. With no customer_id, also creates a fresh (empty) customer + account row for it;
        with one, links the login to that existing customer (admin / demo paths only)."""
        if customer_id is not None and not self.customer_exists(customer_id):
            raise NotAvailable(f"Unknown customer {customer_id!r}.")
        user_id, now = uuid.uuid4().hex, _utcnow()
        for _ in range(5):                                   # retry only if a random customer id collides
            cid = customer_id or new_customer_id()
            try:
                with self.db.transaction() as q:
                    if customer_id is None:
                        q("INSERT INTO customers (customer_id, name, monthly_income) VALUES (:cid, :name, :inc)",
                          {"cid": cid, "name": name, "inc": monthly_income})
                        q("INSERT INTO accounts (account_id, customer_id, opening_balance, opening_reserve, emi_amount, sip_amount) "
                          "VALUES (:aid, :cid, 0, 0, 0, 0)", {"aid": "A" + cid[1:], "cid": cid})
                    q("INSERT INTO users (user_id, email, password_hash, customer_id, is_demo, created_at, email_verified_at, token_version) "
                      "VALUES (:uid, :email, :ph, :cid, :demo, :now, :ver, 0)",
                      {"uid": user_id, "email": email, "ph": password_hash, "cid": cid, "demo": is_demo, "now": now,
                       "ver": now if verified else None})
                    for purpose in consents:
                        privacy.insert_consent(q, user_id, purpose, now)
                break
            except DuplicateKey:
                if self.find_by_email(email):
                    raise EmailTaken("An account with this email already exists.") from None
                if customer_id is not None:
                    raise InvalidInput("That customer already has a login.") from None
        else:
            raise RuntimeError("Could not allocate a customer id.")
        return _row_to_user(self.find_by_email(email))

    # -- email verification / password reset tokens (single use: the row is deleted when consumed)
    def create_token(self, user_id: str, purpose: str, ttl_s: int) -> str:
        raw, now = secrets.token_urlsafe(32), _utcnow()
        with self.db.transaction() as q:
            q("DELETE FROM auth_tokens WHERE expires_at < :now OR (user_id = :u AND purpose = :p)",
              {"now": now, "u": user_id, "p": purpose})           # also drops the user's older, unused token of this kind
            q("INSERT INTO auth_tokens (token_hash, user_id, purpose, expires_at, created_at) VALUES (:h, :u, :p, :exp, :now)",
              {"h": sha256_hex(raw), "u": user_id, "p": purpose, "exp": now + timedelta(seconds=ttl_s), "now": now})
        return raw

    def consume_token(self, raw: str, purpose: str) -> Optional[str]:
        """Atomically delete a valid token and return its user id (None if unknown, used, expired or wrong purpose)."""
        if not isinstance(raw, str) or not 10 <= len(raw) <= 200:
            return None
        with self.db.transaction() as q:
            rows = q("DELETE FROM auth_tokens WHERE token_hash = :h AND purpose = :p AND expires_at > :now RETURNING user_id",
                     {"h": sha256_hex(raw), "p": purpose, "now": _utcnow()})
        return rows[0]["user_id"] if rows else None

    def set_email_verified(self, user_id: str) -> None:
        with self.db.transaction() as q:
            q("UPDATE users SET email_verified_at = :now WHERE user_id = :u AND email_verified_at IS NULL",
              {"now": _utcnow(), "u": user_id})

    def set_password(self, user_id: str, password_hash: str) -> None:
        """New password: bump token_version (kills every old API token), delete web sessions and reset tokens.
        Completing a reset also proves control of the mailbox, so the email counts as verified."""
        with self.db.transaction() as q:
            q("UPDATE users SET password_hash = :ph, token_version = token_version + 1, "
              "email_verified_at = COALESCE(email_verified_at, :now) WHERE user_id = :u",
              {"ph": password_hash, "now": _utcnow(), "u": user_id})
            q("DELETE FROM web_sessions WHERE user_id = :u", {"u": user_id})
            q("DELETE FROM auth_tokens WHERE user_id = :u AND purpose = 'reset_password'", {"u": user_id})

    # -- server-side web sessions (Streamlit cookie)
    def create_web_session(self, user_id: str, days: int) -> str:
        raw, now = secrets.token_urlsafe(32), _utcnow()
        with self.db.transaction() as q:
            q("DELETE FROM web_sessions WHERE expires_at < :now", {"now": now})
            q("INSERT INTO web_sessions (session_hash, user_id, created_at, expires_at) VALUES (:h, :u, :now, :exp)",
              {"h": sha256_hex(raw), "u": user_id, "now": now, "exp": now + timedelta(days=days)})
        return raw

    def find_by_web_session(self, raw: str) -> Optional[dict]:
        if not isinstance(raw, str) or not 10 <= len(raw) <= 200:
            return None
        return self._one("JOIN web_sessions s ON s.user_id = u.user_id WHERE s.session_hash = :h AND s.expires_at > :now",
                         {"h": sha256_hex(raw), "now": _utcnow()})

    def delete_web_session(self, raw: str) -> None:
        with self.db.transaction() as q:
            q("DELETE FROM web_sessions WHERE session_hash = :h", {"h": sha256_hex(raw or "")})


# ------------------------------------------------------------------------------------------------ service
class AuthService:
    """Register / login / verify / reset / demo flows. Returns `User` objects; turning a User into an API token is the API's job."""

    def __init__(self, store: UserStore, *, login_limiter=None, reset_limiter=None, resend_limiter=None,
                 ip_fail_limiter=None, ip_mail_limiter=None, allow_signup: bool = True, demo_customer_id: Optional[str] = None, mailer=None,
                 app_base_url: str = "http://localhost:8501", require_verification: bool = True,
                 background_mail: bool = False, verify_ttl_s: int = 86400, reset_ttl_s: int = 1800, session_days: int = 7):
        from core.mailer import ConsoleMailer
        self.store = store
        self.login_limiter = login_limiter or WindowLimiter(max_events=5, window_s=900)
        self.reset_limiter = reset_limiter or WindowLimiter(max_events=3, window_s=3600)
        self.resend_limiter = resend_limiter or WindowLimiter(max_events=3, window_s=3600)
        self.ip_fail_limiter = ip_fail_limiter or WindowLimiter(max_events=20, window_s=900)     # failed logins per network address
        self.ip_mail_limiter = ip_mail_limiter or WindowLimiter(max_events=10, window_s=3600)    # sign-up / resend / reset requests per address
        self.allow_signup, self.demo_customer_id = allow_signup, demo_customer_id
        self.mailer = mailer or ConsoleMailer()
        self.app_base_url, self.require_verification, self.background_mail = app_base_url.rstrip("/"), require_verification, background_mail
        self.verify_ttl_s, self.reset_ttl_s, self.session_days = verify_ttl_s, reset_ttl_s, session_days

    @property
    def demo_enabled(self) -> bool:
        return bool(self.demo_customer_id)

    def limiter(self, name: str, max_events: int, window_s: int) -> DbWindowLimiter:
        """A shared (database-backed) limiter, e.g. for chat requests."""
        return DbWindowLimiter(self.store.db, name, max_events, window_s)

    # -- sign-up / login
    def register(self, email: str, password: str, name: str, monthly_income: float = 0.0, *,
                 accept_privacy_notice: bool = False, confirm_adult: bool = False, ip: Optional[str] = None) -> Optional[User]:
        """Create an account. With email verification on, an already-registered address does NOT raise: the owner gets a
        "you already have an account" email instead and None is returned, so the caller shows the same message either way."""
        if not self.allow_signup:
            raise SignupClosed("Sign-up is closed on this deployment.")
        if not (accept_privacy_notice and confirm_adult):
            raise InvalidInput("Please confirm that you are 18 or older and that you accept the privacy notice.")
        self._check_ip_mail(ip)
        user = self._create(email, password, name, monthly_income, verified=not self.require_verification,
                            consents=(privacy.PURPOSE_ACCOUNT, privacy.PURPOSE_ADULT), hide_existing=self.require_verification)
        if user is not None and self.require_verification:
            self._send_verification(user)
        return user

    def login(self, email: str, password: str, ip: Optional[str] = None) -> User:
        if ip and self.ip_fail_limiter.blocked(ip):
            raise Throttled("Too many failed attempts from your network. Please wait a few minutes and try again.")
        try:
            e = normalize_email(email)
        except InvalidInput:
            if ip:
                self.ip_fail_limiter.record(ip)
            raise InvalidCredentials(GENERIC_LOGIN_ERROR) from None
        if self.login_limiter.blocked(e):
            raise Throttled("Too many failed attempts. Please wait a few minutes and try again.")
        row = self.store.find_by_email(e)
        pw = password if isinstance(password, str) and len(password) <= MAX_PASSWORD else ""
        ok = verify_password(pw, row["password_hash"] if row else _dummy_hash())
        if not (row and ok and pw):
            self.login_limiter.record(e)
            if ip:
                self.ip_fail_limiter.record(ip)
            raise InvalidCredentials(GENERIC_LOGIN_ERROR)
        if self.require_verification and not row["email_verified_at"] and not row["is_demo"]:
            raise EmailNotVerified("Please verify your email address first. Use \"Resend verification email\" if you cannot find the message.")
        self.login_limiter.reset(e)
        self.store.touch_login(row["user_id"])
        return _row_to_user(row)

    def ensure_current(self, user: User) -> User:
        """Re-check a token's user against the database: still exists, and no password reset since the token was issued."""
        row = self.store.find_by_id(user.user_id)
        if row is None or int(row.get("token_version") or 0) != user.token_version or row["customer_id"] != user.customer_id:
            raise InvalidCredentials("Your session is no longer valid. Please log in again.")
        return _row_to_user(row)

    # -- email verification
    def verify_email(self, token: str) -> User:
        uid = self.store.consume_token(token, "verify_email")
        if not uid:
            raise InvalidInput("This verification link is invalid or has expired. Request a new one from the login page.")
        self.store.set_email_verified(uid)
        return _row_to_user(self.store.find_by_id(uid))

    def resend_verification(self, email: str, ip: Optional[str] = None) -> None:
        """Always returns None: the caller shows the same message whether or not the email exists."""
        self._check_ip_mail(ip)
        try:
            e = normalize_email(email)
        except InvalidInput:
            return
        if self.resend_limiter.blocked(e):
            return
        self.resend_limiter.record(e)
        row = self.store.find_by_email(e)
        if row and not row["email_verified_at"] and not row["is_demo"]:
            self._send_verification(_row_to_user(row))

    # -- password reset
    def request_password_reset(self, email: str, ip: Optional[str] = None) -> None:
        """Always returns None (no hint whether the email has an account)."""
        self._check_ip_mail(ip)
        try:
            e = normalize_email(email)
        except InvalidInput:
            return
        if self.reset_limiter.blocked(e):
            return
        self.reset_limiter.record(e)
        row = self.store.find_by_email(e)
        if row and not row["is_demo"]:
            raw = self.store.create_token(row["user_id"], "reset_password", self.reset_ttl_s)
            self._deliver(e, "Reset your FinSight password",
                          "Someone asked to reset the password for your FinSight account.\n\n"
                          f"Choose a new password here (the link works once and expires in {self.reset_ttl_s // 60} minutes):\n"
                          f"{self._link('reset', raw)}\n\nIf this was not you, ignore this email; your password has not changed.")

    def reset_password(self, token: str, new_password: str) -> User:
        check_password_rules(new_password)                       # before consuming, so a typo does not burn the link
        uid = self.store.consume_token(token, "reset_password")
        if not uid:
            raise InvalidInput("This reset link is invalid or has expired. Request a new one from the login page.")
        self.store.set_password(uid, hash_password(new_password))
        row = self.store.find_by_id(uid)
        self.login_limiter.reset(row["email"])
        return _row_to_user(row)

    # -- web sessions (Streamlit)
    def start_web_session(self, user: User) -> str:
        return self.store.create_web_session(user.user_id, self.session_days)

    def user_from_web_session(self, raw: Optional[str]) -> Optional[User]:
        row = self.store.find_by_web_session(raw or "")
        if row is None or (self.require_verification and not row["email_verified_at"] and not row["is_demo"]):
            return None
        return _row_to_user(row)

    def end_web_session(self, raw: Optional[str]) -> None:
        if raw:
            self.store.delete_web_session(raw)

    # -- demo / admin
    def demo_user(self) -> User:
        """The shared demo login (no password: nobody can log into it with one). Enabled by DEMO_CUSTOMER_ID."""
        cid = self.demo_customer_id
        if not cid:
            raise NotAvailable("The demo account is not enabled.")
        email = demo_email(cid)
        row = self.store.find_by_email(email)
        if row is None:
            try:
                return self.store.create_user(email=email, password_hash=hash_password(secrets.token_urlsafe(32)),
                                              name="Demo", customer_id=cid, is_demo=True, verified=True)
            except EmailTaken:                              # another request created it first
                row = self.store.find_by_email(email)
        return _row_to_user(row)

    def admin_create_user(self, email: str, password: str, name: str, monthly_income: float = 0.0,
                          customer_id: Optional[str] = None) -> User:
        """CLI only, never exposed over HTTP. Marked verified (the admin vouches for the address)."""
        return self._create(email, password, name, monthly_income, customer_id, verified=True)

    # -- privacy
    def delete_account(self, user: User, password: str) -> None:
        """Erase the login and (for sign-up accounts) all the user's data. Needs the current password."""
        if user.is_demo:
            raise InvalidInput("The shared demo account cannot be deleted.")
        row = self.store.find_by_id(user.user_id)
        key = row["email"] if row else user.email
        if self.login_limiter.blocked(key):
            raise Throttled("Too many failed attempts. Please wait a few minutes and try again.")
        if not row or not isinstance(password, str) or not verify_password(password, row["password_hash"]):
            self.login_limiter.record(key)
            raise InvalidCredentials("That password is not correct.")
        privacy.erase_user(self.store.db, user.user_id, user.customer_id)

    # -- internals
    def _create(self, email, password, name, monthly_income, customer_id=None, *, verified, consents=(),
                hide_existing: bool = False) -> Optional[User]:
        e, name, income = normalize_email(email), _clean_name(name), _clean_income(monthly_income)
        check_password_rules(password)
        password_hash = hash_password(password)             # always, so a taken address costs the same time as a free one

        def taken() -> None:
            if hide_existing:
                self._notify_existing_account(e)
                return None
            raise EmailTaken("An account with this email already exists.")

        if self.store.find_by_email(e):
            return taken()
        try:
            return self.store.create_user(email=e, password_hash=password_hash, name=name, monthly_income=income,
                                          customer_id=customer_id, verified=verified, consents=consents)
        except EmailTaken:                                   # lost a race with another sign-up for the same address
            return taken()

    def _check_ip_mail(self, ip: Optional[str]) -> None:
        if ip:
            if self.ip_mail_limiter.blocked(ip):
                raise Throttled("Too many requests from your network. Please try again later.")
            self.ip_mail_limiter.record(ip)

    def _notify_existing_account(self, email: str) -> None:
        key = f"signup:{email}"                              # same silent 3-per-hour cap as resend, so nobody can mail-bomb an address
        if self.resend_limiter.blocked(key):
            return
        self.resend_limiter.record(key)
        self._deliver(email, "You already have a FinSight account",
                      "Someone (hopefully you) tried to create a FinSight account with this email address, but one already exists.\n\n"
                      f"Log in here: {self.app_base_url}\nForgot the password? Use \"Forgot your password?\" on the login page.\n\n"
                      "If this was not you, you can ignore this email; nothing has changed.")

    def _link(self, kind: str, token: str) -> str:
        return f"{self.app_base_url}/?{kind}={token}"

    def _send_verification(self, user: User) -> None:
        raw = self.store.create_token(user.user_id, "verify_email", self.verify_ttl_s)
        self._deliver(user.email, "Verify your FinSight email address",
                      f"Hi {user.name},\n\nPlease confirm your email address to finish creating your FinSight account "
                      f"(the link expires in {self.verify_ttl_s // 3600} hours):\n{self._link('verify', raw)}\n\n"
                      "If you did not sign up, ignore this email and the account will stay unverified.")

    def _deliver(self, to: str, subject: str, body: str) -> None:
        """Send mail; never raise into the caller (and run in the background in production so response time
        does not reveal whether an address has an account)."""
        def send():
            try:
                self.mailer.send(to, subject, body)
            except Exception:
                log.exception("Could not send %r email", subject)
        if self.background_mail:
            threading.Thread(target=send, daemon=True).start()
        else:
            send()


def build_auth_service(get_engine: Callable[[], object], env=os.environ) -> AuthService:
    from core.mailer import build_mailer
    db = SqlAlchemyDb(get_engine)
    truthy = lambda k, d: env.get(k, d).strip().lower() in ("1", "true", "yes")
    return AuthService(UserStore(db),
                       login_limiter=DbWindowLimiter(db, "login", 5, 900),
                       reset_limiter=DbWindowLimiter(db, "forgot", 3, 3600),
                       resend_limiter=DbWindowLimiter(db, "resend", 3, 3600),
                       ip_fail_limiter=DbWindowLimiter(db, "login_ip", 20, 900),
                       ip_mail_limiter=DbWindowLimiter(db, "mail_ip", 10, 3600),
                       allow_signup=truthy("ALLOW_SIGNUP", "true"),
                       demo_customer_id=(env.get("DEMO_CUSTOMER_ID") or "").strip() or None,
                       mailer=build_mailer(env), app_base_url=env.get("APP_BASE_URL", "http://localhost:8501"),
                       require_verification=truthy("REQUIRE_EMAIL_VERIFICATION", "true"), background_mail=True,
                       session_days=int(env.get("SESSION_DAYS", "7")))


# ------------------------------------------------------------------------------------------------ CLI
def _cli(argv=None) -> None:
    import argparse
    import getpass

    ap = argparse.ArgumentParser(prog="python -m core.auth", description="FinSight user administration")
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("create-user", help="create a verified login, optionally linked to an existing customer such as C0001")
    c.add_argument("--email", required=True)
    c.add_argument("--name", required=True)
    c.add_argument("--customer", default=None, help="existing customer id to link (omit to create a new empty customer)")
    c.add_argument("--income", type=float, default=0.0)
    a = ap.parse_args(argv)

    password = getpass.getpass("Password: ")
    if password != getpass.getpass("Repeat password: "):
        raise SystemExit("Passwords do not match.")
    from sqlalchemy import create_engine
    engine = create_engine(os.getenv("DATABASE_URL", DEFAULT_DB_URL), pool_pre_ping=True)
    svc = AuthService(UserStore(SqlAlchemyDb(lambda: engine)), require_verification=False)
    try:
        u = svc.admin_create_user(a.email, password, a.name, a.income, a.customer)
    except AuthError as e:
        raise SystemExit(f"Error: {e}")
    print(f"Created {u.email} -> customer {u.customer_id}")


if __name__ == "__main__":
    _cli()
