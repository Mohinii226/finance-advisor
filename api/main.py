"""
FastAPI application: thin routing glue over api/handlers.py and core/auth.py.

    uvicorn api.main:app --host 0.0.0.0 --port 8000

Environment
  DATABASE_URL        PostgreSQL URL
  JWT_SECRET          REQUIRED, >= 32 random characters (the app refuses to start without it)
  JWT_TTL_MINUTES     token lifetime, default 60
  ALLOW_SIGNUP        "true" (default) or "false" to close public sign-up
  DEMO_CUSTOMER_ID    optional, e.g. C0001: enables POST /auth/demo (shared, password-less demo login)
  REQUIRE_EMAIL_VERIFICATION  "true" (default): new accounts must click the emailed link before they can log in
  APP_BASE_URL        public URL of the Streamlit app, used in emailed links (e.g. https://app.example.com)
  MAIL_BACKEND        console (dev only) | smtp (+ SMTP_HOST, SMTP_PORT, SMTP_USER, SMTP_PASSWORD, SMTP_FROM); see core/mailer.py
  CHAT_PER_HOUR / CHAT_PER_HOUR_DEMO   per-user chat limits (default 60 / 20; the demo login is one shared bucket).
                      Limits live in PostgreSQL (rate_events), so they hold across workers and restarts.
  ANTHROPIC_API_KEY + ADVISOR_MODEL    optional; otherwise the offline rule-based demo chat is used

WHO IS CALLING is decided only by the bearer token. No route takes a customer id from the URL or the body: every /me/...
route and /chat uses the `cid` claim of the verified token, so one user cannot read another user's data by changing an id.

STATUS: like the original file, this is NOT exercised by the test-suite in my sandbox (FastAPI is not installed there); it is
compile-checked only. The behaviour behind it (core/auth.py, api/handlers.py) is unit-tested; tests/test_auth_api.py contains
TestClient tests that run automatically when FastAPI is installed.
"""
from __future__ import annotations

import os
from typing import Optional

from fastapi import Depends, FastAPI, HTTPException, Query, Request
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from core import rls
from core.auth import AuthError, TokenManager, User, build_auth_service
from core.insights import make_runner
from .handlers import ApiError, Backend
from .schemas import (ChatRequest, DeleteAccountRequest, EmailRequest, LoginRequest, RegisterRequest, ResetPasswordRequest,
                      SavingsRequest, VerifyEmailRequest)

DEFAULT_DB_URL = "postgresql+psycopg2://finance:finance@localhost:5432/finance_advisor"


def _lazy_engine():
    """Create the SQLAlchemy engine on first use so importing the app does not need a database."""
    state = {}

    def get():
        if "engine" not in state:
            from sqlalchemy import create_engine
            # APP_DATABASE_URL (the limited finsight_app role) wins when set; prepare_engine is a no-op unless RLS=on
            state["engine"] = rls.prepare_engine(create_engine(rls.database_url(DEFAULT_DB_URL), pool_pre_ping=True))
        return state["engine"]
    return get


def _lazy_runner(get_engine):
    state = {}

    def run(name, **params):
        if "run" not in state:
            state["run"] = make_runner(get_engine())
        return state["run"](name, **params)
    return run


def create_app(backend: Optional[Backend] = None, auth=None, tokens: Optional[TokenManager] = None) -> FastAPI:
    get_engine = _lazy_engine()
    backend = backend or Backend(_lazy_runner(get_engine))
    auth = auth or build_auth_service(get_engine)
    tokens = tokens or TokenManager.from_env()          # raises at start-up if JWT_SECRET is missing or too short
    chat_limit = auth.limiter("chat", int(os.getenv("CHAT_PER_HOUR", "60")), 3600)
    demo_chat_limit = auth.limiter("chat_demo", int(os.getenv("CHAT_PER_HOUR_DEMO", "20")), 3600)
    bearer = HTTPBearer(auto_error=False)
    app = FastAPI(title="Intelligent Personal Finance Advisor", version="0.3.0")

    def session(user: User) -> dict:
        return {"access_token": tokens.issue(user), "token_type": "bearer", "expires_in": tokens.ttl_seconds,
                "user": user.public()}

    def current_user(creds: Optional[HTTPAuthorizationCredentials] = Depends(bearer)) -> User:
        challenge = {"WWW-Authenticate": "Bearer"}
        if creds is None:
            raise HTTPException(status_code=401, detail="Not authenticated.", headers=challenge)
        try:
            return auth.ensure_current(tokens.verify(creds.credentials))     # also rejects tokens issued before a password reset
        except AuthError as e:
            raise HTTPException(status_code=e.status, detail=str(e), headers=challenge)

    def call(fn, *args):
        try:
            return fn(*args)
        except (ApiError, AuthError) as e:
            raise HTTPException(status_code=e.status, detail=str(e))

    def call_as(user: User, fn, *args):
        """Run a data call as ONE customer: with RLS on the database itself then refuses rows of any other customer."""
        with rls.customer_scope(user.customer_id):
            return call(fn, *args)

    def ip_of(request: Request):
        """The caller's address. Behind Caddy, uvicorn --proxy-headers turns X-Forwarded-For into request.client.host."""
        return request.client.host if request.client else None

    # ------------------------------------------------------------------ public
    @app.get("/health")
    def health():
        return {"status": "ok", "auth": "jwt", "signup": "open" if auth.allow_signup else "closed",
                "demo_login": auth.demo_enabled, "email_verification": auth.require_verification,
                "chat": "anthropic" if os.getenv("ANTHROPIC_API_KEY") else "offline rule-based demo"}

    def registered(body: RegisterRequest, ip) -> dict:
        user = auth.register(body.email, body.password, body.name, body.monthly_income,
                             accept_privacy_notice=body.accept_privacy_notice, confirm_adult=body.confirm_adult, ip=ip)
        if auth.require_verification:
            # Identical for a new and an already-registered address (an existing owner gets a different email instead).
            return {"verification_required": True,
                    "message": "We sent an email to that address. Open it and follow the link to finish creating your account."}
        return session(user)

    @app.post("/auth/register", status_code=201)
    def register(body: RegisterRequest, request: Request):
        return call(lambda: registered(body, ip_of(request)))

    @app.post("/auth/verify-email")
    def verify_email(body: VerifyEmailRequest):
        return call(lambda: {"verified": True, "email": auth.verify_email(body.token).email})

    @app.post("/auth/resend-verification", status_code=202)
    def resend_verification(body: EmailRequest, request: Request):
        call(auth.resend_verification, body.email, ip_of(request))
        return {"message": "If that address has an unverified account, a new verification email is on its way."}

    @app.post("/auth/forgot-password", status_code=202)
    def forgot_password(body: EmailRequest, request: Request):
        call(auth.request_password_reset, body.email, ip_of(request))
        return {"message": "If that address has an account, a password-reset email is on its way."}

    @app.post("/auth/reset-password")
    def reset_password(body: ResetPasswordRequest):
        return call(lambda: {"reset": True, "email": auth.reset_password(body.token, body.new_password).email})

    @app.post("/auth/login")
    def login(body: LoginRequest, request: Request):
        return call(lambda: session(auth.login(body.email, body.password, ip=ip_of(request))))

    @app.post("/auth/demo")
    def demo():
        return call(lambda: session(auth.demo_user()))

    # ------------------------------------------------------------------ authenticated: the caller's own data only
    @app.get("/me")
    def me(user: User = Depends(current_user)):
        return user.public()

    @app.post("/me/delete")
    def delete_me(body: DeleteAccountRequest, user: User = Depends(current_user)):
        """Erase the account and all of the user's data (needs the current password)."""
        call(auth.delete_account, user, body.password)
        return {"deleted": True}

    @app.get("/me/summary")
    def summary(month: Optional[str] = None, user: User = Depends(current_user)):
        return call_as(user, backend.summary, user.customer_id, month)

    @app.get("/me/categories")
    def categories(month: Optional[str] = None, limit: int = Query(10, ge=1, le=10), user: User = Depends(current_user)):
        return call_as(user, backend.categories, user.customer_id, month, limit)

    @app.get("/me/alerts")
    def alerts(month: Optional[str] = None, user: User = Depends(current_user)):
        return call_as(user, backend.alerts, user.customer_id, month)

    @app.get("/me/health-score")
    def health_score(month: Optional[str] = None, user: User = Depends(current_user)):
        return call_as(user, backend.health_score, user.customer_id, month)

    @app.post("/me/savings-plan")
    def savings_plan(body: SavingsRequest, user: User = Depends(current_user)):
        return call_as(user, backend.savings_plan, user.customer_id, body.target_amount)

    @app.get("/me/insights")
    def insights(month: Optional[str] = None, topic: Optional[str] = None, user: User = Depends(current_user)):
        return call_as(user, backend.insights, user.customer_id, month, topic)

    @app.post("/chat")
    def chat(body: ChatRequest, user: User = Depends(current_user)):
        limiter = demo_chat_limit if user.is_demo else chat_limit
        if limiter.blocked(user.user_id):
            raise HTTPException(status_code=429, detail="Chat limit reached for now. Please try again later.")
        limiter.record(user.user_id)
        return call_as(user, backend.chat, user.customer_id, body.message, body.session_id)

    return app


app = create_app()
