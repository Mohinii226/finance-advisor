"""
Streamlit dashboard + chat for the Personal Finance Advisor.

    streamlit run app/streamlit_app.py

Needs the database loaded (see README). Chat works offline without an API key (rule-based demo); set ANTHROPIC_API_KEY for a real model.
All logic lives in app/data_access.py, app/charts.py, app/chat.py and core/; this file only lays things out.

LOGIN: nothing is shown until the visitor logs in (core/auth.py). The customer comes from the logged-in user, never from a
widget, so people cannot browse other customers. Logins survive a browser refresh through a random session id kept in a cookie
and, hashed, in the `web_sessions` table (needs Streamlit >= 1.37 to read the cookie; best effort: without it a refresh just asks
for a login again). Sessions last SESSION_DAYS (default 7) and end on logout or password reset. Emailed links (?verify=, ?reset=)
are handled before the login screen. Signed-in users can import CSV bank statements and delete their data under "Your data".
DEMO_CUSTOMER_ID enables a "Try the demo" button; ALLOW_SIGNUP=false closes sign-up.
"""
from __future__ import annotations

import inspect
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:                  # `streamlit run app/streamlit_app.py` puts app/, not the repo root, on sys.path
    sys.path.insert(0, str(ROOT))

import pandas as pd
import streamlit as st

from app import charts
from app.chat import SAMPLE_QUESTIONS, chat_turn
from app.data_access import (available_months, category_history, largest_transactions, load_model_lab, money,
                             month_str, safe, seasonal_index, slider_bounds, spend_history, top_merchants, weekday_pattern)
from core import privacy, rls
from core.auth import AuthError, User, build_auth_service
from core.statements import import_statement, parse_statement, preview_categories
from core.service import AdvisorService

_RUNNER = None
DB_HELP = ("Could not reach the database. Start it with `docker compose up -d db`, load the data with "
           "`python -m data_gen.load_to_postgres --reset`, and check `DATABASE_URL` in your `.env`.")


# ---------------------------------------------------------------------------------------------- data plumbing
@st.cache_resource(show_spinner=False)
def _engine():
    from sqlalchemy import create_engine
    url = rls.database_url("postgresql+psycopg2://finance:finance@localhost:5432/finance_advisor")    # APP_DATABASE_URL wins
    return rls.prepare_engine(create_engine(url, pool_pre_ping=True))      # no-op unless RLS=on


@st.cache_resource(show_spinner=False)
def _engine_runner():
    from core.insights import make_runner
    return make_runner(_engine())


@st.cache_resource(show_spinner=False)
def _auth_service():
    return build_auth_service(_engine)


@st.cache_data(ttl=600, show_spinner=False)
def _cached_query(name: str, params: tuple, scope: str | None) -> pd.DataFrame:
    """`scope` (the logged-in customer) is part of the cache key, so one user's cached result can never be served to another
    (it matters for queries without a customer parameter once row-level security makes them per-customer)."""
    return _RUNNER(name, **dict(params))


def run(name: str, **params) -> pd.DataFrame:
    """Cached query runner handed to every data function (params must be hashable primitives)."""
    return _cached_query(name, tuple(sorted(params.items())), rls.CURRENT_CUSTOMER.get())


def show_chart(fig) -> None:
    """Streamlit renamed the chart width argument across versions; use whichever this install supports."""
    try:
        if "width" in inspect.signature(st.plotly_chart).parameters:
            st.plotly_chart(fig, width="stretch")
            return
    except (TypeError, ValueError):
        pass
    st.plotly_chart(fig, use_container_width=True)


def table(df: pd.DataFrame) -> None:
    st.dataframe(df, hide_index=True)


# ---------------------------------------------------------------------------------------------- login
COOKIE_NAME = "finsight_session"
_SESSION_KEYS = ("user", "web_token", "chats", "pending_question")


def _cookie_js(value: str, max_age: int) -> None:
    """Set (or, with max_age=0, clear) the login cookie from the browser through a hidden component.
    Best effort: if the browser or a Streamlit upgrade blocks it, the only effect is that a refresh asks for a login again."""
    import json
    import streamlit.components.v1 as components
    secure = "; Secure" if os.getenv("COOKIE_SECURE", "true").strip().lower() in ("1", "true", "yes") else ""
    cookie = json.dumps(f"{COOKIE_NAME}={value}; path=/; max-age={max_age}; SameSite=Lax{secure}")
    components.html("<script>(function(){var c=%s;try{window.parent.document.cookie=c}catch(e){document.cookie=c}})();</script>" % cookie,
                    height=0)


def _client_ip() -> str | None:
    """The visitor's address as seen through the proxy (Caddy sets X-Forwarded-For). None when unknown, in which case only the
    per-account limits apply."""
    try:
        return st.context.headers.get("X-Forwarded-For", "").split(",")[0].strip() or None
    except Exception:
        return None


def _flush_cookie_commands() -> None:
    cmd = st.session_state.pop("_cookie_cmd", None)
    if cmd:
        _cookie_js(*cmd)


def _read_cookie() -> str | None:
    try:
        return st.context.cookies.get(COOKIE_NAME)          # needs Streamlit >= 1.37
    except Exception:
        return None


def _enter(user, token: str, auth) -> None:
    st.session_state["user"] = dict(user.public(), expires=time.time() + auth.session_days * 86400)
    st.session_state["web_token"] = token


def _current_user(auth) -> dict | None:
    """The logged-in user of this browser: from this session, or restored from the cookie after a refresh.
    The server-side session is re-checked on every run, so a password reset or a logout elsewhere ends it."""
    u = st.session_state.get("user")
    if u:
        if u["expires"] < time.time() or auth.user_from_web_session(st.session_state.get("web_token")) is None:
            _logout(auth, rerun=False)
            return None
        return u
    token = _read_cookie()
    user = auth.user_from_web_session(token) if token else None
    if user:
        _enter(user, token, auth)
        return st.session_state["user"]
    return None


def _start_session(auth, user) -> None:
    token = auth.start_web_session(user)
    for k in _SESSION_KEYS:
        st.session_state.pop(k, None)
    _enter(user, token, auth)
    st.session_state["_cookie_cmd"] = (token, auth.session_days * 86400)
    st.rerun()


def _logout(auth, rerun: bool = True) -> None:
    auth.end_web_session(st.session_state.get("web_token"))
    for k in _SESSION_KEYS:
        st.session_state.pop(k, None)
    st.session_state["_cookie_cmd"] = ("", 0)
    if rerun:
        st.rerun()


def _attempt(fn, on_success) -> None:
    """Run a login / sign-up / reset action; show a readable error, or hand the result to on_success."""
    try:
        result = fn()
    except AuthError as e:
        st.error(str(e))
        return
    except Exception as e:                                  # database hiccup etc.: never show a traceback to a visitor
        st.error(f"Something went wrong ({type(e).__name__}). Please try again.")
        return
    on_success(result)


def _privacy_notice() -> str:
    p = ROOT / "docs" / "privacy_notice_template.md"
    return p.read_text(encoding="utf-8") if p.exists() else "The privacy notice is not available on this deployment."


def _after_register(auth, user, email: str) -> None:
    if auth.require_verification:
        # The same words whether the address is new or already registered (the owner of an existing account gets a different email).
        st.success(f"We sent an email to {email}. Open it and follow the link to finish creating your account.")
    else:
        _start_session(auth, user)


def email_link_screen(auth) -> bool:
    """Handles ?verify=TOKEN and ?reset=TOKEN links from emails. Returns True when it drew a screen (caller stops)."""
    qp = st.query_params
    if "verify" in qp:
        token = qp["verify"]
        done = st.session_state.get("verify_result")
        if not done or done[0] != token:                    # a rerun must not consume the single-use token twice
            try:
                auth.verify_email(token)
                done = (token, True, "Your email address is verified. You can log in now.")
            except AuthError as e:
                done = (token, False, str(e))
            except Exception as e:
                done = (token, False, f"Something went wrong ({type(e).__name__}). Please try again.")
            st.session_state["verify_result"] = done
        (st.success if done[1] else st.error)(done[2])
        if st.button("Go to the login page"):
            st.session_state.pop("verify_result", None)
            st.query_params.clear()
            st.rerun()
        return True
    if "reset" in qp:
        token = qp["reset"]
        if st.session_state.get("reset_done"):
            st.success("Your password was changed and all other sessions were signed out. You can log in now.")
        else:
            st.subheader("Choose a new password")
            with st.form("reset_form"):
                p1 = st.text_input("New password (at least 10 characters)", type="password")
                p2 = st.text_input("Repeat the new password", type="password")
                go = st.form_submit_button("Change password")
            if go:
                if p1 != p2:
                    st.error("The two passwords do not match.")
                else:
                    _attempt(lambda: auth.reset_password(token, p1), lambda _u: (st.session_state.update(reset_done=True), st.rerun()))
        if st.button("Go to the login page"):
            st.session_state.pop("reset_done", None)
            st.query_params.clear()
            st.rerun()
        return True
    return False


def login_screen(auth) -> None:
    _, mid, _ = st.columns([1, 2, 1])
    with mid:
        st.subheader("Log in to see your finances")
        tab_in, tab_up = st.tabs(["Log in", "Create account"])
        with tab_in:
            with st.form("login_form"):
                email = st.text_input("Email")
                password = st.text_input("Password", type="password")
                go = st.form_submit_button("Log in")
            if go:
                _attempt(lambda: auth.login(email, password, ip=_client_ip()), lambda u: _start_session(auth, u))
            with st.expander("Forgot your password?"):
                with st.form("forgot_form"):
                    fe = st.text_input("Your account email", key="forgot_email")
                    fgo = st.form_submit_button("Email me a reset link")
                if fgo:
                    _attempt(lambda: auth.request_password_reset(fe, ip=_client_ip()),
                             lambda _r: st.success("If that address has an account, a reset link is on its way. It works once and expires soon."))
            if auth.require_verification:
                with st.expander("Did not get the verification email?"):
                    with st.form("resend_form"):
                        re_ = st.text_input("Your account email", key="resend_email")
                        rgo = st.form_submit_button("Resend verification email")
                    if rgo:
                        _attempt(lambda: auth.resend_verification(re_, ip=_client_ip()),
                                 lambda _r: st.success("If that address has an unverified account, a new link is on its way."))
        with tab_up:
            if not auth.allow_signup:
                st.info("Sign-up is closed on this deployment.")
            elif os.getenv("APP_ENV", "").lower() == "production" and "[[" in _privacy_notice():
                st.error("Sign-up is switched off until the site owner completes the privacy notice (docs/privacy_notice_template.md).")
            else:
                with st.expander("Read the privacy notice"):
                    st.markdown(_privacy_notice())
                with st.form("signup_form"):
                    name = st.text_input("Name")
                    email2 = st.text_input("Email", key="signup_email")
                    password2 = st.text_input("Password (at least 10 characters)", type="password", key="signup_pw")
                    income = st.number_input("Monthly income in Rs (optional, helps classify your salary)", min_value=0,
                                             max_value=10_000_000, value=0, step=1000)
                    adult = st.checkbox("I am 18 years of age or older.")
                    notice = st.checkbox("I have read the privacy notice and agree to FinSight processing my name, email and "
                                         "income to run my account.")
                    go2 = st.form_submit_button("Create account")
                if go2:
                    _attempt(lambda: auth.register(email2, password2, name, float(income), accept_privacy_notice=notice,
                                                   confirm_adult=adult, ip=_client_ip()),
                             lambda u: _after_register(auth, u, email2.strip()))
        if auth.demo_enabled:
            st.divider()
            st.caption("Just looking around? The demo account uses synthetic data.")
            if st.button("Try the demo account"):
                _attempt(auth.demo_user, lambda u: _start_session(auth, u))


def _val(p, key):
    try:
        v = p[key]
    except (KeyError, TypeError):
        return None
    return None if v is None or (isinstance(v, float) and pd.isna(v)) else v


def profile_caption(p) -> str:
    """New accounts have no occupation / city yet, so only show what exists."""
    parts = [str(_val(p, k)) for k in ("occupation", "city", "income_type") if _val(p, k)]
    income = _val(p, "monthly_income")
    if income and float(income) > 0:
        parts.append(f"income about {money(income)}/month")
    return " | ".join(parts)


def _user_obj(u: dict) -> User:
    return User(user_id=u["user_id"], email=u["email"], customer_id=u["customer_id"], name=u["name"], is_demo=u["is_demo"])


def _declared_income(cid: str) -> float:
    ok, df = safe(run, "customer_profile", cid=cid)
    try:
        return float(df["monthly_income"].iloc[0] or 0) if ok and len(df) else 0.0
    except (TypeError, ValueError, KeyError):
        return 0.0


# ---------------------------------------------------------------------------------------------- tabs
def tab_overview(svc, run_, cid, month):
    ms = month_str(month)
    ok_h, health = safe(svc.get_health_score, ms)
    ok_s, story = safe(svc.explain_spending_change, ms)
    ok_f, fc = safe(svc.get_spending_forecast)
    ok_a, alerts = safe(svc.get_alerts, ms)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric(f"Health score ({health['band']})" if ok_h else "Health score", f"{health['total']:.0f} / 100" if ok_h else "n/a", help=None if ok_h else health)
    c2.metric(f"Spending in {pd.Timestamp(month):%b %Y}", money(story["total_spend"]) if ok_s else "n/a",
              f"{story['difference']:+,.0f} vs usual" if ok_s else None, delta_color="inverse", help=None if ok_s else story)
    c3.metric("Expected next month", money(fc["expected_spend"]) if ok_f else "n/a",
              f"{money(fc['likely_range_low'])} to {money(fc['likely_range_high'])}" if ok_f and fc.get("likely_range_low") is not None else None,
              delta_color="off", help=None if ok_f else fc)
    c4.metric("Overspending alerts", alerts["alert_count"] if ok_a else "n/a", help=None if ok_a else alerts)

    left, right = st.columns([3, 2])
    with left:
        hist = spend_history(run_, cid, month, 12)
        show_chart(charts.spend_trend(hist, fc if ok_f else None))
    with right:
        ok_t, top = safe(svc.get_top_categories, ms, 6)
        if ok_t:
            show_chart(charts.category_donut(top["top_categories"]))
        else:
            st.info(top)
    if ok_s:
        st.info(story["story"])
    else:
        st.caption(story)


def tab_spending(svc, run_, cid, month):
    long_df = category_history(run_, cid, month, 12)
    if long_df.empty:
        st.info("No spending recorded in this period.")
        return
    show_chart(charts.category_stack(long_df))
    a, b = st.columns(2)
    with a:
        wd = weekday_pattern(run_, cid, month)
        if len(wd):
            show_chart(charts.weekday_bar(wd))
    with b:
        seas = seasonal_index(run_, cid)
        if len(seas):
            show_chart(charts.seasonal_line(seas))
    c, d = st.columns(2)
    with c:
        st.subheader("Top discretionary spending (6 months)")
        tm = top_merchants(run_, cid, month)
        table(tm.assign(spend=tm["spend"].map(money))[["merchant", "category", "spend", "txns"]] if len(tm) else tm)
    with d:
        st.subheader(f"Largest transactions in {pd.Timestamp(month):%b %Y}")
        lt = largest_transactions(run_, cid, month)
        table(lt.assign(amount=lt["amount"].map(money)) if len(lt) else lt)


def tab_health(svc, run_, cid, month):
    ok, h = safe(svc.get_health_score, month_str(month))
    if not ok:
        st.info(h)
    else:
        a, b = st.columns([2, 3])
        with a:
            show_chart(charts.health_gauge(h["total"], h["band"]))
        with b:
            show_chart(charts.components_bar(h["components"]))
        st.write(h["explanation"])
    st.divider()
    st.subheader("Savings planner")
    st.caption("Based on your most recent three months of spending.")
    income = float(svc.profile["monthly_income"])
    lo, hi, default = slider_bounds(income)
    target = st.slider("How much do you want to save each month (Rs)?", lo, hi, default, step=500)
    ok_p, plan = safe(svc.suggest_savings, float(target))
    if not ok_p:
        st.warning(plan)
        return
    st.markdown(plan["plan_text"])
    if plan["cuts"]:
        cuts = pd.DataFrame(plan["cuts"])
        table(pd.DataFrame({"Category": cuts["category"], "Now (Rs/month)": cuts["current"].map(lambda x: f"{x:,.0f}"),
                            "Cut (Rs/month)": cuts["cut"].map(lambda x: f"{x:,.0f}"), "Cut %": cuts["pct_cut"].map(lambda x: f"{x:.0f}%"),
                            "Per year (Rs)": cuts["annual_saving"].map(lambda x: f"{x:,.0f}")}))
    imp = plan.get("health_score_impact")
    if imp:
        st.metric("Health score if you follow this plan", f"{imp['after']:.0f}", f"{imp['gain']:+.1f}")


def tab_alerts(svc, run_, cid, month):
    ok, al = safe(svc.get_alerts, month_str(month))
    st.subheader("Overspending alerts")
    if not ok:
        st.info(al)
    elif not al["alerts"]:
        st.success("No overspending alerts this month.")
        st.caption(al["note"])
    else:
        for a in al["alerts"]:
            st.warning(a["message"])
    st.subheader("Insights")
    ok_i, ins = safe(svc.get_insights, None, month_str(month))
    if not ok_i:
        st.info(ins)
        return
    titles = {"largest_category": "Largest category", "why_spend_changed": "Why spending changed", "top_unnecessary": "Top discretionary expenses",
              "weekday_trend": "Weekday pattern", "seasonal_pattern": "Seasonal pattern"}
    for item in ins["insights"]:
        st.markdown(f"**{titles.get(item['topic'], item['topic'])}**  \n{item['text']}")


def tab_chat(svc, run_, cid, month):
    chats = st.session_state.setdefault("chats", {})
    msgs = chats.setdefault(cid, [])
    st.caption("Answers use your own data through tools. Figures that cannot be traced to a tool result are flagged.")
    cols = st.columns(3)
    for i, q in enumerate(SAMPLE_QUESTIONS):
        if cols[i % 3].button(q, key=f"sample_{i}"):
            st.session_state["pending_question"] = q
            st.rerun()
    for m in msgs:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])
            if m.get("tools"):
                st.caption("Tools used: " + ", ".join(m["tools"]))
    typed = st.chat_input("Ask about your spending, savings or health score")
    question = st.session_state.pop("pending_question", None) or typed
    if question:
        msgs.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)
        with st.chat_message("assistant"):
            with st.spinner("Looking at your data..."):
                res = chat_turn(svc, msgs[:-1], question)
            st.markdown(res["answer"])
            tools = [f"{c['name']}" + ("" if c["ok"] else " (error)") for c in res["tool_calls"]]
            st.caption(("Tools used: " + ", ".join(tools) if tools else "No tools needed") + f" | {res['client']}")
        msgs.append({"role": "assistant", "content": res["answer"], "tools": tools})


def tab_lab():
    lab = load_model_lab(ROOT)
    st.write("Evaluation artefacts written by the training and backtest scripts. All figures come from a simulated dataset with injected patterns.")
    sub = st.tabs(["Forecasting", "Overspend detector", "Categoriser", "Health score & planner"])
    with sub[0]:
        st.markdown(lab["forecast"] or "Run `python -m ml.forecasting.compare` to generate `docs/forecast_results.md`.")
    with sub[1]:
        st.markdown(lab["overspend"] or "Run `python -m ml.anomaly.compare` to generate `docs/overspend_results.md`.")
    with sub[2]:
        m = lab["categorizer_metrics"]
        if m:
            rows = {k: v for k, v in m.items() if isinstance(v, dict) and "accuracy" in v}
            table(pd.DataFrame(rows).T.reset_index().rename(columns={"index": "model / slice"}).round(4))
            if lab["confusion_png"]:
                st.image(lab["confusion_png"], caption="Row-normalised confusion matrix (test)")
        else:
            st.info("Run `python -m ml.categorizer.train` to generate the metrics.")
    with sub[3]:
        st.markdown(lab["core"] or "No results file found.")


def tab_import(auth, user: dict) -> None:
    st.subheader("Import a bank statement")
    if user["is_demo"] or not privacy.owns_customer(user["customer_id"]):
        st.info("This account is read-only. Create your own account to import your statements.")
        return
    db, uid, cid = auth.store.db, user["user_id"], user["customer_id"]
    st.markdown("Export a **CSV** statement from your bank's net-banking and upload it here. We keep only the date, a short "
                "description (long account / reference numbers are masked), the amount and the balance. **The file itself is "
                "not stored.** PDF and Excel statements are not supported yet.")
    has_consent = privacy.has_consent(db, uid, privacy.PURPOSE_STATEMENTS)
    agree = st.checkbox("I agree that FinSight processes the transactions in my statements to show me my spending analysis. "
                        "I can withdraw this consent or delete my data any time under **Your data** in the sidebar.",
                        value=has_consent, disabled=has_consent)
    up = st.file_uploader("Statement file (CSV, up to 5 MB)", type=["csv", "txt"])
    opening = st.number_input("Opening balance in Rs (only used when the file has no balance column)", value=0.0, step=1000.0)
    if up is None:
        return
    try:
        parsed = parse_statement(up.getvalue())
    except AuthError as e:
        st.error(str(e))
        return
    income = _declared_income(cid)
    st.success(f"Read {len(parsed.rows):,} transactions from {parsed.rows[0].date:%d %b %Y} to {parsed.rows[-1].date:%d %b %Y}.")
    for w in parsed.warnings:
        st.warning(w)
    st.caption("Columns we used: " + ", ".join(f"{role} = '{name}'" for role, name in parsed.columns.items()))
    preview = pd.DataFrame([{"Date": f"{r.date:%d %b %Y}", "Description": r.description, "Type": r.type, "Amount": money(r.amount),
                             "Balance": money(r.balance) if r.balance is not None else ""} for r in parsed.rows[:10]])
    table(preview)
    cats = preview_categories(parsed, income)
    st.caption("First-pass categories: " + ", ".join(f"{k} {v}" for k, v in cats.items()))
    if cats.get("Uncategorized"):
        st.caption("Rows marked Uncategorized stay out of the discretionary-spending views until the categoriser has run.")
    if st.button(f"Import {len(parsed.rows):,} transactions", type="primary", disabled=not (agree or has_consent)):
        try:
            if not has_consent:
                privacy.record_consent(db, uid, privacy.PURPOSE_STATEMENTS)
            res = import_statement(db, _user_obj(user), parsed, opening_balance=float(opening), monthly_income=income)
        except AuthError as e:
            st.error(str(e))
            return
        except Exception as e:
            st.error(f"The import failed ({type(e).__name__}); nothing was saved. Please try again.")
            return
        st.cache_data.clear()                                # dashboards must not show the pre-import numbers
        st.success(f"Imported {res.inserted:,} transactions ({res.duplicates:,} were already there).")
        for w in res.warnings:
            st.warning(w)
        st.info("Spending views update right away. Forecasts and overspending alerts refresh with the nightly job.")
        if st.button("Open my dashboard"):
            st.rerun()


def privacy_panel(auth, user: dict) -> None:
    """Withdraw consent / delete imported data / delete the account (DPDP-style rights, one click each)."""
    if user["is_demo"]:
        return
    db, uid, cid = auth.store.db, user["user_id"], user["customer_id"]
    with st.expander("Your data"):
        if privacy.has_consent(db, uid, privacy.PURPOSE_STATEMENTS):
            if st.button("Withdraw statement-processing consent"):
                privacy.withdraw_consent(db, uid, privacy.PURPOSE_STATEMENTS)
                st.rerun()
        if privacy.owns_customer(cid):
            sure = st.checkbox("Yes, delete all my imported transactions", key="del_txn_ok")
            if st.button("Delete my transactions", disabled=not sure):
                privacy.delete_transactions(db, cid)
                st.cache_data.clear()
                st.rerun()
        st.divider()
        st.caption("Deleting your account erases your login, profile and all your transactions. This cannot be undone.")
        pw = st.text_input("Current password", type="password", key="del_acct_pw")
        sure2 = st.checkbox("Yes, permanently delete my account and data", key="del_acct_ok")
        if st.button("Delete my account", disabled=not sure2):
            try:
                auth.delete_account(_user_obj(user), pw)
            except AuthError as e:
                st.error(str(e))
            else:
                st.cache_data.clear()
                _logout(auth, rerun=False)
                st.session_state["_goodbye"] = True
                st.rerun()


# ---------------------------------------------------------------------------------------------- main
def main(runner_factory=None, auth_factory=None) -> None:
    global _RUNNER
    st.set_page_config(page_title="Personal Finance Advisor", page_icon=":moneybag:", layout="wide")
    _flush_cookie_commands()
    rls.CURRENT_CUSTOMER.set(None)                         # nothing is scoped until a user is logged in (this run)
    st.title("Personal Finance Advisor")
    try:
        _RUNNER = (runner_factory or _engine_runner)()
        _RUNNER("month_range")                             # fail early, with a helpful message, if the database is unreachable
    except Exception as e:
        st.error(DB_HELP)
        st.caption(f"Technical detail: {str(e) if isinstance(e, RuntimeError) else type(e).__name__}")
        st.stop()
        return
    try:
        auth = (auth_factory or _auth_service)()
    except Exception as e:
        st.error("Login is not available. Check that the auth tables exist (apply sql/migrations/002_auth_extras.sql) and the "
                 "database / mail settings in your environment.")
        st.caption(f"Technical detail: {type(e).__name__}: {e}")
        st.stop()
        return

    if email_link_screen(auth):                            # ?verify=... / ?reset=... links from emails
        st.stop()
        return
    user = _current_user(auth)
    if user is None:
        if st.session_state.pop("_goodbye", False):
            st.success("Your account and data were deleted.")
        login_screen(auth)
        st.stop()
        return

    cid = user["customer_id"]                              # from the login, never from a widget
    rls.CURRENT_CUSTOMER.set(cid)                          # every query from here on is scoped to this customer (RLS, if on)
    ok_svc, svc = safe(AdvisorService, run, cid)
    months = available_months(svc) if ok_svc else []
    if ok_svc and not months:                              # under 6 months of history: still show what exists
        months = available_months(svc, min_history=1)

    with st.sidebar:
        st.header(user["name"])
        st.caption("Shared demo account (synthetic data)" if user["is_demo"] else user["email"])
        if st.button("Log out"):
            _logout(auth)
        if ok_svc and months:
            month = st.selectbox("Month", months, format_func=lambda m: pd.Timestamp(m).strftime("%B %Y"))
            cap = profile_caption(svc.profile)
            if cap:
                st.caption(cap)
        st.caption("Chat mode: " + ("Anthropic model" if os.getenv("ANTHROPIC_API_KEY") else "offline demo (no API key)"))
        privacy_panel(auth, user)

    if not ok_svc or not months:
        st.info("Your account has no transactions yet. Import a bank statement to see your spending analysis.")
        tab_import(auth, user)
        st.stop()
        return

    tabs = st.tabs(["Overview", "Spending", "Health & savings", "Alerts & insights", "Chat", "Import statement", "Model lab"])
    for t, fn in zip(tabs, (tab_overview, tab_spending, tab_health, tab_alerts, tab_chat)):
        with t:
            fn(svc, run, cid, month)
    with tabs[5]:
        tab_import(auth, user)
    with tabs[6]:
        tab_lab()


if __name__ == "__main__":          # Streamlit executes the script as __main__; tests import the module and call main()
    main()
