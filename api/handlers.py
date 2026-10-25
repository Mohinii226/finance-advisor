"""
Framework-free request handlers. All behaviour worth testing lives here; api/main.py only maps HTTP onto these.

DEMO-LEVEL SECURITY (read before deploying)
  * Auth is a single shared API key (X-API-Key). It does NOT identify a customer, so any key holder can query any
    customer id. A real deployment must derive the customer from an authenticated identity (JWT / session) instead of the URL or body.
  * Chat sessions are in memory (lost on restart, not shared between workers) and bound to the customer that created them.
  * No rate limiting.
"""
from __future__ import annotations

import hmac
import os
import re
import uuid
from collections import OrderedDict
from typing import Callable

import pandas as pd

from core.service import AdvisorService, ToolError
from llm.agent import AdvisorAgent
from llm.rulebased import RuleBasedClient
from llm.tools import ToolExecutor

CUSTOMER_ID_RE = re.compile(r"^[A-Za-z0-9_-]{1,20}$")


class ApiError(Exception):
    status = 400


class BadRequest(ApiError):
    status = 400


class Unauthorized(ApiError):
    status = 401


class NotFound(ApiError):
    status = 404


def check_api_key(provided: str | None, expected: str | None) -> None:
    """If an expected key is configured, require it (constant-time compare). Unset = open demo mode."""
    if expected and not hmac.compare_digest((provided or "").encode(), expected.encode()):
        raise Unauthorized("Missing or invalid API key.")


class ChatSessions:
    """In-memory, bounded, per-customer chat history (text turns only)."""

    def __init__(self, max_sessions: int = 500, max_messages: int = 20):
        self._s: OrderedDict[str, dict] = OrderedDict()
        self.max_sessions, self.max_messages = max_sessions, max_messages

    def history(self, session_id: str, customer_id: str) -> list[dict]:
        s = self._s.get(session_id)
        if s is None:
            return []
        if s["customer_id"] != customer_id:
            raise NotFound("Unknown session.")           # same message as a missing session: no oracle
        return list(s["messages"])

    def append(self, session_id: str, customer_id: str, user: str, assistant: str) -> None:
        s = self._s.setdefault(session_id, {"customer_id": customer_id, "messages": []})
        if s["customer_id"] != customer_id:
            raise NotFound("Unknown session.")
        s["messages"] = (s["messages"] + [{"role": "user", "content": user}, {"role": "assistant", "content": assistant}])[-self.max_messages:]
        self._s.move_to_end(session_id)
        while len(self._s) > self.max_sessions:
            self._s.popitem(last=False)


def default_client_factory(service: AdvisorService):
    """Anthropic if a key is configured and the SDK is installed, else the offline rule-based demo client."""
    if os.getenv("ANTHROPIC_API_KEY"):
        try:
            from llm.clients import AnthropicClient
            return AnthropicClient()
        except Exception:
            pass
    return RuleBasedClient(service.as_of)


class Backend:
    def __init__(self, runner: Callable[..., pd.DataFrame], client_factory=default_client_factory, sessions: ChatSessions | None = None):
        self.runner, self.client_factory = runner, client_factory
        self.sessions = sessions or ChatSessions()

    # ------------------------------------------------------------------ plumbing
    def service(self, customer_id: str) -> AdvisorService:
        if not CUSTOMER_ID_RE.match(customer_id or ""):
            raise BadRequest("Invalid customer id.")
        try:
            return AdvisorService(self.runner, customer_id)
        except ToolError as e:
            raise NotFound(str(e)) from e

    @staticmethod
    def _wrap(fn, *a, **kw):
        try:
            return fn(*a, **kw)
        except ToolError as e:
            raise BadRequest(str(e)) from e

    # ------------------------------------------------------------------ endpoints
    def summary(self, cid: str, month: str | None = None) -> dict:
        s = self.service(cid)
        out = {"customer": {k: s.profile[k] for k in ("customer_id", "name", "city", "occupation", "income_type")},
               "as_of": s.as_of, "top_categories": self._wrap(s.get_top_categories, month, 3)}
        for key, fn in (("health_score", s.get_health_score), ("alerts", s.get_alerts)):
            try:
                out[key] = fn(month)
            except ToolError as e:                       # e.g. not enough history: report, do not fail the whole summary
                out[key] = {"unavailable": str(e)}
        return out

    def categories(self, cid: str, month: str | None, limit: int = 10) -> dict:
        return self._wrap(self.service(cid).get_top_categories, month, max(1, min(limit, 10)))

    def alerts(self, cid: str, month: str | None) -> dict:
        return self._wrap(self.service(cid).get_alerts, month)

    def health_score(self, cid: str, month: str | None) -> dict:
        return self._wrap(self.service(cid).get_health_score, month)

    def savings_plan(self, cid: str, target_amount: float) -> dict:
        return self._wrap(self.service(cid).suggest_savings, float(target_amount))

    def insights(self, cid: str, month: str | None, topic: str | None) -> dict:
        return self._wrap(self.service(cid).get_insights, topic, month)

    def chat(self, cid: str, message: str, session_id: str | None = None) -> dict:
        message = (message or "").strip()
        if not message or len(message) > 1000:
            raise BadRequest("Message must be between 1 and 1000 characters.")
        service = self.service(cid)
        session_id = session_id or uuid.uuid4().hex
        history = self.sessions.history(session_id, cid)
        agent = AdvisorAgent(self.client_factory(service), ToolExecutor(service))
        result = agent.chat(message, history)
        self.sessions.append(session_id, cid, message, result.answer)
        g = result.grounding
        return {"session_id": session_id, "answer": result.answer, "stopped": result.stopped,
                "tool_calls": [{"name": c["name"], "input": c["input"], "ok": c["ok"]} for c in result.tool_calls],
                "grounding": {"ok": g.ok, "unverified_amounts": g.ungrounded_amounts, "unverified_percents": g.ungrounded_percents},
                "client": type(agent.client).__name__}
