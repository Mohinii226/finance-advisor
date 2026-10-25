"""
Tool-calling agent loop, independent of any LLM vendor.

An `LLMClient` is anything with
    create(*, system, messages, tools) -> LLMResponse(content=[{type:'text'|'tool_use', ...}], stop_reason)
so the loop can be tested with scripted fake models, run offline with the rule-based client, or pointed at the
Anthropic API (llm/clients.py).
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Protocol

from . import grounding
from .prompts import SYSTEM_PROMPT
from .tools import TOOLS, ToolExecutor

MAX_PARALLEL_TOOLS = 4


@dataclass
class LLMResponse:
    content: list[dict]
    stop_reason: str = "end_turn"


class LLMClient(Protocol):
    def create(self, *, system: str, messages: list[dict], tools: list[dict]) -> LLMResponse: ...


@dataclass
class ChatResult:
    answer: str
    tool_calls: list[dict] = field(default_factory=list)       # [{name, input, ok}]
    grounding: grounding.GroundingReport = field(default_factory=grounding.GroundingReport)
    steps: int = 0
    stopped: str = "answered"                                   # answered | max_steps | empty


class AdvisorAgent:
    def __init__(self, client: LLMClient, executor: ToolExecutor, max_steps: int = 6, annotate_ungrounded: bool = True):
        self.client, self.executor, self.max_steps, self.annotate = client, executor, max_steps, annotate_ungrounded

    def chat(self, user_message: str, history: list[dict] | None = None) -> ChatResult:
        """history: earlier turns as [{'role': 'user'|'assistant', 'content': '<text>'}] (text only)."""
        messages = list(history or []) + [{"role": "user", "content": user_message}]
        calls: list[dict] = []
        ok_results: list = []
        answer, stopped = "", "max_steps"
        for step in range(1, self.max_steps + 1):
            resp = self.client.create(system=SYSTEM_PROMPT, messages=messages, tools=TOOLS)
            messages.append({"role": "assistant", "content": resp.content})
            uses = [b for b in resp.content if b.get("type") == "tool_use"]
            if not uses:
                answer = "\n".join(b.get("text", "") for b in resp.content if b.get("type") == "text").strip()
                stopped = "answered" if answer else "empty"
                break
            results = []
            for i, u in enumerate(uses):
                if i >= MAX_PARALLEL_TOOLS:
                    r = {"ok": False, "error": f"Too many tool calls at once (limit {MAX_PARALLEL_TOOLS}). Ask for fewer."}
                else:
                    r = self.executor.execute(u.get("name", ""), u.get("input"))
                calls.append({"name": u.get("name"), "input": u.get("input"), "ok": r["ok"]})
                if r["ok"]:
                    ok_results.append(r["result"])
                results.append({"type": "tool_result", "tool_use_id": u.get("id"), "content": json.dumps(r, default=str),
                                "is_error": not r["ok"]})
            messages.append({"role": "user", "content": results})
        else:
            answer = "I could not finish working that out. Could you rephrase or ask a narrower question?"

        report = grounding.check(answer, ok_results, user_message)
        if self.annotate and not report.ok:
            answer += grounding.caution(report)
        return ChatResult(answer=answer, tool_calls=calls, grounding=report, steps=step, stopped=stopped)
