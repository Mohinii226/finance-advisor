"""Vendor adapter. UNTESTED in my sandbox (no network, no SDK installed); the agent loop is tested with scripted clients."""
from __future__ import annotations

import os

from .agent import LLMResponse

DEFAULT_MODEL = "claude-haiku-4-5-20251001"      # small, fast, supports tool use; override with ADVISOR_MODEL


class AnthropicClient:
    def __init__(self, model: str | None = None, max_tokens: int = 1024, api_key: str | None = None):
        import anthropic                                   # imported lazily: optional dependency
        self.client = anthropic.Anthropic(api_key=api_key or os.getenv("ANTHROPIC_API_KEY"))
        self.model, self.max_tokens = model or os.getenv("ADVISOR_MODEL", DEFAULT_MODEL), max_tokens

    def create(self, *, system: str, messages: list[dict], tools: list[dict]) -> LLMResponse:
        resp = self.client.messages.create(model=self.model, max_tokens=self.max_tokens, system=system,
                                           tools=tools, messages=messages)
        blocks = []
        for b in resp.content:
            if b.type == "text":
                blocks.append({"type": "text", "text": b.text})
            elif b.type == "tool_use":
                blocks.append({"type": "tool_use", "id": b.id, "name": b.name, "input": dict(b.input)})
        return LLMResponse(content=blocks, stop_reason=resp.stop_reason)
