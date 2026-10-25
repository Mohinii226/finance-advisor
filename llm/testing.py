"""Test doubles for the agent loop."""
from __future__ import annotations

from .agent import LLMResponse


class ScriptedClient:
    """Plays back a fixed list of responses (or calls `policy(messages) -> content blocks`). Records every request."""

    def __init__(self, script=None, policy=None):
        self.script, self.policy, self.calls = list(script or []), policy, []

    def create(self, *, system, messages, tools):
        self.calls.append(dict(system=system, messages=[dict(m) for m in messages], tools=tools))
        if self.policy:
            return LLMResponse(self.policy(messages))
        return LLMResponse(self.script.pop(0) if len(self.script) > 1 else self.script[0])


def tool_use(name: str, args: dict, id_: str = "t1") -> dict:
    return {"type": "tool_use", "id": id_, "name": name, "input": args}


def text(s: str) -> dict:
    return {"type": "text", "text": s}
