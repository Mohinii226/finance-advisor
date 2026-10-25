"""Chat turn for the Streamlit app: same agent, tools and grounding check as the API, called in-process."""
from __future__ import annotations

from api.handlers import default_client_factory
from core.service import AdvisorService
from llm.agent import AdvisorAgent
from llm.tools import ToolExecutor

SAMPLE_QUESTIONS = [
    "Where did I spend the most last month?",
    "How much did I spend on shopping?",
    "Why did I spend more this month?",
    "Suggest ways to save Rs 10,000 this month.",
    "How much will I spend next month?",
    "What is my financial health score?",
]


def history_for_model(messages: list[dict], max_turns: int = 10) -> list[dict]:
    """Text-only turns for the model (tool details are re-fetched when needed)."""
    return [{"role": m["role"], "content": m["content"]} for m in messages[-2 * max_turns:]]


def chat_turn(svc: AdvisorService, messages: list[dict], question: str) -> dict:
    agent = AdvisorAgent(default_client_factory(svc), ToolExecutor(svc))
    r = agent.chat(question, history_for_model(messages))
    return dict(answer=r.answer, tool_calls=[dict(name=c["name"], input=c["input"], ok=c["ok"]) for c in r.tool_calls],
                grounded=r.grounding.ok, client=type(agent.client).__name__, stopped=r.stopped)
