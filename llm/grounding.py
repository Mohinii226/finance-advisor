"""
Grounding check: are the rupee amounts and percentages in an answer supported by tool results?

This catches the most damaging chatbot failure in a finance product: a confident, invented number.
It is a *heuristic*:
  * only rupee amounts (Rs / ₹ / INR) and percentages are checked, not counts, dates or years;
  * an amount is grounded if it matches a number from a tool result (or from the user's own message) within 2.5%,
    or matches 12x a tool number (annualised savings), ignoring sign;
  * arithmetic the model did itself (sums, averages) is reported as "unverified", not necessarily wrong.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

AMOUNT_RE = re.compile(r"(?:Rs\.?|₹|INR)\s?([0-9][0-9,]*(?:\.[0-9]+)?)", re.I)
PCT_RE = re.compile(r"([0-9]+(?:\.[0-9]+)?)\s?%")
REL_TOL, ABS_TOL, PCT_TOL = 0.025, 1.0, 0.6


@dataclass
class GroundingReport:
    ungrounded_amounts: list[float] = field(default_factory=list)
    ungrounded_percents: list[float] = field(default_factory=list)
    checked: int = 0

    @property
    def ok(self) -> bool:
        return not self.ungrounded_amounts and not self.ungrounded_percents


def _to_float(s: str) -> float:
    return float(s.replace(",", ""))


def numbers_in_text(text: str) -> tuple[list[float], list[float]]:
    return [_to_float(m) for m in AMOUNT_RE.findall(text)], [float(m) for m in PCT_RE.findall(text)]


def collect_numbers(obj) -> list[float]:
    """All numeric leaves of a JSON-like object, plus amounts / percents quoted inside its strings."""
    out: list[float] = []
    if isinstance(obj, bool) or obj is None:
        return out
    if isinstance(obj, (int, float)):
        return [float(obj)]
    if isinstance(obj, str):
        a, p = numbers_in_text(obj)
        return a + p
    if isinstance(obj, dict):
        for v in obj.values():
            out += collect_numbers(v)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            out += collect_numbers(v)
    return out


def check(answer: str, tool_results: list, user_message: str = "") -> GroundingReport:
    nums = [abs(x) for r in tool_results for x in collect_numbers(r)]
    a_user, p_user = numbers_in_text(user_message)
    nums += a_user + p_user
    amounts, pcts = numbers_in_text(answer)
    rep = GroundingReport(checked=len(amounts) + len(pcts))

    def amount_ok(a: float) -> bool:
        return any(abs(a - t) <= max(ABS_TOL, REL_TOL * t) or abs(a - 12 * t) <= max(ABS_TOL, REL_TOL * 12 * t) for t in nums)

    rep.ungrounded_amounts = [a for a in amounts if not amount_ok(a)]
    rep.ungrounded_percents = [p for p in pcts if not any(abs(p - t) <= PCT_TOL for t in nums)]
    return rep


def caution(rep: GroundingReport) -> str:
    bits = [f"Rs {a:,.0f}" for a in rep.ungrounded_amounts] + [f"{p:g}%" for p in rep.ungrounded_percents]
    return ("\n\n(Note: I could not verify these figures directly from your data: " + ", ".join(bits) +
            ". They may be rounded or calculated by me, so please double-check them.)")
