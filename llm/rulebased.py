"""
Offline "demo mode" client. NOT an LLM: a deterministic router that maps the common budgeting questions
(from the project spec) onto the same tools the real model would call, then writes the answer from the tool results.

It exists so that (a) the app works with no API key or cost, and (b) the whole stack can be tested deterministically.
Anything it does not recognise gets a polite menu of what it can answer.
"""
from __future__ import annotations

import json
import re

from .agent import LLMResponse

MONTHS = {m: i for i, m in enumerate(["january", "february", "march", "april", "may", "june", "july", "august", "september",
                                       "october", "november", "december"], 1)}
MONTH_ABBR = {m[:3]: i for m, i in MONTHS.items()}
HELP = ("I can answer questions about your spending using your own data, for example: 'Where did I spend the most last month?', "
        "'How much did I spend on shopping?', 'Why did I spend more this month?', 'Suggest ways to save a monthly target', "
        "'How much will I spend next month?', 'What is my financial health score?', 'Do I have any overspending alerts?', 'Which weekday do I spend most on?'")


def _rs(x) -> str:
    return f"Rs {float(x):,.0f}"


class RuleBasedClient:
    def __init__(self, as_of: str):
        self.as_of_year, self.as_of_month = int(as_of[:4]), int(as_of[5:7])

    # ------------------------------------------------------------------ parsing helpers
    def _month_in(self, text: str) -> str | None:
        m = re.search(r"\b(20\d{2})-(\d{2})\b", text)
        if m:
            return f"{m.group(1)}-{m.group(2)}"
        m = re.search(r"\b(" + "|".join(MONTHS) + r"|" + "|".join(MONTH_ABBR) + r")\b[\s,]*(20\d{2})?", text)
        if m:
            mo = MONTHS.get(m.group(1)) or MONTH_ABBR[m.group(1)]
            year = int(m.group(2)) if m.group(2) else (self.as_of_year if mo <= self.as_of_month else self.as_of_year - 1)
            return f"{year}-{mo:02d}"
        if "last month" in text:
            return "last_month"
        if "this month" in text:
            return "this_month"
        return None

    def _period_in(self, text: str) -> str:
        m = re.search(r"last (\d+) months", text)
        if m:
            return f"last_{m.group(1)}_months"
        if "last year" in text:
            return str(self.as_of_year - 1)
        if "this year" in text:
            return str(self.as_of_year)
        y = re.search(r"\b(20\d{2})\b", text)
        mo = self._month_in(text)
        if mo and not re.fullmatch(r"20\d{2}", mo):
            return mo
        if y:
            return y.group(1)
        if re.search(r"all time|overall|ever\b", text):
            return "all"
        return "last_month"

    @staticmethod
    def _amount_in(text: str) -> float | None:
        for m in re.finditer(r"(?:rs\.?|₹|inr)?\s?(\d[\d,]*(?:\.\d+)?)\s?(k|lakh|lac|l)?\b", text):
            v = float(m.group(1).replace(",", ""))
            mult = {"k": 1e3, "lakh": 1e5, "lac": 1e5, "l": 1e5}.get(m.group(2) or "", 1)
            if v * mult >= 100 and not re.fullmatch(r"20\d{2}", m.group(1)):
                return v * mult
        return None

    # ------------------------------------------------------------------ routing: question -> one tool call
    def _route(self, q: str) -> tuple[str, dict] | None:
        t = q.lower()
        amt = self._amount_in(t)
        if re.search(r"\bsave|saving", t) and amt:
            return "suggest_savings", {"target_amount": amt}
        if re.search(r"\bwhy\b", t) and re.search(r"more|higher|increase|spent|spend|less|lower|jump|spike", t):
            mo = self._month_in(t)
            return "explain_spending_change", ({"month": mo} if mo else {})
        if re.search(r"next month|forecast|predict|expect to spend|will i spend|going to spend", t):
            return "get_spending_forecast", {}
        if re.search(r"health score|financial health|my score|how healthy", t):
            return "get_health_score", {}
        if re.search(r"alert|overspend|over-spend|warning|unusual", t):
            return "get_alerts", {}
        if "compare" in t:
            found = re.findall(r"\b20\d{2}-\d{2}\b|\b(?:" + "|".join(MONTHS) + r")\b(?:\s+20\d{2})?", t)
            months = [self._month_in(f) for f in found if self._month_in(f)]
            if len(months) >= 2:
                return "compare_months", {"month_a": months[0], "month_b": months[1]}
        if re.search(r"weekday|weekend|day of the week|which day", t):
            return "get_insights", {"topic": "weekday_trend"}
        if re.search(r"season|festival|diwali|christmas|holiday period|time of year", t):
            return "get_insights", {"topic": "seasonal_pattern"}
        if re.search(r"unnecessary|wasteful|cut back|non-essential|discretionary|splurg", t):
            return "get_insights", {"topic": "top_unnecessary"}
        if re.search(r"\binsights?\b", t):
            return "get_insights", {}
        if re.search(r"most|biggest|largest|top|highest", t) and re.search(r"spend|spent|categor|expens", t):
            mo = self._month_in(t)
            return "get_top_categories", ({"month": mo} if mo else {"month": "last_month"})
        m = re.search(r"(?:spend|spent|spending|expenses?)\s+(?:on|for)\s+(.+?)(?:\s+(?:last|this|in|during|over|for|since)\b|[?.!]|$)", t)
        if m:
            return "get_category_total", {"category": m.group(1).strip(), "period": self._period_in(t)}
        return None

    # ------------------------------------------------------------------ composing answers from tool results
    @staticmethod
    def _compose(tool: str, r: dict) -> str:
        if not r.get("ok"):
            return f"I could not get that: {r.get('error', 'unknown problem')}"
        d = r["result"]
        if tool == "get_top_categories":
            rest = ", ".join(f"{c['category']} ({_rs(c['spend'])})" for c in d["top_categories"][1:4])
            return d["text"] + (f" Next: {rest}." if rest else "") + f" Total spending that month was {_rs(d['total_spend'])}."
        if tool == "get_category_total":
            lines = "; ".join(f"{c['category']}: {_rs(c['spend'])} across {c['transactions']} transactions" for c in d["categories"])
            tail = f" Combined: {_rs(d['combined_total'])}." if len(d["categories"]) > 1 else ""
            return f"For {d['period']}: {lines}.{tail}"
        if tool == "compare_months":
            mv = "; ".join(f"{c['category']} ({'+' if c['diff'] > 0 else '-'}{_rs(abs(c['diff']))})" for c in d["biggest_changes"][:3])
            return (f"You spent {_rs(d['total_a'])} in {d['month_a']} and {_rs(d['total_b'])} in {d['month_b']}. "
                    f"Biggest changes: {mv}.")
        if tool == "explain_spending_change":
            return d["story"]
        if tool == "get_alerts":
            if not d["alerts"]:
                return f"No overspending alerts for {d['month']}. {d['note']}"
            return f"Alerts for {d['month']}: " + " ".join(a["message"] for a in d["alerts"][:3])
        if tool == "get_health_score":
            comps = ", ".join(f"{k} {v['points']:.0f}/{v['max_points']}" for k, v in d["components"].items())
            return f"{d['explanation']} Breakdown: {comps}."
        if tool == "suggest_savings":
            txt = d["plan_text"]
            imp = d.get("health_score_impact")
            if imp:
                txt += f"\nFollowing this plan would move your health score from {imp['before']:.0f} to {imp['after']:.0f}."
            return txt
        if tool == "get_spending_forecast":
            return d["text"]
        if tool == "get_insights":
            return " ".join(i["text"] for i in d["insights"])
        return json.dumps(d)[:500]

    # ------------------------------------------------------------------ LLMClient interface
    def create(self, *, system: str, messages: list[dict], tools: list[dict]) -> LLMResponse:
        last = messages[-1]
        if last["role"] == "user" and isinstance(last["content"], str):
            route = self._route(last["content"])
            if route is None:
                return LLMResponse([{"type": "text", "text": HELP}])
            name, args = route
            return LLMResponse([{"type": "tool_use", "id": "rb_1", "name": name, "input": args}], "tool_use")
        use = next(b for b in messages[-2]["content"] if b.get("type") == "tool_use")
        result = json.loads(last["content"][0]["content"])
        return LLMResponse([{"type": "text", "text": self._compose(use["name"], result)}])
