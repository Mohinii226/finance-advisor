"""
Tool definitions (Anthropic tool-use format) + a validating executor.

SECURITY MODEL
  * The customer id is NOT a tool argument. It is bound to the AdvisorService at session creation, so a model (or a
    prompt injection) cannot ask for another customer's data.
  * Arguments are validated against the schema before anything runs: unknown keys, wrong types, out-of-range
    numbers and over-long strings are rejected with a message the model can act on.
  * Tool results are sanitised (control characters stripped, strings and lists capped) before they go back to the model.
"""
from __future__ import annotations

import json
import re

from core.service import AdvisorService, ToolError

MONTH_DESC = "'this_month' (latest month in the data, default), 'last_month', or 'YYYY-MM'"
PERIOD_DESC = ("'this_month', 'last_month', 'last_3_months', 'last_6_months', 'last_12_months' (any last_N_months, N up to 36), "
               "a calendar year like '2025', a month 'YYYY-MM', or 'all'")

TOOLS: list[dict] = [
    {"name": "get_top_categories",
     "description": "Spending by category for one month, biggest first, with totals and share of spending. Use for 'where did I spend the most'.",
     "input_schema": {"type": "object", "additionalProperties": False, "properties": {
         "month": {"type": "string", "description": MONTH_DESC},
         "limit": {"type": "integer", "minimum": 1, "maximum": 10, "description": "How many categories to return (default 5)"}}}},
    {"name": "get_category_total",
     "description": "Total spent in a category (or category group such as 'food') over a period. Use for 'how much did I spend on shopping'.",
     "input_schema": {"type": "object", "additionalProperties": False, "required": ["category"], "properties": {
         "category": {"type": "string", "description": "Category or everyday word, e.g. 'Shopping', 'food', 'cab', 'rent'"},
         "period": {"type": "string", "description": PERIOD_DESC + ". Default 'last_month'."}}}},
    {"name": "compare_months",
     "description": "Compare total and per-category spending between two months, with the biggest changes.",
     "input_schema": {"type": "object", "additionalProperties": False, "required": ["month_a", "month_b"], "properties": {
         "month_a": {"type": "string", "description": MONTH_DESC}, "month_b": {"type": "string", "description": MONTH_DESC}}}},
    {"name": "explain_spending_change",
     "description": "Explain why spending in a month was higher or lower than the customer's usual (trailing 6-month) level: which habits changed, "
                    "one-off purchases, and fixed or periodic bills. Use for 'why did I spend more this month'.",
     "input_schema": {"type": "object", "additionalProperties": False, "properties": {"month": {"type": "string", "description": MONTH_DESC}}}},
    {"name": "get_alerts",
     "description": "Overspending alerts raised by the detector for a month (sustained category overspend or one-off spikes).",
     "input_schema": {"type": "object", "additionalProperties": False, "properties": {"month": {"type": "string", "description": MONTH_DESC}}}},
    {"name": "get_health_score",
     "description": "Financial health score 0-100 with component points (savings rate, debt ratio, emergency fund, expense consistency, income stability) and the biggest drag.",
     "input_schema": {"type": "object", "additionalProperties": False, "properties": {"month": {"type": "string", "description": MONTH_DESC}}}},
    {"name": "suggest_savings",
     "description": "Build the lowest-impact plan to save a target amount per month by trimming flexible categories; also shows the effect on the health score. "
                    "Use for 'suggest ways to save Rs 10,000 this month'.",
     "input_schema": {"type": "object", "additionalProperties": False, "required": ["target_amount"], "properties": {
         "target_amount": {"type": "number", "minimum": 1, "maximum": 10000000, "description": "Rupees to save per month"}}}},
    {"name": "get_spending_forecast",
     "description": "Expected total spending for NEXT month with a likely range, compared with the customer's usual month. Use for 'how much will I spend next month'.",
     "input_schema": {"type": "object", "additionalProperties": False, "properties": {}}},
    {"name": "get_insights",
     "description": "Explainable insights: largest category, top discretionary merchants, weekday pattern, seasonal pattern, or why spending changed.",
     "input_schema": {"type": "object", "additionalProperties": False, "properties": {
         "topic": {"type": "string", "enum": ["largest_category", "why_spend_changed", "top_unnecessary", "weekday_trend", "seasonal_pattern"],
                   "description": "Omit to get all insights"},
         "month": {"type": "string", "description": MONTH_DESC}}}},
]
TOOL_SCHEMAS = {t["name"]: t["input_schema"] for t in TOOLS}

MAX_STR, MAX_LIST, MAX_RESULT_CHARS = 1500, 8, 9000
_CTRL = re.compile(r"[\x00-\x08\x0b-\x1f\x7f]")


def validate_args(schema: dict, args) -> dict:
    """Minimal JSON-schema validation (object with typed properties, enum, min/max, required, no extra keys)."""
    if args is None:
        args = {}
    if not isinstance(args, dict):
        raise ToolError("Tool arguments must be a JSON object.")
    props = schema.get("properties", {})
    extra = set(args) - set(props)
    if extra and not schema.get("additionalProperties", True):
        raise ToolError(f"Unknown argument(s): {', '.join(sorted(extra))}. Allowed: {', '.join(props) or 'none'}.")
    missing = [k for k in schema.get("required", []) if k not in args or args[k] in (None, "")]
    if missing:
        raise ToolError(f"Missing required argument(s): {', '.join(missing)}.")
    out = {}
    for k, v in args.items():
        if v is None:
            continue
        spec, t = props[k], props[k].get("type")
        if t == "string":
            if not isinstance(v, str):
                raise ToolError(f"'{k}' must be a string.")
            v = _CTRL.sub("", v).strip()
            if len(v) > 100:
                raise ToolError(f"'{k}' is too long.")
            if "enum" in spec and v not in spec["enum"]:
                raise ToolError(f"'{k}' must be one of: {', '.join(spec['enum'])}.")
        elif t in ("integer", "number"):
            if isinstance(v, bool) or not isinstance(v, (int, float)) or (t == "integer" and isinstance(v, float) and not v.is_integer()):
                raise ToolError(f"'{k}' must be a {t}.")
            if "minimum" in spec and v < spec["minimum"] or "maximum" in spec and v > spec["maximum"]:
                raise ToolError(f"'{k}' must be between {spec.get('minimum')} and {spec.get('maximum')}.")
            v = int(v) if t == "integer" else float(v)
        out[k] = v
    return out


def sanitize(obj):
    """Strip control characters, cap string length and list length (marks truncation)."""
    if isinstance(obj, str):
        s = _CTRL.sub(" ", obj)
        return s if len(s) <= MAX_STR else s[:MAX_STR] + "..."
    if isinstance(obj, list):
        items = [sanitize(x) for x in obj[:MAX_LIST]]
        return items + [f"... {len(obj) - MAX_LIST} more omitted"] if len(obj) > MAX_LIST else items
    if isinstance(obj, dict):
        return {str(k): sanitize(v) for k, v in obj.items()}
    return obj


class ToolExecutor:
    """Runs tools for ONE customer (bound through the service). Never raises: returns {'ok': bool, ...}."""

    def __init__(self, service: AdvisorService):
        self.service = service
        self._dispatch = {
            "get_top_categories": service.get_top_categories, "get_category_total": service.get_category_total,
            "compare_months": service.compare_months, "explain_spending_change": service.explain_spending_change,
            "get_alerts": service.get_alerts, "get_health_score": service.get_health_score,
            "suggest_savings": service.suggest_savings, "get_insights": service.get_insights,
            "get_spending_forecast": service.get_spending_forecast,
        }

    def execute(self, name: str, args) -> dict:
        if name not in self._dispatch:
            return {"ok": False, "error": f"Unknown tool '{name}'. Available tools: {', '.join(self._dispatch)}."}
        try:
            clean = validate_args(TOOL_SCHEMAS[name], args)
            result = sanitize(self._dispatch[name](**clean))
            if len(json.dumps(result, default=str)) > MAX_RESULT_CHARS:
                result = {"truncated": True, "text": json.dumps(result, default=str)[:MAX_RESULT_CHARS]}
            return {"ok": True, "result": result}
        except ToolError as e:
            return {"ok": False, "error": str(e)}
        except Exception as e:      # a bug or a database problem must not crash the conversation
            return {"ok": False, "error": f"The data service had a problem ({type(e).__name__}). Please try again."}
