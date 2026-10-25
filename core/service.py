"""
AdvisorService: every question the chatbot / API can ask, answered from SQL + the core modules.

* Framework-free and database-agnostic: it only needs `run(query_name, **params) -> DataFrame`
  (see core.insights.make_runner for PostgreSQL; tests use the SQLite harness).
* Scoped to ONE customer, fixed at construction. Nothing here accepts a customer id from a caller of a method.
* "Now" is the latest month in the data (`as_of`), not the wall clock: the dataset is a simulation ending in the past.
* Returns plain JSON-serialisable dicts. Raises ToolError with a message that is safe to show to a model or user.
"""
from __future__ import annotations

import difflib
import math
import re
from decimal import Decimal
from typing import Callable

import numpy as np
import pandas as pd

from . import health_score as hs
from . import insights as ins
from . import savings_planner as sp

# words people use -> candidate categories (a word can map to several)
CATEGORY_ALIASES = {
    "food": ["Food Delivery", "Dining Out", "Groceries"], "eating out": ["Dining Out", "Food Delivery"],
    "restaurants": ["Dining Out"], "restaurant": ["Dining Out"], "delivery": ["Food Delivery"], "swiggy": ["Food Delivery"],
    "zomato": ["Food Delivery"], "grocery": ["Groceries"], "shopping": ["Shopping"], "clothes": ["Shopping"],
    "cab": ["Transport"], "taxi": ["Transport"], "petrol": ["Fuel"], "bills": ["Utilities", "Mobile & Internet"],
    "electricity": ["Utilities"], "phone": ["Mobile & Internet"], "internet": ["Mobile & Internet"],
    "movies": ["Entertainment"], "streaming": ["Subscriptions"], "medical": ["Health"], "doctor": ["Health"],
    "school": ["Education"], "fees": ["Education"], "trips": ["Travel"], "holiday": ["Travel"], "vacation": ["Travel"],
    "gifts": ["Gifts & Donations"], "donations": ["Gifts & Donations"], "cash": ["Cash Withdrawal"], "atm": ["Cash Withdrawal"],
    "loan": ["EMI"], "emi": ["EMI"], "salon": ["Personal Care"], "insurance": ["Insurance"], "rent": ["Rent"],
}


class ToolError(Exception):
    """A user-facing error (bad argument, no data...). The message is safe to return to the model."""


def clean_value(v):
    """Make a value JSON-safe: Decimal/NumPy -> float/int, NaN -> None, Timestamp -> ISO string."""
    if isinstance(v, (Decimal,)):
        v = float(v)
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating, float)):
        return None if (isinstance(v, float) and (math.isnan(v) or math.isinf(v))) or (isinstance(v, np.floating) and not np.isfinite(v)) else round(float(v), 2)
    if isinstance(v, (pd.Timestamp,)):
        return v.strftime("%Y-%m-%d")
    if v is pd.NaT or v is None:
        return None
    if isinstance(v, (np.bool_,)):
        return bool(v)
    return v


def records(df: pd.DataFrame) -> list[dict]:
    return [{k: clean_value(v) for k, v in r.items()} for r in df.to_dict("records")]


def _rs_(x) -> str:
    return f"Rs {float(x):,.0f}"


MONTH_RE = re.compile(r"^(\d{4})-(\d{2})(?:-01)?$")


class AdvisorService:
    def __init__(self, run: Callable[..., pd.DataFrame], customer_id: str):
        self._run = run
        self.customer_id = customer_id
        prof = run("customer_profile", cid=customer_id)
        if prof.empty:
            raise ToolError(f"Unknown customer '{customer_id}'.")
        self.profile = records(prof)[0]
        rng = run("month_range").iloc[0]
        self.first_month, self.last_month = pd.Timestamp(rng["first_month"]), pd.Timestamp(rng["last_month"])
        self._cats: list[str] | None = None

    # ------------------------------------------------------------------ helpers
    def q(self, name: str, **params) -> pd.DataFrame:
        return self._run(name, cid=self.customer_id, **params)

    @property
    def as_of(self) -> str:
        return self.last_month.strftime("%Y-%m-%d")

    @staticmethod
    def _label(m: pd.Timestamp) -> str:
        return f"{ins.MONTHS[m.month - 1]} {m.year}"

    def resolve_month(self, value: str | None) -> pd.Timestamp:
        """'this_month' (default) = latest month in the data, 'last_month', or 'YYYY-MM'."""
        v = (value or "this_month").strip().lower()
        if v in ("this_month", "current", "latest", "now"):
            m = self.last_month
        elif v in ("last_month", "previous", "previous_month"):
            m = self.last_month - pd.DateOffset(months=1)
        elif MONTH_RE.match(v):
            y, mo = MONTH_RE.match(v).groups()
            if not 1 <= int(mo) <= 12:
                raise ToolError(f"'{value}' is not a valid month. Use YYYY-MM, 'this_month' or 'last_month'.")
            m = pd.Timestamp(int(y), int(mo), 1)
        else:
            raise ToolError(f"Could not understand month '{value}'. Use YYYY-MM, 'this_month' or 'last_month'.")
        if not (self.first_month <= m <= self.last_month):
            raise ToolError(f"No data for {self._label(m)}. Data covers {self._label(self.first_month)} to {self._label(self.last_month)}.")
        return m

    def resolve_period(self, value: str | None) -> tuple[pd.Timestamp, pd.Timestamp, str]:
        """-> (start inclusive, end exclusive, label). Supports months, last_N_months, a year, or 'all'."""
        v = (value or "this_month").strip().lower()
        m = re.match(r"^last_(\d+)_months$", v)
        if m:
            n = int(m.group(1))
            if not 1 <= n <= 36:
                raise ToolError("last_N_months supports N from 1 to 36.")
            start = max(self.last_month - pd.DateOffset(months=n - 1), self.first_month)
            return start, self.last_month + pd.DateOffset(months=1), f"the last {n} months ({self._label(start)} to {self._label(self.last_month)})"
        if re.match(r"^\d{4}$", v):
            s, e = pd.Timestamp(int(v), 1, 1), pd.Timestamp(int(v) + 1, 1, 1)
            if e <= self.first_month or s > self.last_month:
                raise ToolError(f"No data for {v}.")
            return max(s, self.first_month), min(e, self.last_month + pd.DateOffset(months=1)), v
        if v == "all":
            return self.first_month, self.last_month + pd.DateOffset(months=1), "all available data"
        mo = self.resolve_month(v)
        return mo, mo + pd.DateOffset(months=1), self._label(mo)

    def categories(self) -> list[str]:
        if self._cats is None:
            self._cats = list(self._run("expense_categories")["category"]) + ["Uncategorized"]
        return self._cats

    def resolve_categories(self, name: str) -> list[str]:
        n = (name or "").strip().lower()
        if not n:
            raise ToolError("A category name is required.")
        exact = [c for c in self.categories() if c.lower() == n]
        if exact:
            return exact
        if n in CATEGORY_ALIASES:
            return [c for c in CATEGORY_ALIASES[n] if c in self.categories()]
        partial = [c for c in self.categories() if n in c.lower()]
        if partial:
            return partial
        close = difflib.get_close_matches(n, [c.lower() for c in self.categories()], n=2, cutoff=0.6)
        if close:
            return [c for c in self.categories() if c.lower() in close]
        raise ToolError(f"I do not recognise the category '{name}'. Known categories: {', '.join(self.categories())}.")

    # ------------------------------------------------------------------ tools
    def get_top_categories(self, month: str | None = None, limit: int = 5) -> dict:
        m = self.resolve_month(month)
        df = self.q("top_categories_month", month=m.strftime("%Y-%m-%d"))
        if df.empty:
            raise ToolError(f"No spending recorded in {self._label(m)}.")
        total = float(df["spend"].astype(float).sum())
        top = records(df.head(limit))
        return dict(as_of=self.as_of, month=self._label(m), total_spend=round(total, 2), top_categories=top,
                    text=ins.largest_category(df, self._label(m)))

    def get_category_total(self, category: str, period: str | None = "last_month") -> dict:
        cats = self.resolve_categories(category)
        start, end, label = self.resolve_period(period)
        rows, total = [], 0.0
        for c in cats:
            df = self.q("category_total", category=c, start=start.strftime("%Y-%m-%d"), end=end.strftime("%Y-%m-%d"))
            spend, txns = (float(df["spend"].iloc[0]), int(df["txns"].iloc[0])) if not df.empty else (0.0, 0)
            rows.append(dict(category=c, spend=round(spend, 2), transactions=txns)); total += spend
        out = dict(as_of=self.as_of, period=label, categories=rows, combined_total=round(total, 2))
        if len(cats) > 1:
            out["note"] = f"'{category}' maps to several categories; totals are listed separately and combined."
        return out

    def compare_months(self, month_a: str, month_b: str) -> dict:
        a, b = self.resolve_month(month_a), self.resolve_month(month_b)
        df = self.q("month_compare", m1=a.strftime("%Y-%m-%d"), m2=b.strftime("%Y-%m-%d"))
        if df.empty:
            raise ToolError("No spending found in either month.")
        ta, tb = float(df["m1_spend"].astype(float).sum()), float(df["m2_spend"].astype(float).sum())
        return dict(as_of=self.as_of, month_a=self._label(a), month_b=self._label(b), total_a=round(ta, 2), total_b=round(tb, 2),
                    difference_b_minus_a=round(tb - ta, 2), biggest_changes=records(df.head(6)))

    def explain_spending_change(self, month: str | None = None) -> dict:
        m = self.resolve_month(month)
        mstr = m.strftime("%Y-%m-%d")
        hist = self.q("category_history", month=mstr, start=(m - pd.DateOffset(months=6)).strftime("%Y-%m-%d"))
        drivers = ins.robust_drivers(hist, mstr)
        if drivers.empty:
            raise ToolError(f"Not enough history to compare {self._label(m)} with your usual spending (needs at least 3 earlier months).")
        story = ins.spend_story(self._label(m), drivers, lambda c: self.q("largest_txns_in_category", month=mstr, category=c, limit=3))
        d = drivers.assign(spend=lambda x: x["spend"].astype(float), baseline_mean=lambda x: x["baseline_mean"].astype(float))
        return dict(as_of=self.as_of, month=self._label(m), total_spend=round(float(d.spend.sum()), 2),
                    usual_spend=round(float(d.baseline_mean.sum()), 2),
                    difference=round(float(d.spend.sum() - d.baseline_mean.sum()), 2),
                    drivers=records(drivers.head(6)), story=story)

    def get_alerts(self, month: str | None = None) -> dict:
        m = self.resolve_month(month)
        try:
            df = self.q("alerts_table_for_month", month=m.strftime("%Y-%m-%d"))
        except Exception as e:  # the alerts table may not exist / be empty if the detector was never run
            raise ToolError("Overspending alerts are not available yet (the detector has not been run).") from e
        return dict(as_of=self.as_of, month=self._label(m), alert_count=len(df), alerts=records(df),
                    note="Alerts come from a conservative detector that favours fewer, higher-confidence alerts; no alert does not prove nothing changed.")

    def _features(self, m: pd.Timestamp) -> pd.Series:
        f = self.q("health_inputs", month=m.strftime("%Y-%m-%d"))
        if f.empty or int(f["n6"].iloc[0]) < 6:
            raise ToolError("A health score needs six full months of history, which this month does not have yet.")
        row = f.iloc[0].copy()
        for c in row.index.difference(["customer_id", "month"]):
            row[c] = pd.to_numeric(row[c], errors="coerce")
        row["month"] = pd.Timestamp(row["month"])
        return row

    def get_health_score(self, month: str | None = None) -> dict:
        m = self.resolve_month(month)
        row = self._features(m)
        s = hs.score(pd.DataFrame([row]), pd.Series({self.customer_id: self.profile["income_type"]})).iloc[0]
        comps = {hs.COMPONENTS[k][3]: dict(points=float(s[k]), max_points=hs.COMPONENTS[k][0]) for k in hs.COMPONENTS}
        return dict(as_of=self.as_of, month=self._label(m), total=float(s["total"]), band=s["band"], components=comps,
                    biggest_drag=hs.COMPONENTS[s["top_drag"]][3], explanation=s["explanation"])

    def suggest_savings(self, target_amount: float) -> dict:
        income = float(self.profile["monthly_income"])
        if not (target_amount > 0):
            raise ToolError("The savings target must be a positive amount in rupees.")
        if target_amount > income:
            raise ToolError(f"A target of Rs {target_amount:,.0f} a month is more than your monthly income (about Rs {income:,.0f}).")
        cands = self.q("savings_candidates", month=self.as_of)
        plan = sp.plan(cands, float(target_amount))
        out = dict(as_of=self.as_of, target=plan["target"], achievable=plan["feasible"], planned_saving=round(plan["achieved"], 2),
                   shortfall=round(plan["shortfall"], 2), plan_text=sp.format_plan(plan),
                   cuts=[{k: clean_value(v) for k, v in i.items() if k in ("category", "current", "cut", "new_level", "pct_cut", "annual_saving")}
                         for i in sorted(plan["items"], key=lambda i: -i["cut"])])
        try:
            out["health_score_impact"] = {k: clean_value(v) for k, v in
                                          sp.score_impact(plan, self._features(self.last_month), self.profile["income_type"]).items()}
        except ToolError:
            pass
        return out

    def get_insights(self, topic: str | None = None, month: str | None = None) -> dict:
        m = self.resolve_month(month)
        out = ins.generate_insights(lambda n, **kw: self._run(n, **kw), self.customer_id, m.strftime("%Y-%m-%d"))
        if topic:
            out = [o for o in out if o["topic"] == topic]
            if not out:
                raise ToolError(f"No insight available for topic '{topic}'.")
        return dict(as_of=self.as_of, month=self._label(m), insights=out)

    def get_spending_forecast(self) -> dict:
        """Next month's expected total spending with a likely range, from the batch forecasting job."""
        try:
            df = self.q("forecast_for_customer")
        except Exception as e:
            raise ToolError("Spending forecasts are not available yet (the forecasting job has not been run).") from e
        if df.empty:
            raise ToolError("There is no forecast for you yet.")
        df = df.assign(target_month=pd.to_datetime(df["target_month"]))
        latest = df[df["target_month"] == df["target_month"].max()]
        primary = latest[latest["model_name"] == "gbm_l1"]
        row = (primary if len(primary) else latest).iloc[0]
        hist = self.q("monthly_expense_history", month=self.as_of)
        recent = [float(x) for x in hist["expenses"]]
        pred = float(row["predicted_amount"])
        usual = float(np.median(recent)) if recent else None
        out = dict(as_of=self.as_of, forecast_month=self._label(row["target_month"]), expected_spend=round(pred, 2),
                   likely_range_low=clean_value(row["lower_bound"]), likely_range_high=clean_value(row["upper_bound"]),
                   range_meaning="about 80% of past months fell inside a range built the same way", model=row["model_name"],
                   usual_monthly_spend_last_6_months=round(usual, 2) if usual is not None else None)
        text = f"I expect you to spend about {_rs_(pred)} in {out['forecast_month']}"
        if out["likely_range_low"] is not None and out["likely_range_high"] is not None:
            text += f", most likely between {_rs_(out['likely_range_low'])} and {_rs_(out['likely_range_high'])}"
        if usual:
            diff = (pred / usual - 1) * 100
            text += (f". Your usual month over the last 6 is {_rs_(usual)}, so that is "
                     f"{'about the same' if abs(diff) < 5 else f'{abs(diff):.0f}% ' + ('higher' if diff > 0 else 'lower')}.")
        out["text"] = text
        return out
