"""
Customer profiles + injected ground-truth events.

A profile fixes *who the customer is* (income, city tier, habits). The
transaction engine then turns a profile into months of realistic behaviour.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import config as C

try:  # Faker is the intended dependency; fall back to a tiny list if absent
    from faker import Faker
    _fake = Faker("en_IN")
except Exception:  # pragma: no cover
    _fake = None

_FALLBACK_FIRST = ["Aarav", "Vivaan", "Aditya", "Rohan", "Karan", "Neha", "Priya", "Ananya", "Sneha", "Pooja",
                   "Rahul", "Amit", "Sanjay", "Kavita", "Meera", "Rajesh", "Sunita", "Vikram", "Divya", "Arjun"]
_FALLBACK_LAST = ["Sharma", "Patil", "Iyer", "Reddy", "Gupta", "Nair", "Joshi", "Kulkarni", "Singh", "Mehta",
                  "Das", "Khan", "Menon", "Verma", "Desai", "Rao", "Banerjee", "Shah", "Pillai", "Chavan"]


def random_name(rng: np.random.Generator) -> str:
    if _fake is not None:
        _fake.seed_instance(int(rng.integers(0, 2**31 - 1)))
        return _fake.name()
    return f"{rng.choice(_FALLBACK_FIRST)} {rng.choice(_FALLBACK_LAST)}"


@dataclass
class Customer:
    customer_id: str
    name: str
    persona: str
    age: int
    city: str
    city_tier: int
    occupation: str
    income_type: str
    base_income: float          # monthly take-home at start of window
    dependents: int
    has_loan: bool
    emi_amount: float
    sip_amount: float
    owns_home: bool
    has_vehicle: bool
    primary_bank: str
    account_last4: str
    volatility: float
    savings_rate: float
    opening_balance: float
    opening_reserve: float      # emergency fund / FD (liquid savings outside the main account)
    employer: str
    favourite_merchants: dict = field(default_factory=dict)   # category -> (names, probs)
    cat_weights: dict = field(default_factory=dict)           # category -> share of income


def _loguniform(rng, lo, hi):
    return float(np.exp(rng.uniform(np.log(lo), np.log(hi))))


def _round_income(x: float) -> float:
    return float(round(x / 500) * 500)


def _pick_favourites(rng) -> dict:
    """Each customer has a few 'go-to' merchants per category (brand loyalty)."""
    fav = {}
    for cat, merchants in C.MERCHANTS.items():
        names = [m for m, _ in merchants]
        pop = np.array([p for _, p in merchants], dtype=float)
        # Dirichlet around popularity -> every customer has a different loyalty pattern
        probs = rng.dirichlet(pop * 1.2)
        fav[cat] = (names, probs)
    return fav


def build_customer(idx: int, rng: np.random.Generator) -> Customer:
    names = list(C.ARCHETYPES)
    probs = np.array([C.ARCHETYPES[n]["prob"] for n in names])
    persona = str(rng.choice(names, p=probs / probs.sum()))
    a = C.ARCHETYPES[persona]

    tier = int(rng.choice([1, 2, 3], p=[0.50, 0.35, 0.15]))
    income = _round_income(_loguniform(rng, *a["income"]) * (1.08 if tier == 1 else 1.0 if tier == 2 else 0.9))
    owns_home = bool(rng.random() < a["owns_home"])
    has_loan = bool(rng.random() < a["loan_prob"])
    emi = _round_income(income * rng.uniform(0.10, 0.35)) if has_loan else 0.0
    sip = float(round(income * rng.uniform(*a["sip"]) / 500) * 500)
    savings_rate = float(rng.uniform(*a["savings_rate"]))

    # Category weights = baseline * persona multiplier * small individual noise
    weights = {}
    has_vehicle = bool(rng.random() < a["vehicle"])
    for cat, spec in C.CATEGORIES.items():
        w = spec["weight"] * a["mult"].get(cat, 1.0) * float(rng.lognormal(0, 0.25))
        if cat == "Rent":
            w = 0.0 if owns_home else w * C.RENT_TIER_MULT[tier]
        if cat == "Fuel" and not has_vehicle:
            w = 0.0
        if cat == "Education" and a["dependents"][1] == 0 and persona != "student_parttime":
            w *= 0.3
        weights[cat] = w

    # Spendable share = 1 - savings - EMI - SIP; split across categories by weight
    spend_ratio = max(0.25, 1.0 - savings_rate - emi / income - sip / income)
    total_w = sum(weights.values())
    cat_weights = {c: (w / total_w) * spend_ratio for c, w in weights.items()}

    monthly_spend = income * spend_ratio
    return Customer(
        customer_id=f"C{idx:04d}",
        name=random_name(rng),
        persona=persona,
        age=int(rng.integers(a["age"][0], a["age"][1] + 1)),
        city=str(rng.choice(C.CITIES[tier])),
        city_tier=tier,
        occupation=str(rng.choice(a["occupation"])),
        income_type=a["income_type"],
        base_income=income,
        dependents=int(rng.integers(a["dependents"][0], a["dependents"][1] + 1)),
        has_loan=has_loan,
        emi_amount=emi,
        sip_amount=sip,
        owns_home=owns_home,
        has_vehicle=has_vehicle,
        primary_bank=str(rng.choice(C.BANKS)),
        account_last4=f"{int(rng.integers(0, 10000)):04d}",
        volatility=float(rng.uniform(*a["volatility"])),
        savings_rate=savings_rate,
        opening_balance=float(round(income * rng.uniform(0.3, 1.5), -2)),
        opening_reserve=float(round(monthly_spend * rng.choice([0.0, 0.5, 1, 2, 3, 4, 6, 8, 12],
                                                              p=[.08, .10, .12, .15, .15, .12, .12, .08, .08]), -2)),
        employer=str(rng.choice(C.COMPANIES)),
        favourite_merchants=_pick_favourites(rng),
        cat_weights=cat_weights,
    )


# --------------------------------------------------------------------------
# Injected events (ground truth)
# --------------------------------------------------------------------------
@dataclass
class Event:
    customer_id: str
    event_type: str            # category_creep | category_spike | job_loss | income_raise | medical_emergency
    category: str | None
    start_month: int           # 0-based month index in the window
    end_month: int             # inclusive
    magnitude: float           # multiplier (e.g. 1.8 = +80%)
    description: str = ""


def _month_label(i: int) -> str:
    y = C.START_DATE.year + (C.START_DATE.month - 1 + i) // 12
    m = (C.START_DATE.month - 1 + i) % 12 + 1
    return f"{y}-{m:02d}"


def build_events(cust: Customer, n_months: int, rng: np.random.Generator) -> list[Event]:
    events: list[Event] = []
    lo, hi = C.MIN_HISTORY_MONTHS, n_months - 3     # keep 6 clean months of history first

    def start():
        return int(rng.integers(lo, hi + 1))

    if rng.random() < C.EVENT_PROBS["category_creep"]:
        cats = [c for c in C.CREEP_CATEGORIES if cust.cat_weights.get(c, 0) > 0]
        cat = str(rng.choice(cats))
        s = start()
        mag = float(round(rng.uniform(1.5, 2.3), 2))
        events.append(Event(cust.customer_id, "category_creep", cat, s, n_months - 1, mag,
                            f"{cat} spending ramps up to {mag:.2f}x from {_month_label(s)}"))

    if rng.random() < C.EVENT_PROBS["category_spike"]:
        cat = str(rng.choice([c for c in C.SPIKE_CATEGORIES if cust.cat_weights.get(c, 0) > 0]))
        s = start()
        dur = int(rng.integers(1, 3))
        mag = float(round(rng.uniform(2.0, 3.2), 2))
        events.append(Event(cust.customer_id, "category_spike", cat, s, min(s + dur - 1, n_months - 1), mag,
                            f"One-off {cat} burst of {mag:.2f}x for {dur} month(s)"))

    if cust.income_type in ("salaried", "freelance") and rng.random() < C.EVENT_PROBS["job_loss"]:
        s = int(rng.integers(max(lo, 12), hi + 1))
        dur = int(rng.integers(2, 5))
        events.append(Event(cust.customer_id, "job_loss", None, s, min(s + dur - 1, n_months - 1), 0.0,
                            f"Income stops for {dur} months from {_month_label(s)}; discretionary spend cut"))

    if cust.income_type in ("salaried", "freelance", "govt") and rng.random() < C.EVENT_PROBS["income_raise"]:
        s = int(rng.integers(lo, hi + 1))
        mag = float(round(rng.uniform(1.15, 1.40), 2))
        events.append(Event(cust.customer_id, "income_raise", None, s, n_months - 1, mag,
                            f"Income rises {int((mag - 1) * 100)}% from {_month_label(s)}"))

    if rng.random() < C.EVENT_PROBS["medical_emergency"]:
        s = start()
        mag = float(round(rng.uniform(3.0, 7.0), 1))     # multiple of monthly income spent in one month
        events.append(Event(cust.customer_id, "medical_emergency", "Health", s, s, mag,
                            f"Medical emergency costing ~{mag:.1f}x monthly income in {_month_label(s)}"))
    return events


def events_to_frame(events: list[Event]) -> pd.DataFrame:
    rows = []
    for i, e in enumerate(events, 1):
        sm = C.START_DATE.month - 1 + e.start_month
        em = C.START_DATE.month - 1 + e.end_month
        rows.append(dict(
            event_id=i, customer_id=e.customer_id, event_type=e.event_type, category=e.category,
            start_month=f"{C.START_DATE.year + sm // 12}-{sm % 12 + 1:02d}-01",
            end_month=f"{C.START_DATE.year + em // 12}-{em % 12 + 1:02d}-01",
            magnitude=e.magnitude, description=e.description,
        ))
    return pd.DataFrame(rows, columns=["event_id", "customer_id", "event_type", "category",
                                       "start_month", "end_month", "magnitude", "description"])


def customers_to_frames(customers: list[Customer]) -> tuple[pd.DataFrame, pd.DataFrame]:
    cust_df = pd.DataFrame([dict(
        customer_id=c.customer_id, name=c.name, age=c.age, city=c.city, city_tier=c.city_tier,
        occupation=c.occupation, income_type=c.income_type, monthly_income=c.base_income,
        dependents=c.dependents, has_loan=c.has_loan, owns_home=c.owns_home, has_vehicle=c.has_vehicle,
        persona=c.persona,
    ) for c in customers])
    acc_df = pd.DataFrame([dict(
        account_id=f"A{c.customer_id[1:]}", customer_id=c.customer_id, bank=c.primary_bank,
        account_last4=c.account_last4, opening_balance=c.opening_balance,
        opening_reserve=c.opening_reserve, emi_amount=c.emi_amount, sip_amount=c.sip_amount,
    ) for c in customers])
    return cust_df, acc_df
