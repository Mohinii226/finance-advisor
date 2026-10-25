"""
Transaction engine: turns a Customer profile (+ injected events) into
months of realistic Indian banking transactions.

Design notes
------------
* Spend per category per month = customer share x income x seasonality x festival
  x event multiplier x personal month-to-month noise x inflation.
* Variable categories: number of transactions ~ Poisson, ticket ~ lognormal.
* Fixed categories (rent, EMI, SIP, subscriptions, recharge) recur on a fixed day.
* Descriptions are rendered in messy bank-statement style (UPI / POS / ACH ...).
* A running ledger adds balance_after, month-end sweeps to savings and
  top-ups when the account would go negative.
"""
from __future__ import annotations

import calendar
import random
import re
import string
from datetime import date, datetime, timedelta

import numpy as np
import pandas as pd

from . import config as C
from .profiles import Customer, Event, random_name

MONTH_ABBR = [m.upper() for m in calendar.month_abbr]
FEST_WINDOW = 7   # days either side of a festival

# --------------------------------------------------------------------------
# Description rendering
# --------------------------------------------------------------------------

def _vpa(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", name.lower())[:14] or "merchant"


def _ref(r: random.Random, n: int = 12) -> str:
    return "".join(r.choice(string.digits) for _ in range(n))


def _alnum(r: random.Random, n: int = 6) -> str:
    return "".join(r.choice(string.ascii_uppercase + string.digits) for _ in range(n))


def _noisy_name(name: str, r: random.Random) -> str:
    """Apply bank-statement style noise to a merchant string."""
    out = name
    x = r.random()
    if x < 0.12 and len(out) > 8:                       # truncation
        out = out[: r.randint(7, len(out) - 1)]
    elif x < 0.20 and len(out) > 4:                     # dropped char (typo)
        i = r.randint(1, len(out) - 2)
        out = out[:i] + out[i + 1:]
    if r.random() < 0.12:                               # aggregator prefix
        out = r.choice(["RAZORPAY*", "PAYU*", "PTM*", "BILLDESK*", "CCAV*"]) + out
    if r.random() < 0.15:
        out = out.title()
    if r.random() < 0.07:
        out = out.replace(" ", "  ")
    return out


def render_description(merchant: str, channel: str, city: str, r: random.Random, person: bool = False) -> str:
    m = _noisy_name(merchant, r)
    handle = r.choice(C.UPI_HANDLES)
    vpa = _vpa(merchant)
    if channel == "UPI":
        style = r.randint(1, 4)
        if style == 1:
            d = f"UPI-{m}-{vpa}@{handle}-{_ref(r)}-UPI"
        elif style == 2:
            d = f"UPI/{_ref(r)}/{m}/{vpa}@{handle}"
        elif style == 3:
            d = f"{m}@{handle} UPI {_ref(r)}"
        else:
            d = f"UPI/{'P2P' if person else 'P2M'}/{_ref(r)}/{m}"
    elif channel == "Card":
        style = r.randint(1, 5)
        d = {1: f"POS {_ref(r, 6)} {m} {city[:12].upper()}", 2: f"{m} {city.upper()} IN",
             3: f"ECOM PUR/{m}", 4: f"{m}*{_alnum(r)}", 5: f"PURCHASE {m} {city.upper()}"}[style]
    elif channel == "NetBanking":
        d = r.choice([f"BILLDESK*{m}", f"NB-{m}-{_ref(r, 8)}", f"IB BILLPAY DR-{m}", f"NEFT-{m}-{_ref(r, 10)}"])
    elif channel == "AutoDebit":
        d = r.choice([f"ACH D- {m}-{_ref(r, 8)}", f"NACH-{m}-{_ref(r, 6)}", f"SI-{m}-{_ref(r, 8)}", f"AUTOPAY {m}"])
    elif channel == "ATM":
        d = r.choice([f"ATM WDL {_alnum(r, 8)} {city.upper()}", f"CASH WDL {_alnum(r, 6)}", f"NFS*CASH WDL {_alnum(r, 8)}"])
    else:
        d = f"{channel}-{m}-{_ref(r, 8)}"
    return d if r.random() < 0.85 else d.upper()


# --------------------------------------------------------------------------
# Calendar helpers
# --------------------------------------------------------------------------

def _festival_month_boost(year: int, month: int, cat: str) -> float:
    """Budget boost for the month = sum of festival boosts x share of the +/-7d window inside this month."""
    first = date(year, month, 1)
    last = date(year, month, calendar.monthrange(year, month)[1])
    boost = 0.0
    for fdate, _, boosts in C.FESTIVALS:
        b = boosts.get(cat)
        if not b:
            continue
        w0, w1 = fdate - timedelta(days=FEST_WINDOW), fdate + timedelta(days=FEST_WINDOW)
        overlap = (min(last, w1) - max(first, w0)).days + 1
        if overlap > 0:
            boost += b * overlap / (2 * FEST_WINDOW + 1)
    return boost


def _day_weights(year: int, month: int, cat: str) -> np.ndarray:
    n = calendar.monthrange(year, month)[1]
    wk = C.WEEKDAY_WEIGHTS.get(cat, C.WEEKDAY_WEIGHTS["default"])
    w = np.array([wk[date(year, month, d).weekday()] for d in range(1, n + 1)], dtype=float)
    for fdate, _, boosts in C.FESTIVALS:
        b = boosts.get(cat)
        if not b:
            continue
        for d in range(1, n + 1):
            if abs((date(year, month, d) - fdate).days) <= FEST_WINDOW:
                w[d - 1] *= 1 + 2 * b
    return w / w.sum()


def _rand_time(cat: str, channel: str, r: random.Random) -> tuple[int, int, int]:
    if channel == "AutoDebit":
        return r.randint(1, 6), r.randint(0, 59), r.randint(0, 59)
    if cat in C.EVENING_CATEGORIES:
        h = r.choices(range(8, 24), weights=[1, 1, 2, 3, 4, 3, 2, 2, 3, 4, 6, 8, 8, 6, 4, 2])[0]
    elif channel == "ATM":
        h = r.randint(9, 20)
    else:
        h = r.randint(7, 21)
    return h, r.randint(0, 59), r.randint(0, 59)


# --------------------------------------------------------------------------
# Event helpers
# --------------------------------------------------------------------------

def _event_mult(events: list[Event], cat: str, m: int) -> float:
    mult = 1.0
    for e in events:
        if e.category != cat or not (e.start_month <= m <= e.end_month):
            continue
        if e.event_type == "category_creep":
            ramp = min(1.0, (m - e.start_month + 1) / 3)       # ramps up over 3 months
            mult *= 1 + (e.magnitude - 1) * ramp
        elif e.event_type == "category_spike":
            mult *= e.magnitude
    return mult


def _in_event(events: list[Event], etype: str, m: int) -> Event | None:
    for e in events:
        if e.event_type == etype and e.start_month <= m <= e.end_month:
            return e
    return None


# --------------------------------------------------------------------------
# Main generator
# --------------------------------------------------------------------------

def _pick_channel(cat: str, r: random.Random) -> str:
    mix = C.CHANNEL_MIX.get(cat, C.CHANNEL_MIX["default"])
    return r.choices(list(mix), weights=list(mix.values()))[0]


def _round_amount(x: float, channel: str, cat: str, r: random.Random) -> float:
    if channel == "ATM":
        return float(max(500, round(x / 500) * 500))
    if channel == "Card" and r.random() < 0.6:
        return round(x, 2)
    return float(max(1, round(x)))


def _income_levels(cust: Customer, n_months: int, events: list[Event], rng: np.random.Generator) -> np.ndarray:
    """Monthly income level (before job-loss), including hikes and raise events."""
    level = np.empty(n_months)
    cur = cust.base_income
    for m in range(n_months):
        cal_month = (C.START_DATE.month - 1 + m) % 12 + 1
        if m > 0:
            if cust.income_type == "salaried" and cal_month == 4:
                cur *= rng.uniform(1.04, 1.12)
            elif cust.income_type == "govt" and cal_month in (1, 7):
                cur *= 1.04
            elif cust.income_type == "freelance":
                cur *= 1.004
            elif cust.income_type == "pension" and cal_month == 7:
                cur *= 1.03
        mult = 1.0
        for e in events:
            if e.event_type == "income_raise" and m >= e.start_month:
                mult = max(mult, e.magnitude)
        level[m] = cur * mult
    return level


def generate_customer_transactions(cust: Customer, events: list[Event], n_months: int,
                                   rng: np.random.Generator) -> list[tuple]:
    r = random.Random(int(rng.integers(0, 2**31 - 1)))
    rows: list[tuple] = []        # (cid, ts, amount, type, desc, merchant, true_cat, channel)
    cid, city = cust.customer_id, cust.city
    levels = _income_levels(cust, n_months, events, rng)
    parent = random_name(rng)
    contacts = [random_name(rng) for _ in range(rng.integers(4, 9))]
    clients = [f"{r.choice(['ACME','BLUEWAVE','NEXGEN','PIXEL','ORBIT','SKYLINE','VERTEX'])} "
               f"{r.choice(['TECH','MEDIA','SOLUTIONS','LABS','STUDIOS'])}" for _ in range(r.randint(3, 6))]

    def add(ts, amount, typ, desc, merchant, cat, channel):
        rows.append((cid, ts, float(amount), typ, desc, merchant, cat, channel))

    def pick_merchant(cat):
        names, probs = cust.favourite_merchants[cat]
        return str(rng.choice(names, p=probs))

    # ---- static (customer-level) recurring items -------------------------------------
    base0 = cust.base_income
    fixed: list[dict] = []
    for cat in ("Rent", "Education", "Mobile & Internet"):
        b0 = cust.cat_weights.get(cat, 0) * base0
        if b0 < C.CATEGORIES[cat]["floor"]:
            continue
        if cat == "Mobile & Internet":
            for share, mer in ((0.6, pick_merchant(cat)), (0.4, pick_merchant(cat))):
                amt = max(99, round(b0 * share / 10) * 10 - 1)
                fixed.append(dict(cat=cat, merchant=mer, day=r.randint(1, 28), amount=amt, step=1.03,
                                  channel=_pick_channel(cat, r)))
        else:
            amt = round(b0 / (500 if cat == "Rent" else 100)) * (500 if cat == "Rent" else 100)
            fixed.append(dict(cat=cat, merchant=pick_merchant(cat), day=r.randint(1, 7) if cat == "Rent" else r.randint(5, 15),
                              amount=amt, step=1.06 if cat == "Rent" else 1.05, channel=_pick_channel(cat, r)))
    # Subscriptions (fixed price, fixed day)
    b_sub = cust.cat_weights.get("Subscriptions", 0) * base0
    n_sub = int(np.clip(round(b_sub / 250), 1, 6)) if b_sub > 0 else 0
    sub_names, sub_probs = cust.favourite_merchants["Subscriptions"]
    for mer in (rng.choice(sub_names, size=min(n_sub, len(sub_names)), replace=False, p=sub_probs) if n_sub else []):
        fixed.append(dict(cat="Subscriptions", merchant=str(mer), day=r.randint(1, 28),
                          amount=C.SUBSCRIPTION_PRICES.get(str(mer), 199), step=1.0, channel=_pick_channel("Subscriptions", r)))
    # Insurance (quarterly)
    b_ins = cust.cat_weights.get("Insurance", 0) * base0 * 3
    ins = None
    if b_ins >= C.CATEGORIES["Insurance"]["floor"]:
        ins = dict(merchant=pick_merchant("Insurance"), offset=r.randint(0, 2), day=r.randint(3, 25),
                   amount=round(b_ins / 100) * 100, channel=_pick_channel("Insurance", r))
    emi_day, emi_lender = r.randint(5, 10), r.choice(C.LENDERS)
    sip_day, sip_inv = r.randint(7, 12), r.choice(C.INVESTMENT_MERCHANTS)
    sal_day_choice = r.choice([1, 1, 1, 1, 28, 30, 31])
    bonus_months = {y: r.random() < 0.5 for y in range(C.START_DATE.year, C.START_DATE.year + n_months // 12 + 2)}

    # per-category static frequency (persona scaling)
    freq_static = {}
    for cat, spec in C.CATEGORIES.items():
        ratio = cust.cat_weights.get(cat, 0) / (spec["weight"] * 0.65)
        freq_static[cat] = spec["freq"] * float(np.clip(ratio ** 0.5, 0.4, 2.5)) if ratio > 0 else 0.0

    # ---- month loop -----------------------------------------------------------------
    for m in range(n_months):
        year = C.START_DATE.year + (C.START_DATE.month - 1 + m) // 12
        month = (C.START_DATE.month - 1 + m) % 12 + 1
        dim = calendar.monthrange(year, month)[1]
        infl = (1 + C.ANNUAL_INFLATION) ** (m / 12)
        level = levels[m]
        spend_base = base0 * (1 + 0.6 * (level / base0 - 1)) * infl
        job_loss = _in_event(events, "job_loss", m)

        def ts_for(day, cat, channel):
            h, mi, s = _rand_time(cat, channel, r)
            return datetime(year, month, min(day, dim), h, mi, s)

        # -- income --
        inc_cat = C.INCOME_CATEGORY[cust.income_type]
        if not job_loss:
            if cust.income_type in ("salaried", "govt"):
                d = sal_day_choice
                if d >= 28:                       # end-of-month payroll
                    d = dim
                ts = datetime(year, month, min(d, dim), r.randint(0, 9), r.randint(5, 59), r.randint(0, 59))
                add(ts, round(level, 0), "credit",
                    f"NEFT-{cust.employer}-SALARY {MONTH_ABBR[month]}{str(year)[2:]}", cust.employer, inc_cat, "NEFT")
                if cust.income_type == "salaried" and month == 3 and bonus_months.get(year):
                    add(datetime(year, month, min(31, dim), 11, r.randint(0, 59), 0), round(level * rng.uniform(0.3, 1.0), 0),
                        "credit", f"NEFT-{cust.employer}-PERF BONUS", cust.employer, inc_cat, "NEFT")
            elif cust.income_type == "pension":
                add(datetime(year, month, dim, 8, r.randint(0, 59), 0), round(level, 0), "credit",
                    f"PENSION CREDIT-{r.choice(['TREASURY','EPFO','SBI PENSION'])}-{MONTH_ABBR[month]}", "PENSION", inc_cat, "NEFT")
            elif cust.income_type == "student":
                add(datetime(year, month, r.randint(1, 5), 10, r.randint(0, 59), 0), round(level * 0.7, -2), "credit",
                    f"UPI-{parent.upper()}-{_vpa(parent)}@{r.choice(C.UPI_HANDLES)}-{_ref(r)}", parent.upper(), inc_cat, "UPI")
                if r.random() < 0.8:
                    add(datetime(year, month, r.randint(8, 28), 15, r.randint(0, 59), 0), round(level * 0.3 * rng.uniform(0.5, 1.4), -1),
                        "credit", f"NEFT-{r.choice(C.COMPANIES)}-STIPEND", "STIPEND", inc_cat, "NEFT")
            else:  # freelance: lumpy invoices
                total = level * float(rng.lognormal(0, max(cust.volatility, 0.2)))
                k = int(rng.poisson(2.5)) + 1
                parts = rng.dirichlet(np.ones(k)) * total
                for p in parts:
                    cl = r.choice(clients)
                    add(datetime(year, month, r.randint(1, dim), r.randint(9, 19), r.randint(0, 59), 0), round(p, -1), "credit",
                        f"NEFT-{cl}-INV{r.randint(100, 999)}", cl, inc_cat, "NEFT")
        elif cust.income_type == "freelance" and r.random() < 0.5:   # trickle during "job loss"
            add(datetime(year, month, r.randint(1, dim), 12, 0, 0), round(level * 0.15, -1), "credit",
                f"NEFT-{r.choice(clients)}-INV{r.randint(100, 999)}", "CLIENT", inc_cat, "NEFT")

        # other income: interest, cashback/refunds
        if month in (3, 6, 9, 12):
            add(datetime(year, month, dim, 3, 30, 0), round(cust.opening_reserve * 0.015 + r.randint(50, 400), 0), "credit",
                "INT.PD:SAVINGS A/C INTEREST", "BANK INTEREST", "Other Income", "NEFT")
        if r.random() < 0.45:
            mer = r.choice(["AMAZON", "FLIPKART", "SWIGGY", "MYNTRA", "UBER"])
            ch = "UPI"
            add(ts_for(r.randint(1, dim), "Other Income", ch), round(r.uniform(80, 1800)), "credit",
                f"UPI-REFUND-{mer}-{_ref(r, 10)}", mer, "Other Income", ch)

        # -- fixed outgoings --
        yrs = m // 12
        for f in fixed:
            amt = round(f["amount"] * (f["step"] ** yrs))
            ts = ts_for(f["day"], f["cat"], f["channel"])
            add(ts, amt, "debit", render_description(f["merchant"], f["channel"], city, r), f["merchant"], f["cat"], f["channel"])
        if ins and (month - 1) % 3 == ins["offset"]:
            add(ts_for(ins["day"], "Insurance", ins["channel"]), ins["amount"], "debit",
                render_description(ins["merchant"], ins["channel"], city, r), ins["merchant"], "Insurance", ins["channel"])
        if cust.has_loan:
            add(ts_for(emi_day, "EMI", "AutoDebit"), cust.emi_amount, "debit",
                f"ACH D- {emi_lender}-{_ref(r, 8)}", emi_lender, "EMI", "AutoDebit")
        if cust.sip_amount > 0 and not job_loss:
            sip_amt = round(cust.sip_amount * (level / base0) / 500) * 500
            add(ts_for(sip_day, "Investments", "AutoDebit"), max(500, sip_amt), "debit",
                f"ACH D- {sip_inv}-{_ref(r, 8)}", sip_inv, "Investments", "AutoDebit")

        # -- variable categories --
        for cat, spec in C.CATEGORIES.items():
            if spec["kind"] != "variable" or cust.cat_weights.get(cat, 0) <= 0:
                continue
            seasonal = C.SEASONAL.get(cat, {}).get(month, 1.0)
            fest = 1 + _festival_month_boost(year, month, cat)
            emult = _event_mult(events, cat, m)
            noise = float(rng.lognormal(0, cust.volatility))
            cut = 0.6 if (job_loss and not spec["essential"]) else 1.0
            mult = seasonal * fest * emult * noise * cut
            budget = cust.cat_weights[cat] * spend_base * mult
            f_eff = freq_static[cat] * float(np.clip(mult ** 0.5, 0.5, 2.2))
            n = int(rng.poisson(f_eff))
            if emult > 1.3:
                n = max(n, 1)
            elif f_eff >= 2:
                n = max(n, 1)
            if n == 0:
                continue
            tm = budget / max(f_eff, 1e-6)
            sig = spec["sigma"]
            tickets = rng.lognormal(np.log(tm) - sig ** 2 / 2, sig, size=n)
            days = rng.choice(np.arange(1, dim + 1), size=n, p=_day_weights(year, month, cat))
            for tk, d in zip(tickets, days):
                if cat == "Transfers":
                    person = r.choice(contacts)
                    ch = "UPI"
                    desc = render_description(person.upper(), ch, city, r, person=True)
                    mer = person.upper()
                elif cat == "Cash Withdrawal":
                    ch, mer = "ATM", "ATM"
                    desc = render_description("ATM", ch, city, r)
                else:
                    mer = pick_merchant(cat)
                    ch = _pick_channel(cat, r)
                    desc = render_description(mer, ch, city, r)
                tk = max(spec["floor"], tk)
                add(ts_for(int(d), cat, ch), _round_amount(tk, ch, cat, r), "debit", desc, mer, cat, ch)

        # -- medical emergency (one-off lump) --
        med = _in_event(events, "medical_emergency", m)
        if med:
            total = med.magnitude * level
            k = r.randint(1, 3)
            for p in rng.dirichlet(np.ones(k)) * total:
                mer = r.choice(["MANIPAL HOSPITAL", "FORTIS HOSPITAL", "APOLLO PHARMACY", "DR LAL PATHLABS", "LOCAL CLINIC"])
                ch = r.choice(["Card", "NetBanking", "UPI"])
                add(ts_for(r.randint(1, dim), "Health", ch), max(500, round(p, -1)), "debit",
                    render_description(mer, ch, city, r), mer, "Health", ch)

    return rows


# --------------------------------------------------------------------------
# Ledger: running balance, savings sweeps, top-ups
# --------------------------------------------------------------------------

def apply_ledger(df: pd.DataFrame, customers: dict[str, Customer], rng: np.random.Generator) -> pd.DataFrame:
    """Walk each customer's rows in time order; add balance_after, sweeps and top-ups."""
    out = []
    r = random.Random(int(rng.integers(0, 2**31 - 1)))
    for cid, g in df.sort_values(["customer_id", "txn_ts"]).groupby("customer_id", sort=False):
        cust = customers[cid]
        bal, reserve = cust.opening_balance, cust.opening_reserve
        parent = random_name(rng).upper()
        cur_month = None
        for row in g.itertuples(index=False):
            ym = (row.txn_ts.year, row.txn_ts.month)
            if ym != cur_month:
                # month-start sweep of idle cash into savings (FD/RD)
                if cur_month is not None and bal > 2.5 * cust.base_income:
                    sweep = float(round((bal - 1.0 * cust.base_income) / 1000) * 1000)
                    bal -= sweep
                    reserve += sweep
                    out.append((cid, row.txn_ts.replace(day=1, hour=0, minute=1, second=0), sweep, "debit",
                                f"UPI-SELF TRANSFER-RD/FD A/C-{_ref(r, 10)}", "SELF TRANSFER", "Self Transfer", "UPI", bal, None))
                cur_month = ym
            if row.type == "debit" and bal - row.amount < 500:
                topup = float(np.ceil((row.amount - bal + 2000) / 1000) * 1000)
                ts = row.txn_ts - timedelta(minutes=1)
                if reserve >= topup:
                    reserve -= topup
                    bal += topup
                    out.append((cid, ts, topup, "credit", f"UPI-SELF TRANSFER-FD LIQUIDATION-{_ref(r, 10)}",
                                "SELF TRANSFER", "Self Transfer", "UPI", bal, None))
                else:
                    bal += topup
                    out.append((cid, ts, topup, "credit", f"UPI-{parent}-{_vpa(parent)}@oksbi-{_ref(r)}",
                                parent, "Other Income", "UPI", bal, None))
            bal += row.amount if row.type == "credit" else -row.amount
            out.append((cid, row.txn_ts, row.amount, row.type, row.raw_description, row.merchant,
                        row.true_category, row.channel, bal, None))
    cols = ["customer_id", "txn_ts", "amount", "type", "raw_description", "merchant", "true_category",
            "channel", "balance_after", "category"]
    res = pd.DataFrame(out, columns=cols)
    res["balance_after"] = res["balance_after"].round(2)
    return res


def apply_label_noise(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Create the 'user/bank label' column: mostly right, 8% blank, 3% wrong."""
    df = df.copy()
    eligible = (df["type"] == "debit") & (~df["true_category"].isin(["EMI", "Investments", "Self Transfer"]))
    u = rng.random(len(df))
    cat = df["true_category"].copy()
    blank = eligible & (u < C.BLANK_LABEL_RATE)
    wrong = eligible & (u >= C.BLANK_LABEL_RATE) & (u < C.BLANK_LABEL_RATE + C.WRONG_LABEL_RATE)
    all_cats = list(C.CATEGORIES)
    for idx in df.index[wrong]:
        t = df.at[idx, "true_category"]
        opts = C.CONFUSABLE.get(t, [])
        cat.at[idx] = rng.choice(opts) if (opts and rng.random() < 0.6) else rng.choice([c for c in all_cats if c != t])
    cat[blank] = None
    df["category"] = cat
    return df
