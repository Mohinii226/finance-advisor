"""
Realistic description ambiguity (applied AFTER everything else, with its own RNG, so amounts, dates, categories and
labels are untouched and earlier results stay reproducible).

Real bank statements do not always name the merchant:
  1. Personal-UPI payments   : you pay the local kirana / barber / clinic via a person's UPI id  -> looks like a P2P transfer.
                               The same shopkeeper keeps the same alias for a customer (1-2 names), as in real statements.
  2. Parent-brand collapse   : "AMAZON PRIME", "AMAZON FRESH" and "AMAZON" all appear as AMAZON; Swiggy / Swiggy
                               Instamart / Swiggy One all appear as SWIGGY; Jio recharge vs JioFiber as JIO ...
  3. Generic aggregators     : "PAYTM", "PHONEPE", "RAZORPAY" with no merchant name at all.

These make part of the task genuinely ambiguous from text alone, so amount, time, channel and customer context
matter (and a Bayes-error ceiling exists). `merchant` and `true_category` are NOT changed.
"""
from __future__ import annotations

import random

import numpy as np
import pandas as pd

from .profiles import random_name
from .transactions import render_description

PERSONAL_UPI_MERCHANTS = {"LOCAL KIRANA", "LOCAL RESTAURANT", "CHAI POINT", "LOCAL BARBER", "LOCAL CLINIC",
                          "TUITION FEE", "COACHING CLASSES", "HOUSE RENT", "PG RENT", "TEMPLE DONATION"}
PARENT_BRAND = {
    "AMAZON PRIME": "AMAZON", "AMAZON FRESH": "AMAZON", "AMAZON": "AMAZON",
    "SWIGGY INSTAMART": "SWIGGY", "SWIGGY ONE": "SWIGGY", "ZOMATO GOLD": "ZOMATO",
    "JIO PREPAID": "JIO", "JIOFIBER": "JIO", "JIOMART": "JIO", "AIRTEL XSTREAM": "AIRTEL",
    "APPLE SERVICES": "APPLE", "GOOGLE ONE": "GOOGLE", "PLAYSTORE": "GOOGLE", "YOUTUBE PREMIUM": "GOOGLE",
    "RELIANCE FRESH": "RELIANCE", "RELIANCE DIGITAL": "RELIANCE", "RELIANCE PETROL": "RELIANCE",
    "BILLDESK ELECTRICITY": "BILLDESK", "NOBROKER RENT": "NOBROKER", "CRED RENT PAY": "CRED",
    "POLICYBAZAAR": "POLICYBAZAAR", "OLA CABS": "OLA", "BLINKIT": "BLINKIT",
}
AGGREGATORS = ["PAYTM", "PHONEPE", "GPAY", "BHARATPE", "RAZORPAY", "PAYU", "CASHFREE", "BILLDESK"]
NEVER_GENERIC = {"Rent", "EMI", "Investments", "Self Transfer", "Cash Withdrawal", "Transfers"}


def make_ambiguous(txns: pd.DataFrame, customers: pd.DataFrame, seed: int = 0,
                   personal_rate: float = 0.60, parent_rate: float = 0.45, generic_rate: float = 0.05) -> pd.DataFrame:
    """Rewrite raw_description for a subset of debit rows. Returns a copy."""
    r = random.Random(seed + 12345)
    rng = np.random.default_rng(seed + 12345)
    city = customers.set_index("customer_id")["city"]
    out = txns.copy()
    deb = out.index[out["type"] == "debit"]
    u = rng.random((len(out), 3))
    pos = {idx: i for i, idx in enumerate(out.index)}
    desc = out["raw_description"].to_numpy(dtype=object)
    stats = {"personal": 0, "parent": 0, "generic": 0}
    aliases: dict = {}          # (customer, merchant) -> 1-2 stable personal-UPI names (the shopkeeper / tutor / landlord)

    def alias(cid, mer):
        names = aliases.setdefault((cid, mer), [random_name(rng).upper() for _ in range(1 + (rng.random() < 0.3))])
        return names[0] if (len(names) == 1 or rng.random() < 0.8) else names[1]

    cities = out["customer_id"].map(city).to_numpy()
    for idx in deb:
        i = pos[idx]
        mer, ch, cat = out.at[idx, "merchant"], out.at[idx, "channel"], out.at[idx, "true_category"]
        if mer in PERSONAL_UPI_MERCHANTS and ch == "UPI" and u[i, 0] < personal_rate:
            desc[i] = render_description(alias(out.at[idx, "customer_id"], mer), "UPI", cities[i], r, person=True)
            stats["personal"] += 1
        elif mer in PARENT_BRAND and u[i, 1] < parent_rate:
            desc[i] = render_description(PARENT_BRAND[mer], ch, cities[i], r)
            stats["parent"] += 1
        elif ch in ("UPI", "Card") and cat not in NEVER_GENERIC and u[i, 2] < generic_rate:
            desc[i] = render_description(r.choice(AGGREGATORS), ch, cities[i], r)
            stats["generic"] += 1
    out["raw_description"] = desc
    out.attrs["ambiguity_stats"] = stats
    return out
