"""
Bank-style SMS generator (bonus module).

Produces realistic transaction alerts from the bank each customer uses, plus
non-transactional noise (OTPs, promos, balance alerts) that a parser must ignore.
`true_txn_id` is the ground truth link back to `transactions`, so you can later
measure your regex/ML parser's precision and recall.
"""
from __future__ import annotations

import random
import re

import numpy as np
import pandas as pd

from . import config as C

SENDER = {"HDFC": "HDFCBK", "SBI": "SBIINB", "ICICI": "ICICIB", "AXIS": "AXISBK", "KOTAK": "KOTAKB"}
PREFIXES = ["AX", "VM", "JD", "BZ", "TX", "CP"]


def _vpa(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", name.lower())[:14] or "merchant"


def _fmt_amt(a: float, style: str) -> str:
    return {"dot2": f"Rs.{a:,.2f}", "plain": f"Rs {a:,.2f}", "inr": f"INR {a:,.2f}", "rs0": f"Rs.{a:,.0f}"}[style]


def _debit_sms(bank: str, last4: str, row, bal: float, r: random.Random) -> str:
    d = row.txn_ts
    ref = "".join(r.choice("0123456789") for _ in range(12))
    mer = str(row.merchant)
    vpa = f"{_vpa(mer)}@{r.choice(C.UPI_HANDLES)}"
    if row.channel == "ATM":
        return (f"{_fmt_amt(row.amount, 'dot2')} withdrawn from A/c XX{last4} at ATM on "
                f"{d:%d-%b-%y %H:%M}. Avl Bal {_fmt_amt(bal, 'dot2')}. -{bank} Bank")
    if row.channel == "Card":
        return (f"Spent {_fmt_amt(row.amount, 'plain')} on {bank} Bank Card XX{last4} at {mer} on "
                f"{d:%d-%b-%y:%H:%M:%S}. Avl bal {_fmt_amt(bal, 'dot2')}. Not you? Call 1800-XXX-XXXX")
    if row.channel == "UPI":
        t = {
            "HDFC": f"{_fmt_amt(row.amount, 'dot2')} debited from A/c XX{last4} on {d:%d-%b-%y} to VPA {vpa} (UPI Ref No {ref}).",
            "SBI": f"Dear UPI user A/C X{last4} debited by {row.amount:.1f} on date {d:%d%b%y} trf to {mer} Refno {ref}. If not done by u, fwd this SMS to 9223008333",
            "ICICI": f"ICICI Bank Acct XX{last4} debited for {_fmt_amt(row.amount, 'plain')} on {d:%d-%b-%y}; {mer} credited. UPI:{ref}.",
            "AXIS": f"INR {row.amount:,.2f} debited from A/c no. XX{last4} on {d:%d-%m-%y} {d:%H:%M:%S} UPI/P2M/{ref}/{mer}. Not you? SMS BLOCKUPI",
            "KOTAK": f"Sent {_fmt_amt(row.amount, 'dot2')} from Kotak Bank AC X{last4} to {vpa} on {d:%d-%m-%y}. UPI Ref {ref}.",
        }
        return t[bank]
    return (f"{_fmt_amt(row.amount, 'dot2')} debited from A/c XX{last4} on {d:%d-%b-%y} towards {mer}. "
            f"Avl Bal {_fmt_amt(bal, 'dot2')}. -{bank} Bank")


def _credit_sms(bank: str, last4: str, row, bal: float, r: random.Random) -> str:
    d = row.txn_ts
    tmpl = [
        f"{_fmt_amt(row.amount, 'dot2')} credited to A/c XX{last4} on {d:%d-%b-%y} by {str(row.merchant)[:28]}. Avl Bal {_fmt_amt(bal, 'dot2')}",
        f"INR {row.amount:,.2f} credited to {bank} A/c XX{last4} on {d:%d-%m-%y}. Info: {str(row.raw_description)[:30]}. Bal: INR {bal:,.2f}",
        f"Dear Customer, A/c XX{last4} is credited with {_fmt_amt(row.amount, 'plain')} on {d:%d-%b-%Y}. Avl Bal {_fmt_amt(bal, 'dot2')}.",
    ]
    return r.choice(tmpl)


NOISE_TEMPLATES = [
    "{otp} is your OTP for transaction of Rs.{amt}. Valid for 10 mins. Do not share with anyone. -{bank} Bank",
    "Dear Customer, your {bank} A/c XX{last4} balance is Rs.{bal}. Avl Bal as on {date}.",
    "Pre-approved personal loan up to Rs.{loan} for you! Apply now at bit.ly/{code}. T&C apply.",
    "Your credit card statement of Rs.{amt} is generated. Min due Rs.{mind} by {date}. Pay via app.",
    "Get 5X reward points on dining this weekend with your {bank} card. T&C apply.",
    "Dear Customer, your {bank} A/c XX{last4} KYC update is due. Click {code} to update.",
]


def generate_sms(txns: pd.DataFrame, customers_df: pd.DataFrame, accounts_df: pd.DataFrame,
                 customer_frac: float = 0.25, months_back: int = 12, noise_ratio: float = 0.25,
                 seed: int = 7) -> pd.DataFrame:
    """Generate SMS rows for a sample of customers over the most recent `months_back` months."""
    rng = np.random.default_rng(seed)
    r = random.Random(seed)
    acc = accounts_df.set_index("customer_id")
    chosen = rng.choice(customers_df["customer_id"].values, size=max(1, int(len(customers_df) * customer_frac)), replace=False)
    cutoff = txns["txn_ts"].max() - pd.DateOffset(months=months_back)
    sub = txns[(txns["customer_id"].isin(chosen)) & (txns["txn_ts"] >= cutoff)]
    sub = sub[sub["true_category"] != "Self Transfer"]       # self sweeps usually don't need an alert here

    rows = []
    for row in sub.itertuples(index=False):
        a = acc.loc[row.customer_id]
        bank, last4 = a["bank"], a["account_last4"]
        body = (_debit_sms if row.type == "debit" else _credit_sms)(bank, last4, row, row.balance_after, r)
        # alerts arrive seconds after the transaction
        recv = row.txn_ts + pd.Timedelta(seconds=int(rng.integers(2, 90)))
        sender = f"{r.choice(PREFIXES)}-{SENDER[bank]}"
        rows.append((row.customer_id, sender, body, recv, int(row.txn_id), True))

    # noise messages
    n_noise = int(len(rows) * noise_ratio)
    for _ in range(n_noise):
        cid = str(rng.choice(chosen))
        a = acc.loc[cid]
        ts = cutoff + pd.Timedelta(seconds=int(rng.integers(0, int((txns["txn_ts"].max() - cutoff).total_seconds()))))
        body = r.choice(NOISE_TEMPLATES).format(
            otp=r.randint(100000, 999999), amt=r.randint(100, 20000), bank=a["bank"], last4=a["account_last4"],
            bal=f"{r.randint(1000, 300000):,}.{r.randint(0, 99):02d}", date=f"{ts:%d-%b-%y}", loan=r.choice([200000, 500000, 1000000]),
            code="".join(r.choice("abcdef0123456789") for _ in range(7)), mind=r.randint(200, 5000))
        rows.append((cid, f"{r.choice(PREFIXES)}-{SENDER[a['bank']]}", body, ts, None, False))

    sms = pd.DataFrame(rows, columns=["customer_id", "sender", "body", "received_at", "true_txn_id", "is_transaction"])
    sms = sms.sort_values("received_at").reset_index(drop=True)
    sms.insert(0, "sms_id", np.arange(1, len(sms) + 1))
    sms["parsed_txn_id"] = None       # your parser fills this in later
    sms["true_txn_id"] = sms["true_txn_id"].astype("Int64")
    return sms
