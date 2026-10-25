"""
Feature engineering for the expense categoriser.

LEAKAGE RULES (enforced here)
  * Text features come from `raw_description` ONLY. The `merchant` and `true_category` columns are never used.
  * Customer-level context (`monthly_income`) is allowed because it is known at prediction time.
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

_HANDLE = re.compile(r"@[a-z0-9.]+")
_LONGNUM = re.compile(r"\d{3,}")
_PUNCT = re.compile(r"[^a-z0-9 ]+")
_SPACE = re.compile(r"\s+")
_ALNUM_ID = re.compile(r"\b(?=[a-z]*\d)(?=\d*[a-z])[a-z0-9]{6,}\b")    # mixed letter+digit ids (ATM / terminal ids)


def clean_description(s: str) -> str:
    """Lower-case, drop UPI handles and long reference numbers, strip punctuation."""
    s = _HANDLE.sub(" ", str(s).lower())
    s = _LONGNUM.sub(" ", s)
    s = _ALNUM_ID.sub(" ", s)
    s = _PUNCT.sub(" ", s)
    return _SPACE.sub(" ", s).strip()


CHANNELS = ["UPI", "Card", "NetBanking", "AutoDebit", "ATM", "NEFT"]
STRUCT_COLUMNS = ["log_real_amount", "amount_round", "amount_ends_9", "amount_mod_500", "has_paise", "real_amount_to_income",
                  "channel_code", "hour", "dow", "dom", "is_month_start", "is_weekend", "n_tokens", "text_len"]
CATEGORICAL_STRUCT = ["channel_code"]


BASE_DATE = pd.Timestamp("2023-01-01")


def structured_features(df: pd.DataFrame, income: pd.Series | None = None, annual_inflation: float = 0.06) -> pd.DataFrame:
    """Numeric context features. `income` maps customer_id -> monthly income (optional).
    Amounts are deflated by a CPI index so trees do not learn absolute thresholds that drift with inflation
    (a model trained on 2023-24 must still work in 2025)."""
    ts = pd.to_datetime(df["txn_ts"])
    amt = df["amount"].astype(float)
    cpi = (1 + annual_inflation) ** (((ts - BASE_DATE).dt.days / 365.25).clip(lower=0))
    real = amt / cpi
    cleaned = df["clean_text"] if "clean_text" in df else df["raw_description"].map(clean_description)
    out = pd.DataFrame(index=df.index)
    out["log_real_amount"] = np.log1p(real)
    out["amount_round"] = (amt % 1 == 0).astype(int)
    out["amount_ends_9"] = ((amt.round() % 10) == 9).astype(int)          # subscription / recharge price points
    out["amount_mod_500"] = ((amt % 500) == 0).astype(int)                 # ATM / rent style
    out["has_paise"] = (np.abs(amt - amt.round()) > 1e-9).astype(int)
    if income is not None:
        inc = df["customer_id"].map(income).astype(float)
        out["real_amount_to_income"] = np.where(inc > 0, real / inc, np.nan)
    else:
        out["real_amount_to_income"] = np.nan
    out["channel_code"] = df["channel"].map({c: i for i, c in enumerate(CHANNELS)}).fillna(-1).astype(int)
    out["hour"] = ts.dt.hour
    out["dow"] = ts.dt.dayofweek
    out["dom"] = ts.dt.day
    out["is_month_start"] = (ts.dt.day <= 7).astype(int)
    out["is_weekend"] = (ts.dt.dayofweek >= 5).astype(int)
    out["n_tokens"] = cleaned.str.count(" ") + 1
    out["text_len"] = cleaned.str.len()
    return out[STRUCT_COLUMNS]


# ------------------------------------------------------------------ payee history (leak-free)
HISTORY_COLUMNS = ["payee_n_prior", "payee_n_90d", "payee_days_since_prev", "payee_same_amount_prev",
                   "payee_monthly_recur", "payee_mem_n"]


def history_features(df: pd.DataFrame, classes: list[str], label_col: str = "category") -> pd.DataFrame:
    """Per-row features from the SAME customer's strictly earlier transactions with the same cleaned payee text.

    * how often this payee was paid before (ever / last 90 days) and how long ago
    * did the previous payment to this payee have (almost) the same amount?  -> rent, tuition, EMI, subscriptions
    * did a same-text payment occur ~28-35 days ago? (monthly recurrence)
    * the customer's own earlier LABELS for this payee: share of each category among prior labeled rows (mem_<class>)

    `df` must contain the history AND the rows to featurise (e.g. concat(past, new)); the result is aligned to df.index.
    It only uses `label_col` (the user/bank label, possibly noisy or blank), never `true_category`.
    """
    d = df.copy()
    d["_text"] = d["clean_text"] if "clean_text" in d else d["raw_description"].map(clean_description)
    d["_ts"] = pd.to_datetime(d["txn_ts"])
    d["_gid"] = pd.factorize(d["customer_id"].astype(str) + "|" + d["_text"])[0]
    order = np.lexsort((d["_ts"].to_numpy(), d["_gid"].to_numpy()))            # by payee group, then time
    g = d["_gid"].to_numpy()[order]
    ts = d["_ts"].to_numpy()[order].astype("datetime64[s]").astype(np.int64)
    amt = d["amount"].to_numpy(float)[order]
    n = len(g)
    first = np.r_[True, g[1:] != g[:-1]]
    start = np.maximum.accumulate(np.where(first, np.arange(n), 0))              # index of group start
    n_prior = np.arange(n) - start

    BIG = 4_000_000_000
    key = g.astype(np.int64) * BIG + ts
    DAY = 86400
    s90 = np.searchsorted(key, g.astype(np.int64) * BIG + (ts - 90 * DAY), side="left")
    n_90 = np.arange(n) - np.maximum(s90, start)

    prev_ts = np.where(first, np.nan, np.r_[np.nan, ts[:-1]].astype(float))
    prev_amt = np.where(first, np.nan, np.r_[np.nan, amt[:-1]])
    days_since = (ts - prev_ts) / DAY
    same_amt = (np.abs(amt - prev_amt) <= 0.02 * np.maximum(amt, 1)).astype(float)
    same_amt[first] = np.nan
    # monthly recurrence: any prior same-payee txn 28-35 days earlier
    lo = np.searchsorted(key, g.astype(np.int64) * BIG + (ts - 35 * DAY), side="left")
    hi = np.searchsorted(key, g.astype(np.int64) * BIG + (ts - 28 * DAY), side="right")
    monthly = (np.maximum(hi, start) - np.maximum(lo, start) > 0).astype(float)

    # customer's own earlier labels for this payee (counts excluding the current row)
    lab = d[label_col].to_numpy(dtype=object)[order]
    K = len(classes)
    cidx = {c: i for i, c in enumerate(classes)}
    onehot = np.zeros((n, K))
    for i, l in enumerate(lab):
        j = cidx.get(l) if l is not None and l == l else None
        if j is not None:
            onehot[i, j] = 1.0
    cs = np.cumsum(onehot, axis=0)
    base = np.where(first[:, None], 0.0, 0.0)
    grp_base = cs[start - 1] * (start > 0)[:, None] if n else cs                  # cumulative count before the group
    prior = cs - onehot - grp_base                                                # prior labeled rows in group, excl. self
    mem_n = prior.sum(1)
    mem = prior / np.maximum(mem_n, 1)[:, None]

    out = np.empty((n, len(HISTORY_COLUMNS) + K))
    out[:, 0], out[:, 1], out[:, 2], out[:, 3], out[:, 4], out[:, 5] = n_prior, n_90, days_since, same_amt, monthly, mem_n
    out[:, 6:] = mem
    res = pd.DataFrame(out, columns=HISTORY_COLUMNS + [f"mem_{c}" for c in classes], index=d.index[order])
    return res.loc[df.index]
