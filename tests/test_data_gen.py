import pandas as pd
import pytest

from data_gen import config as C
from data_gen.generate import build_dataset


@pytest.fixture(scope="module")
def data():
    return build_dataset(n_customers=15, n_months=24, seed=123, sms_frac=0.4)


def test_deterministic():
    a = build_dataset(5, 12, seed=9)["transactions"]
    b = build_dataset(5, 12, seed=9)["transactions"]
    assert a["amount"].sum() == b["amount"].sum() and len(a) == len(b)


def test_basic_integrity(data):
    t = data["transactions"]
    assert t["txn_id"].is_unique
    assert (t["amount"] > 0).all()
    assert t["balance_after"].min() >= 0, "ledger top-ups should keep balance non-negative"
    assert set(t["type"]) == {"debit", "credit"}


def test_label_noise_rates(data):
    t = data["transactions"]
    d = t[(t.type == "debit") & ~t.true_category.isin(["EMI", "Investments", "Self Transfer"])]
    blank = d["category"].isna().mean()
    wrong = ((d["category"].notna()) & (d["category"] != d["true_category"])).mean()
    assert abs(blank - C.BLANK_LABEL_RATE) < 0.02
    assert abs(wrong - C.WRONG_LABEL_RATE) < 0.015


def test_events_leave_clean_history(data):
    ev = data["injected_events"]
    first = pd.Timestamp(C.START_DATE) + pd.DateOffset(months=C.MIN_HISTORY_MONTHS)
    assert (pd.to_datetime(ev["start_month"]) >= first).all()


def test_creep_events_are_visible_in_data(data):
    """Injected overspend should show up: mean spend after start > mean before."""
    t, ev = data["transactions"], data["injected_events"]
    t = t.assign(month=pd.to_datetime(t["txn_ts"]).dt.to_period("M").dt.to_timestamp())
    creeps = ev[ev.event_type == "category_creep"]
    if creeps.empty:
        pytest.skip("no creep events in this sample")
    hits = 0
    for e in creeps.itertuples():
        s = t[(t.customer_id == e.customer_id) & (t.true_category == e.category) & (t.type == "debit")]
        m = s.groupby("month").amount.sum()
        before, after = m[m.index < pd.Timestamp(e.start_month)], m[m.index >= pd.Timestamp(e.start_month)]
        hits += after.mean() > before.mean()
    assert hits / len(creeps) >= 0.7


def test_sms_links_back_to_transactions(data):
    sms, t = data["sms_raw"], data["transactions"]
    linked = sms.dropna(subset=["true_txn_id"])
    assert linked["true_txn_id"].isin(t["txn_id"]).all()
    assert (~sms["is_transaction"]).any(), "noise messages should exist"


def test_owners_pay_no_rent(data):
    c, t = data["customers"], data["transactions"]
    owners = c[c.owns_home].customer_id
    assert not ((t.customer_id.isin(owners)) & (t.true_category == "Rent")).any()
