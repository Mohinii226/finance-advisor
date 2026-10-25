import numpy as np
import pandas as pd

from data_gen.generate import build_dataset
from ml.categorizer.evaluate import apply_threshold, pick_threshold
from ml.categorizer.features import clean_description, history_features, structured_features
from ml.categorizer.model import ExpenseCategorizer
from ml.categorizer.train import time_split


def test_clean_description_removes_refs_and_handles():
    c = clean_description("UPI-SWIGGY-swiggy@icici-123456789012-UPI")
    assert c == "upi swiggy swiggy upi"
    assert "8h2k9x1q" not in clean_description("ATM WDL 8H2K9X1Q MUMBAI")


def _toy():
    return pd.DataFrame({
        "customer_id": ["A", "A", "A", "A", "B"],
        "raw_description": ["UPI-RAMESH K-r@ybl"] * 3 + ["UPI-OTHER PERSON-o@ybl", "UPI-RAMESH K-r@ybl"],
        "txn_ts": pd.to_datetime(["2024-01-01", "2024-02-01", "2024-03-02", "2024-03-05", "2024-04-02"]),
        "amount": [8000, 8000, 8050, 300, 8000], "category": ["Rent", "Rent", None, "Transfers", "Groceries"]})


def test_history_features_are_leak_free():
    """A row's history features must not change if FUTURE rows are removed or altered."""
    cls = ["Groceries", "Rent", "Transfers"]
    df = _toy()
    full = history_features(df, cls)
    # (1) dropping the future rows changes nothing for the earlier rows
    pd.testing.assert_frame_equal(full.iloc[:3], history_features(df.iloc[:3], cls), check_exact=False)
    # (2) rewriting amounts and labels of rows 3+ (strictly later in time) changes nothing for rows 0-2
    future = df.copy()
    future.loc[3:, "amount"] = 1.0
    future.loc[3:, "category"] = "Groceries"
    pd.testing.assert_frame_equal(full.iloc[:3], history_features(future, cls).iloc[:3], check_exact=False)
    # (3) a row's OWN label never feeds its own memory features
    own = df.copy()
    own.loc[1, "category"] = "Groceries"
    pd.testing.assert_series_equal(full.loc[1], history_features(own, cls).loc[1], check_exact=False)
    # sanity on the values themselves
    assert full.loc[2, "payee_n_prior"] == 2 and full.loc[2, "mem_Rent"] == 1.0 and full.loc[2, "payee_monthly_recur"] == 1.0
    assert full.loc[4, "payee_n_prior"] == 0, "same text for a DIFFERENT customer must not leak"


def test_history_never_reads_true_category():
    df = _toy().assign(true_category="SHOULD_NOT_BE_USED")
    h = history_features(df, ["Groceries", "Rent", "Transfers"])
    assert not any("SHOULD_NOT" in c for c in h.columns)


def test_amount_features_are_inflation_deflated():
    base = pd.DataFrame({"customer_id": ["A", "A"], "amount": [1000.0, 1000.0 * 1.06 ** 2], "channel": ["UPI", "UPI"],
                         "txn_ts": pd.to_datetime(["2023-06-01 12:00", "2025-06-01 12:00"]), "raw_description": ["x", "x"]})
    f = structured_features(base, pd.Series({"A": 50000.0}))
    assert abs(f.log_real_amount.iloc[0] - f.log_real_amount.iloc[1]) < 0.01


def test_threshold_selection_hits_target_and_is_monotone():
    rng = np.random.default_rng(0)
    conf = rng.random(5000)
    correct = rng.random(5000) < (0.5 + 0.5 * conf)             # more confident -> more often right
    thr = pick_threshold(conf, correct, 0.9)
    r = apply_threshold(conf, correct, thr)
    assert r["accuracy_accepted"] >= 0.9 and 0 < r["coverage"] < 1
    assert apply_threshold(conf, correct, thr + 0.1)["coverage"] < r["coverage"]


def test_end_to_end_beats_majority_baseline_with_time_split():
    d = build_dataset(25, 36, seed=3, sms_frac=0.0)
    t = d["transactions"].copy(); t["txn_ts"] = pd.to_datetime(t["txn_ts"])
    income = d["customers"].set_index("customer_id")["monthly_income"]
    deb = t[t.type == "debit"].sort_values(["txn_ts", "txn_id"]).set_index("txn_id", drop=False)
    train, val, test = time_split(deb, "2024-10-01", "2025-01-01", cold=set())
    assert train.txn_ts.max() < test.txn_ts.min()
    classes = sorted(train.category.unique())
    H = history_features(deb, classes, "category")
    m = ExpenseCategorizer(stage2="hgb", n_folds=3).fit(train, train.category, income, extra=H)
    pr = m.predict(test, H)
    acc = (pr.predicted_category.to_numpy() == test.true_category.to_numpy()).mean()
    majority = (test.true_category == train.category.mode()[0]).mean()
    assert acc > 0.85 and acc > majority + 0.5, (acc, majority)
    assert pr.confidence.between(0, 1).all() and set(pr.predicted_category) <= set(classes)
