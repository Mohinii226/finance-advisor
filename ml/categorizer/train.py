"""
Train + evaluate the expense categoriser with a strict TIME split.

    python -m ml.categorizer.train --data-dir data/raw --out models/categorizer

Split (debits only)
  train : labeled rows before --train-end           (default 2024-10-01), customers in the "warm" set only
  val   : Q4 2024                                    -> used ONLY to pick the confidence threshold
  test  : 2025                                       -> reported; never used for any choice
  "unseen customers": a random 20% of customers never appear in training (generalisation to new users).
Labels used for training are the noisy bank/user `category`; evaluation uses the clean `true_category`.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

from .evaluate import (apply_threshold, per_class, pick_threshold, plot_confusion, selective_curve, summary,
                       top_confusions)
from .features import STRUCT_COLUMNS, history_features
from .model import ExpenseCategorizer


def time_split(d: pd.DataFrame, train_end: str, val_end: str, cold: set):
    is_cold = d["customer_id"].isin(cold)
    train = d[(d.txn_ts < train_end) & d.category.notna() & ~is_cold]
    val = d[(d.txn_ts >= train_end) & (d.txn_ts < val_end) & ~is_cold]
    test = d[d.txn_ts >= val_end]
    assert train.txn_ts.max() < val.txn_ts.min() and val.txn_ts.max() < test.txn_ts.min(), "time leakage"
    return train, val, test


def _stage2_ablation(model: ExpenseCategorizer, train, test, H, y_train) -> dict:
    """Cheap ablation: re-fit ONLY stage 2 on shared stage-1 outputs with different feature groups."""
    Ftr = model._stage2_frame(model.oof_, train, H)
    Fte = model._stage2_frame(model._stage1_proba(model._clean(test)), test, H)
    P = [c for c in Ftr.columns if c.startswith("p_")]
    groups = {"stage 2 on text probabilities only": P, "+ amount / channel / time features": P + STRUCT_COLUMNS,
              "+ payee history (full model)": list(Ftr.columns)}
    out = {}
    for name, cols in groups.items():
        cat = [cols.index("channel_code")] if "channel_code" in cols else None
        m = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.1, max_leaf_nodes=31, min_samples_leaf=40,
                                           l2_regularization=1.0, early_stopping=True, validation_fraction=0.1,
                                           n_iter_no_change=20, categorical_features=cat, random_state=0)
        m.fit(Ftr[cols], y_train)
        pred = np.array(model.classes_)[m.predict_proba(Fte[cols]).argmax(1)]
        out[name] = summary(test.true_category.to_numpy(), pred)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data/raw")
    ap.add_argument("--out", default="models/categorizer")
    ap.add_argument("--train-end", default="2024-10-01")
    ap.add_argument("--val-end", default="2025-01-01")
    ap.add_argument("--cold-frac", type=float, default=0.2)
    ap.add_argument("--target-accuracy", type=float, default=0.98, help="accuracy required of auto-accepted predictions (must exceed overall accuracy to flag anything)")
    ap.add_argument("--stage2", default="auto", choices=["auto", "xgboost", "hgb"])
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)

    t = pd.read_csv(Path(a.data_dir) / "transactions.csv", parse_dates=["txn_ts"])
    income = pd.read_csv(Path(a.data_dir) / "customers.csv").set_index("customer_id")["monthly_income"]
    d = t[t.type == "debit"].sort_values(["txn_ts", "txn_id"]).set_index("txn_id", drop=False)

    rng = np.random.default_rng(a.seed)
    customers = np.array(sorted(d.customer_id.unique()))
    cold = set(rng.choice(customers, int(len(customers) * a.cold_frac), replace=False))
    train, val, test = time_split(d, a.train_end, a.val_end, cold)
    classes = sorted(train.category.unique())
    print(f"train {len(train):,} | val {len(val):,} | test {len(test):,} | classes {len(classes)} | unseen customers {len(cold)}", flush=True)

    H = history_features(d, classes, "category")                      # strictly past-only, uses noisy labels only
    model = ExpenseCategorizer(stage2=a.stage2, random_state=a.seed).fit(train, train.category, income, extra=H)
    print(f"stage 2 = {model.stage2_kind}; timings {({k: round(v) for k, v in model.timings_.items()})}", flush=True)

    # ---- confidence threshold from VALIDATION only ----
    pv = model.predict(val, H)
    vconf, vok = pv.confidence.to_numpy(), (pv.predicted_category == val.true_category).to_numpy()
    targets = sorted({0.97, 0.98, 0.99, a.target_accuracy})
    thresholds = {t_: pick_threshold(vconf, vok, t_) for t_ in targets}
    thr = thresholds[a.target_accuracy]

    # ---- TEST report ----
    pt = model.predict(test, H)
    yt, pred, conf = test.true_category.to_numpy(), pt.predicted_category.to_numpy(), pt.confidence.to_numpy()
    correct = pred == yt
    is_cold, blank = test.customer_id.isin(cold).to_numpy(), test.category.isna().to_numpy()
    majority = train.category.mode()[0]
    results = {
        "config": {"stage2": model.stage2_kind, "train_end": a.train_end, "val_end": a.val_end, "target_accuracy": a.target_accuracy},
        "baseline_majority_class": summary(yt, np.full(len(yt), majority)),
        "baseline_text_only": summary(yt, model.predict_text_only(test)),
        "full_model": summary(yt, pred),
        "full_model_seen_customers": summary(yt[~is_cold], pred[~is_cold]),
        "full_model_unseen_customers": summary(yt[is_cold], pred[is_cold]),
        "full_model_blank_label_rows": summary(yt[blank], pred[blank]),
        "review_queue": {f"target_{t_:.2f}": apply_threshold(conf, correct, th) for t_, th in thresholds.items()},
        "review_queue_primary_target": a.target_accuracy,
        "ablation": _stage2_ablation(model, train, test, H, model.le.transform(train.category.to_numpy())),
    }
    (out / "metrics.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    per_class(yt, pred).to_csv(out / "per_class.csv")
    top_confusions(yt, pred).to_csv(out / "top_confusions.csv", index=False)
    selective_curve(conf, correct).to_csv(out / "selective_curve.csv", index=False)
    plot_confusion(yt, pred, str(out / "confusion_matrix.png"))
    joblib.dump(dict(model=model, threshold=thr, classes=classes), out / "categorizer.joblib")
    print(json.dumps({k: v for k, v in results.items() if k != "config"}, indent=2))


if __name__ == "__main__":
    main()
