"""Evaluation helpers: metrics, selective accuracy ("needs review" threshold), confusion matrix."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, f1_score


def summary(y_true, y_pred) -> dict:
    return dict(n=int(len(y_true)), accuracy=float(accuracy_score(y_true, y_pred)),
                macro_f1=float(f1_score(y_true, y_pred, average="macro")),
                weighted_f1=float(f1_score(y_true, y_pred, average="weighted")))


def per_class(y_true, y_pred) -> pd.DataFrame:
    r = pd.DataFrame(classification_report(y_true, y_pred, output_dict=True, zero_division=0)).T
    return r.drop(["accuracy", "macro avg", "weighted avg"], errors="ignore").rename(columns={"f1-score": "f1"}).round(4)


def top_confusions(y_true, y_pred, k: int = 10) -> pd.DataFrame:
    m = pd.DataFrame({"true": np.asarray(y_true), "pred": np.asarray(y_pred)})
    m = m[m.true != m.pred].groupby(["true", "pred"]).size().rename("count").reset_index()
    return m.sort_values("count", ascending=False).head(k)


def selective_curve(conf: np.ndarray, correct: np.ndarray, n_points: int = 50) -> pd.DataFrame:
    """Coverage (share auto-accepted) vs accuracy of the accepted predictions, for confidence thresholds."""
    order = np.argsort(-conf)
    c, ok = conf[order], correct[order].astype(float)
    cum_acc = np.cumsum(ok) / (np.arange(len(ok)) + 1)
    idx = np.unique(np.linspace(0, len(c) - 1, n_points).astype(int))
    return pd.DataFrame({"threshold": c[idx], "coverage": (idx + 1) / len(c), "accuracy": cum_acc[idx]})


def pick_threshold(conf: np.ndarray, correct: np.ndarray, target_accuracy: float = 0.95) -> float:
    """Lowest threshold whose accepted predictions reach `target_accuracy` (max coverage). 1.0 if unreachable."""
    order = np.argsort(-conf)
    c, ok = conf[order], correct[order].astype(float)
    cum_acc = np.cumsum(ok) / (np.arange(len(ok)) + 1)
    good = np.where(cum_acc >= target_accuracy)[0]
    return float(c[good.max()]) if len(good) else 1.0


def apply_threshold(conf: np.ndarray, correct: np.ndarray, thr: float) -> dict:
    acc_mask = conf >= thr
    return dict(threshold=thr, coverage=float(acc_mask.mean()), review_rate=float(1 - acc_mask.mean()),
                accuracy_accepted=float(correct[acc_mask].mean()) if acc_mask.any() else float("nan"),
                accuracy_review_queue=float(correct[~acc_mask].mean()) if (~acc_mask).any() else float("nan"))


def plot_confusion(y_true, y_pred, path: str, top: int = 22) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    labels = sorted(set(y_true) | set(y_pred))
    cm = pd.crosstab(pd.Categorical(y_true, labels), pd.Categorical(y_pred, labels), dropna=False).to_numpy().astype(float)
    cm = cm / np.maximum(cm.sum(1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(10, 8.5))
    im = ax.imshow(cm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(labels)), labels, rotation=90, fontsize=7)
    ax.set_yticks(range(len(labels)), labels, fontsize=7)
    ax.set_xlabel("predicted"), ax.set_ylabel("true"), ax.set_title("Row-normalised confusion matrix (test)")
    fig.colorbar(im, fraction=0.04)
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)
