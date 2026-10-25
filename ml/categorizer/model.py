"""
Two-stage expense categoriser.

Stage 1  TF-IDF (char n-grams + word n-grams) on the cleaned description -> linear model (SGD, log loss).
         Char n-grams make it robust to typos, truncation and aggregator prefixes (RAZORPAY*, PTM*).
         Out-of-fold class probabilities are produced for the training rows (stacking without leakage).
Stage 2  Gradient boosting on [stage-1 probabilities + amount / channel / time / amount-to-income features].
         Resolves cases text cannot (Amazon vs Amazon Prime, "Swiggy" vs "Swiggy Instamart" truncations, rent vs groceries).
         XGBoost is used when installed; otherwise scikit-learn's HistGradientBoostingClassifier.

Output: predicted category + confidence (max probability) used for the "needs review" queue.
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder

from .features import CATEGORICAL_STRUCT, STRUCT_COLUMNS, clean_description, structured_features


def _xgboost_available() -> bool:
    try:
        import xgboost  # noqa: F401
        return True
    except Exception:
        return False


class ExpenseCategorizer:
    def __init__(self, stage2: str = "auto", n_folds: int = 5, use_structured: bool = True, alpha: float = 2e-6,
                 char_ngrams: tuple = (2, 5), random_state: int = 0, max_iter2: int = 300,
                 class_weight=None, s2_params: dict | None = None):
        self.stage2_kind = ("xgboost" if _xgboost_available() else "hgb") if stage2 == "auto" else stage2
        self.n_folds, self.use_structured, self.alpha = n_folds, use_structured, alpha
        self.char_ngrams, self.random_state, self.max_iter2 = char_ngrams, random_state, max_iter2
        self.class_weight, self.s2_params = class_weight, dict(s2_params or {})
        self.le = LabelEncoder()

    # ---------------------------------------------------------------- text
    def _clean(self, df: pd.DataFrame) -> np.ndarray:
        return df["raw_description"].map(clean_description).to_numpy()

    def _fit_vectorizers(self, cleaned: np.ndarray):
        uniq, inv = np.unique(cleaned, return_inverse=True)            # vectorise unique strings only (fast)
        self.char_vec = TfidfVectorizer(analyzer="char_wb", ngram_range=self.char_ngrams, min_df=3,
                                        sublinear_tf=True, dtype=np.float32)
        self.word_vec = TfidfVectorizer(analyzer="word", ngram_range=(1, 2), min_df=2, sublinear_tf=True,
                                        token_pattern=r"[a-z0-9]+", dtype=np.float32)
        Xu = sp.hstack([self.char_vec.fit_transform(uniq), self.word_vec.fit_transform(uniq)]).tocsr()
        return Xu, inv

    def _text_matrix(self, cleaned: np.ndarray):
        uniq, inv = np.unique(cleaned, return_inverse=True)
        return sp.hstack([self.char_vec.transform(uniq), self.word_vec.transform(uniq)]).tocsr(), inv

    def _stage1_proba(self, cleaned: np.ndarray) -> np.ndarray:
        Xu, inv = self._text_matrix(cleaned)
        return self.stage1.predict_proba(Xu)[inv]            # predict once per unique string, then expand

    def _new_stage1(self):
        # Plain SGD on the exact rows. (Weighted unique-pair training was ~4x faster but under-converged:
        # 85.9% -> 93.1% accuracy once reverted. Averaged SGD recovered most but not all of it.)
        return SGDClassifier(loss="log_loss", alpha=self.alpha, max_iter=20, tol=None, n_jobs=-1,
                             random_state=self.random_state)

    # ---------------------------------------------------------------- stage 2 plumbing
    def _stage2_frame(self, proba: np.ndarray, df: pd.DataFrame, extra: pd.DataFrame | None = None) -> pd.DataFrame:
        parts = [pd.DataFrame(proba, columns=[f"p_{c}" for c in self.classes_], index=df.index)]
        if self.use_structured:
            parts.append(structured_features(df.assign(clean_text=self._clean(df)), self.income_))
        if extra is not None:
            parts.append(extra.loc[df.index])
        F = pd.concat(parts, axis=1)
        if hasattr(self, "stage2_columns_"):
            F = F.reindex(columns=self.stage2_columns_)
        return F

    def _new_stage2(self, n_classes: int, columns: list[str]):
        if self.stage2_kind == "xgboost":
            from xgboost import XGBClassifier
            return XGBClassifier(n_estimators=400, learning_rate=0.1, max_depth=6, subsample=0.8, colsample_bytree=0.8,
                                 tree_method="hist", objective="multi:softprob", n_jobs=-1,
                                 early_stopping_rounds=20, random_state=self.random_state)
        cat_idx = [columns.index(c) for c in CATEGORICAL_STRUCT if c in columns] or None
        params = dict(max_iter=self.max_iter2, learning_rate=0.1, max_leaf_nodes=31, min_samples_leaf=40,
                      l2_regularization=1.0, early_stopping=True, validation_fraction=0.1, n_iter_no_change=20,
                      class_weight=self.class_weight)
        params.update(self.s2_params)
        return HistGradientBoostingClassifier(categorical_features=cat_idx, random_state=self.random_state, **params)

    # ---------------------------------------------------------------- public API
    def fit(self, df: pd.DataFrame, labels: pd.Series, income: pd.Series | None = None,
            extra: pd.DataFrame | None = None) -> "ExpenseCategorizer":
        self.income_, self.timings_ = income, {}
        t0 = time.time()
        y = self.le.fit_transform(labels.to_numpy())
        self.classes_ = list(self.le.classes_)
        cleaned = self._clean(df)
        Xu, tid = self._fit_vectorizers(cleaned)
        self.timings_["tfidf"] = time.time() - t0

        # stage 1: out-of-fold probabilities for the training rows
        K = len(self.classes_)
        X = Xu[tid]
        skf = StratifiedKFold(self.n_folds, shuffle=True, random_state=self.random_state)
        oof = np.zeros((len(y), K))
        for tr, te in skf.split(np.zeros(len(y)), y):
            clf = self._new_stage1().fit(X[tr], y[tr])
            tids = np.unique(tid[te])                      # predict once per unique string, then expand
            oof[te] = clf.predict_proba(Xu[tids])[np.searchsorted(tids, tid[te])]
        self.stage1 = self._new_stage1().fit(X, y)
        self.oof_ = oof
        self.timings_["stage1_oof"] = time.time() - t0 - self.timings_["tfidf"]

        # stage 2
        F = self._stage2_frame(oof, df, extra)
        self.stage2_columns_ = list(F.columns)
        self.stage2 = self._new_stage2(len(self.classes_), self.stage2_columns_)
        if self.stage2_kind == "xgboost":
            rng = np.random.default_rng(self.random_state)
            val = rng.random(len(F)) < 0.1
            self.stage2.fit(F[~val], y[~val], eval_set=[(F[val], y[val])], verbose=False)
        else:
            self.stage2.fit(F, y)
        self.timings_["total"] = time.time() - t0
        return self

    def predict_proba(self, df: pd.DataFrame, extra: pd.DataFrame | None = None) -> np.ndarray:
        return self.stage2.predict_proba(self._stage2_frame(self._stage1_proba(self._clean(df)), df, extra))

    def predict_text_only(self, df: pd.DataFrame) -> np.ndarray:
        """Stage-1 predictions only (used as an ablation baseline)."""
        return np.array(self.classes_)[self._stage1_proba(self._clean(df)).argmax(1)]

    def predict(self, df: pd.DataFrame, extra: pd.DataFrame | None = None) -> pd.DataFrame:
        proba = self.predict_proba(df, extra)
        return pd.DataFrame({"predicted_category": np.array(self.classes_)[proba.argmax(1)],
                             "confidence": proba.max(1)}, index=df.index)
