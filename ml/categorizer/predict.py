"""
Inference wrapper + database writer.

    from ml.categorizer.predict import Categorizer
    cat = Categorizer.load("models/categorizer")
    out = cat.predict(new_txns, history=past_txns)        # -> predicted_category, confidence, needs_review

CLI (uncategorised debits in Postgres -> predicted_category / confidence columns):
    python -m ml.categorizer.predict --dir models/categorizer
"""
from __future__ import annotations

import argparse
import os

import joblib
import pandas as pd

from .features import history_features


class Categorizer:
    def __init__(self, model, threshold: float, classes: list[str]):
        self.model, self.threshold, self.classes = model, threshold, classes

    @classmethod
    def load(cls, directory: str) -> "Categorizer":
        b = joblib.load(os.path.join(directory, "categorizer.joblib"))
        return cls(b["model"], b["threshold"], b["classes"])

    def predict(self, new: pd.DataFrame, history: pd.DataFrame | None = None) -> pd.DataFrame:
        """`new`: debit rows to label. `history`: the same customers' earlier rows (with their `category` labels)
        so payee-recurrence / own-label-memory features can be computed. Only strictly earlier rows are ever used."""
        new = new.set_index("txn_id", drop=False)
        past = history.set_index("txn_id", drop=False) if history is not None else new.iloc[0:0]
        allrows = pd.concat([past, new]).sort_values(["txn_ts", "txn_id"])
        H = history_features(allrows, self.classes, "category")
        out = self.model.predict(new, H)
        out["needs_review"] = out["confidence"] < self.threshold
        return out


def main() -> None:
    from sqlalchemy import create_engine, text   # lazy import

    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="models/categorizer")
    a = ap.parse_args()
    cat = Categorizer.load(a.dir)
    eng = create_engine(os.getenv("DATABASE_URL", "postgresql+psycopg2://finance:finance@localhost:5432/finance_advisor"))
    debits = pd.read_sql("SELECT txn_id, customer_id, txn_ts, amount, channel, raw_description, category "
                         "FROM transactions WHERE type = 'debit' ORDER BY txn_ts, txn_id", eng, parse_dates=["txn_ts"])
    income = pd.read_sql("SELECT customer_id, monthly_income FROM customers", eng).set_index("customer_id")["monthly_income"]
    cat.model.income_ = income
    todo = debits[debits.category.isna()]
    pred = cat.predict(todo, history=debits[debits.category.notna()])
    with eng.begin() as con:
        con.execute(text("UPDATE transactions SET predicted_category = :p, confidence = :c WHERE txn_id = :i"),
                    [dict(p=r.predicted_category, c=float(r.confidence), i=int(i)) for i, r in pred.iterrows()])
    print(f"labelled {len(pred):,} transactions; {int(pred.needs_review.sum()):,} flagged for review "
          f"(threshold {cat.threshold:.3f})")


if __name__ == "__main__":
    main()
