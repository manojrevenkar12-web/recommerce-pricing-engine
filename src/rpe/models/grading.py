"""What condition arrives: P(like new, very good, good, acceptable, reject) for a title, from the
goods-receipt history. Rejected items cost handling and are not paid; the grade sets the price."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from rpe.calendar import to_date
from rpe.catalog import CATEGORIES, Catalog

GRADE_CODES = {"like_new": 0, "very_good": 1, "good": 2, "acceptable": 3, "reject": 4}


def _X(cat: Catalog, idx: np.ndarray, year: np.ndarray) -> np.ndarray:
    age = np.clip(year - cat.pub_year[idx], 0, 60)
    return np.column_stack([np.eye(len(CATEGORIES))[cat.category[idx]], age / 10.0, (age / 10.0) ** 2, cat.fmt[idx] == 0])


@dataclass
class GradingModel:
    lr: LogisticRegression

    @classmethod
    def fit(cls, cat: Catalog, receipts: pd.DataFrame) -> GradingModel:
        idx = receipts.idx.to_numpy()
        year = np.array([to_date(d).year for d in receipts.day.to_numpy()])
        lr = LogisticRegression(max_iter=2000, C=10.0)
        lr.fit(_X(cat, idx, year), receipts.grade.map(GRADE_CODES).to_numpy())
        return cls(lr)

    def probs(self, cat: Catalog, idx: np.ndarray, day: int) -> np.ndarray:
        """(n, 5) probabilities, columns like_new..acceptable, reject."""
        p = self.lr.predict_proba(_X(cat, idx, np.full(len(idx), to_date(day).year)))
        out = np.zeros((len(idx), 5))
        out[:, self.lr.classes_] = p
        return out
