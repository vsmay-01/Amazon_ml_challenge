"""
model.py -- Gradient-boosted pairwise scorer.

This is the primary, always-on model (Candidate C's core, and the required
baseline for Candidates A/B). It is deliberately simple and fast, so that
hard negatives, cross-source support, calibration and the decision policy
can each be ablated against it (Section 8.1, Section 22).
"""
from __future__ import annotations

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier

from .features import FULL_FEATURE_COLUMNS


class PairwiseMatcher:
    def __init__(self, feature_columns: list[str] | None = None, random_state: int = 42):
        self.feature_columns = feature_columns or FULL_FEATURE_COLUMNS
        self.model = GradientBoostingClassifier(random_state=random_state)
        self._fitted = False

    def fit(self, training_table: pd.DataFrame) -> "PairwiseMatcher":
        X = training_table[self.feature_columns].fillna(0.0).values
        y = training_table["label"].values
        self.model.fit(X, y)
        self._fitted = True
        return self

    def predict_proba(self, feature_table: pd.DataFrame) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("PairwiseMatcher.fit() must be called before predict_proba().")
        cols = [c for c in self.feature_columns if c in feature_table.columns]
        missing = set(self.feature_columns) - set(cols)
        X = feature_table.reindex(columns=self.feature_columns, fill_value=0.0).values
        return self.model.predict_proba(X)[:, 1]

    def feature_importances(self) -> pd.Series:
        if not self._fitted:
            raise RuntimeError("Model not fitted.")
        return pd.Series(self.model.feature_importances_, index=self.feature_columns).sort_values(ascending=False)

    def save(self, path: str) -> None:
        joblib.dump({"model": self.model, "feature_columns": self.feature_columns}, path)

    @classmethod
    def load(cls, path: str) -> "PairwiseMatcher":
        payload = joblib.load(path)
        obj = cls(feature_columns=payload["feature_columns"])
        obj.model = payload["model"]
        obj._fitted = True
        return obj
