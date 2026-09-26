"""
model.py -- Incremental logistic pairwise scorer.

This is the primary, always-on model (Candidate C's core, and the required
baseline for Candidates A/B). It is deliberately simple and fast, so that
hard negatives, cross-source support, calibration and the decision policy
can each be ablated against it (Section 8.1, Section 22).
"""
from __future__ import annotations

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import SGDClassifier

from .features import FULL_FEATURE_COLUMNS


class PairwiseMatcher:
    def __init__(self, feature_columns: list[str] | None = None, random_state: int = 42):
        self.feature_columns = feature_columns or FULL_FEATURE_COLUMNS
        # SGDClassifier is the incremental equivalent needed for bounded-memory
        # training.  It exposes the same probability-scoring contract used by
        # the rest of the pipeline, while partial_fit consumes one batch at a time.
        self.model = SGDClassifier(
            loss="log_loss",
            random_state=random_state,
            average=True,
        )
        self._fitted = False

    def fit(self, training_table: pd.DataFrame) -> "PairwiseMatcher":
        return self.fit_batches([training_table])

    def fit_batches(self, training_batches, on_batch_end=None) -> "PairwiseMatcher":
        """Fit incrementally and optionally checkpoint after each fitted batch."""
        classes = np.array([0, 1], dtype=np.int64)
        saw_batch = False
        batch_number = 0
        for training_table in training_batches:
            if training_table.empty:
                continue
            X = training_table.reindex(columns=self.feature_columns, fill_value=0.0).fillna(0.0).values
            y = training_table["label"].to_numpy(dtype=np.int64)
            initialized = hasattr(self.model, "classes_")
            self.model.partial_fit(X, y, classes=None if initialized else classes)
            saw_batch = True
            batch_number += 1
            self._fitted = True
            del X, y, training_table
            if on_batch_end is not None:
                on_batch_end(batch_number)
        if not saw_batch:
            raise ValueError("No non-empty training batches were provided.")
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
        return pd.Series(np.abs(self.model.coef_[0]), index=self.feature_columns).sort_values(ascending=False)

    def save(self, path: str) -> None:
        joblib.dump({"model": self.model, "feature_columns": self.feature_columns}, path)

    @classmethod
    def load(cls, path: str) -> "PairwiseMatcher":
        payload = joblib.load(path)
        obj = cls(feature_columns=payload["feature_columns"])
        obj.model = payload["model"]
        obj._fitted = True
        return obj
