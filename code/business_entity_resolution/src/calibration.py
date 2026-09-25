"""
calibration.py -- Score calibration + uncertainty estimation
(Innovation 5, Section 10).

Two independent mechanisms:
  1. `calibrate_scores`: Platt scaling / isotonic regression to turn raw
     model scores into calibrated probabilities (this always applies, is
     cheap, and is what most of macro-F0.5 gain typically comes from).
  2. `monte_carlo_uncertainty`: MC-Dropout-style variance/entropy estimate,
     implemented generically over *any* stochastic scoring function so it
     works whether the underlying model is the gradient-boosted baseline
     (via bootstrap re-scoring) or, if enabled, the GAT branch (via real
     dropout at inference time). This is optional and must earn its place
     in the ablation, same as the GAT branch.
"""
from __future__ import annotations

from typing import Callable

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression


class ScoreCalibrator:
    def __init__(self, method: str = "isotonic"):
        assert method in ("isotonic", "platt")
        self.method = method
        self._model = None

    def fit(self, raw_scores: np.ndarray, labels: np.ndarray) -> "ScoreCalibrator":
        if self.method == "isotonic":
            self._model = IsotonicRegression(out_of_bounds="clip")
            self._model.fit(raw_scores, labels)
        else:
            self._model = LogisticRegression()
            self._model.fit(raw_scores.reshape(-1, 1), labels)
        return self

    def transform(self, raw_scores: np.ndarray) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("ScoreCalibrator must be fit() before transform().")
        if self.method == "isotonic":
            return self._model.predict(raw_scores)
        return self._model.predict_proba(raw_scores.reshape(-1, 1))[:, 1]


def monte_carlo_uncertainty(
    score_fn: Callable[[], np.ndarray],
    n_samples: int = 20,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generic MC-Dropout-style estimator (Eq. 12-14). `score_fn` should return
    a fresh array of scores each call, with its own internal stochasticity
    (e.g. dropout at inference, or bootstrap resampling for tree ensembles).

    Returns (mean, variance, entropy), each shape [n_pairs].
    """
    samples = np.stack([score_fn() for _ in range(n_samples)], axis=0)  # [M, N]
    mu = samples.mean(axis=0)
    var = samples.var(axis=0)
    eps = 1e-9
    mu_clip = np.clip(mu, eps, 1 - eps)
    entropy = -mu_clip * np.log(mu_clip) - (1 - mu_clip) * np.log(1 - mu_clip)
    return mu, var, entropy


def bootstrap_score_fn(
    model,
    feature_matrix: np.ndarray,
    dropout_frac: float = 0.1,
    random_state: int | None = None,
) -> Callable[[], np.ndarray]:
    """
    Lightweight stand-in for MC-Dropout that works with the tree-based
    baseline model: randomly zero out a fraction of feature columns per
    call (feature-level dropout) rather than requiring a torch dropout
    layer. This lets the uncertainty-guided abstention policy (Innovation 5)
    be exercised end-to-end without requiring the optional GAT branch.
    """
    rng = np.random.default_rng(random_state)
    n_features = feature_matrix.shape[1]

    def _sample() -> np.ndarray:
        mask = rng.random(n_features) >= dropout_frac
        masked = feature_matrix * mask
        return model.model.predict_proba(masked)[:, 1]

    return _sample
