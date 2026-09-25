# Model Design

## Baseline / primary model
`src/model.py::PairwiseMatcher` wraps a `GradientBoostingClassifier` over
the pair-level feature table (`src/features.py::FULL_FEATURE_COLUMNS`):
name family, address family, country family, source-indicator family,
missingness family, and cross-source support family (Section 12).

This is deliberately the *only* always-on scorer. It is fast to train,
interpretable via feature importances, and gives every other component
(hard negatives, cross-source support, calibration, decision policy, GAT) a
fixed point to be ablated against (Section 8.1, Section 22).

## Hard-negative-aware training
`src/hard_negatives.py` labels the real post-blocking candidate distribution
against ground truth, splits negatives into easy/hard using similarity
thresholds correlated with false merges (high name/address similarity but
wrong identity), and builds a training set with a configurable
hard-negative ratio. This directly targets the "ambiguous candidates that
dominate false merges" from the central hypothesis in the executive summary.

An InfoNCE-style contrastive objective (Eq. 11 of the design doc) is a
documented *optional* extension to `PairwiseMatcher` -- not implemented by
default, since the gradient-boosted classifier already benefits from
hard-negative-weighted training data without requiring a different loss
function or embedding head. This tradeoff should be revisited only if
`reports/model_experiments.csv` shows the boosted-tree model plateauing.

## Cross-source (triadic) support
`src/features.py::add_cross_source_support` computes, for each S1<->S2
candidate, the strongest independent S2<->S3 similarity among the other S3
candidates proposed for the same S1 query (and symmetrically for S1<->S3
candidates). This is used as *evidence* -- a feature the boosted-tree model
can learn to weight -- never as an unconditional merge rule, per Section 7.3.

## Optional GAT branch
See `docs/graph_reasoning.md`.
