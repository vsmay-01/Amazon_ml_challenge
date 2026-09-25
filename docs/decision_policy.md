# Query-Level 0/1/Many Decision Policy (Innovation 6)

Implemented in `src/decision.py`. The scoring objective rewards the best
*decision*, not the highest similarity, so this module is where
macro-F0.5 is actually optimized (Eq. 2).

## Default policy
`DecisionPolicy` accepts candidate i for query q iff:
- `calibrated_score_i > theta1`, **and**
- the top two candidates are not both below `theta_high` while closer than
  `theta_margin` to each other (an ambiguous top cluster demands stronger
  evidence and, by default, accepts neither -- protecting singleton
  precision, which the metric weights heavily since precision is 2x recall).

| Scenario | Example scores | Desired behavior | Policy outcome |
|---|---|---|---|
| One dominant candidate | 0.97, 0.21, 0.11 | Accept if calibrated | top score clears `theta1` and margin, accepted |
| Ambiguous cluster | 0.72, 0.70, 0.68 | Demand stronger evidence or abstain | top-two-close rule fires, none accepted |
| Weak candidate set | 0.42, 0.31, 0.24 | Likely predict empty | below `theta1`, none accepted |
| Several strong candidates | 0.96, 0.94, 0.93 | Allow multiple matches | all clear `theta1`, top-two-close rule does not fire since both are `>= theta_high` |

## Threshold selection
`src/decision.py::sweep_theta1` sweeps `theta1` over the leakage-safe
validation split and reports macro-F0.5 for each value
(`reports/threshold_experiments.csv`); `src/pipeline.py::cmd_train` picks the
best value automatically and persists it in the saved model bundle.

## Query-context features
`rank`, `cand_count`, `score_dispersion`, `margin_to_next` are computed per
query group so a future, richer policy (e.g. a small logistic model over
these features instead of fixed thresholds) can be swapped in without
touching any upstream module.
