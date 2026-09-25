# Limitations and Risks

- A graph layer (Innovation 3) may add complexity without improving the
  competition score; it is gated behind a mandatory ablation and is not
  enabled by default.
- Cross-source support can propagate errors if an incorrect S2 or S3
  candidate is treated as evidence for the other source -- this is why it is
  encoded as a *feature* for the classifier to weigh, not a merge rule.
- The MC-Dropout-style uncertainty estimate (`src/calibration.py`) can be
  computationally expensive over a large candidate set; the feature-dropout
  approximation used for the tree-based baseline is a lighter-weight
  stand-in and should be re-validated if the model is swapped for a neural
  scorer with real dropout layers.
- Source-specific thresholds can overfit if source-level validation samples
  are small (only a concern if the decision policy is extended beyond the
  current single global `theta1`).
- Country shift to France at test time may expose assumptions not visible
  in US/India training data; `src/normalization.py` and `src/blocking.py`
  are written to avoid any US/India-only hard-coding, but feature
  distributions learned on US/India data may still generalize imperfectly.
- Macro averaging can make rare query types (e.g. entities with unusually
  many true matches) disproportionately important to the final score.
- Candidate generation is the hard upper bound on achievable recall: no
  downstream component can recover a true match that no blocking pass ever
  proposed. `docs/blocking_strategy.md` and the marginal-recall accounting
  in `src/blocking.py` exist specifically to make this bound visible early.

## Compliance
No external business lookup, commercial ER API, government-registration
lookup, geocoding API, or internet-based business-data augmentation is used
anywhere in this codebase. `src/normalization.py` performs only local string
normalization; `src/blocking.py` and `src/features.py` use only the three
supplied TSVs.
