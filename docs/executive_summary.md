# Executive Summary

TriVote-ER++ is an experimentally driven architecture for the Amazon Business
Entity Resolution Challenge, combining:

1. **Multi-pass candidate generation** to preserve candidate recall while
   controlling the comparison space.
2. **Triadic evidence fusion** that treats Source 1, Source 2 and Source 3 as
   a three-source evidence system rather than independent pairs.
3. **Hard-negative-aware supervised learning** to separate true matches from
   highly similar non-matches.
4. **Calibrated, query-level selective decisions** that can output zero, one,
   or many matches and abstain when evidence is insufficient.

The architecture is a research ladder, not a large model chosen for
appearance. A strong classical baseline is built first; each advanced
component (cross-source support, GAT, MC-Dropout uncertainty) is retained
only if it improves validation macro-F0.5 under a leakage-safe protocol.

**Central hypothesis:** a multi-source entity matcher can improve
precision-heavy macro-F0.5 by treating cross-source agreement and
query-level uncertainty as evidence for whether a candidate should be
accepted, while hard negatives focus the model on the ambiguous candidates
that dominate false merges.

No claim of guaranteed novelty, state-of-the-art performance, or a win is
made anywhere in this project. Those claims must be established by the
experiments in `reports/`.
