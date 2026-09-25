# Architecture Candidates

## Candidate A: Triadic Evidence Graph
Multi-pass blocking + pairwise evidence features + S2-S3 support edges +
optional GAT + calibration/selective decision policy.
**Hypothesis:** graph context improves discrimination in ambiguous candidate
sets. Highest complexity; gated entirely behind the GAT ablation
(`docs/graph_reasoning.md`).

## Candidate B: Selective MatchSet
Multi-pass blocking + pairwise scorer + hard-negative training + query-level
candidate-set model + calibrated zero/one/many decision.
**Hypothesis:** query context and hard negatives improve precision on
ambiguous candidate sets without graph-network complexity. This is the
architecture implemented by default in `src/pipeline.py`.

## Candidate C: AdaptiveBlock ER
Strong multi-view retrieval + lightweight gradient-boosted matcher +
source-aware thresholds, no graph network unless experiments demand it.
**Hypothesis:** most measurable gain comes from candidate recall and careful
decision control, making a simpler system preferable.

## How the codebase maps to these candidates
`src/pipeline.py` builds Candidate C by default (blocking -> features ->
gradient-boosted model -> calibration -> decision policy). Adding
`add_cross_source_support` features (already on by default) plus a decision
policy sweep moves it toward Candidate B. Wiring in `src/graph_reasoning.py`
(optional, requires `torch`) and re-running the ablation in
`docs/ablation_plan.md` / Section 22 of the design doc is what would promote
the system to Candidate A -- only if the ablation shows a reproducible gain.
