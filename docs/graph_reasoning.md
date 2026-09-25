# Graph Reasoning (Innovation 3, Experimental)

`src/graph_reasoning.py` implements a minimal multi-head Graph Attention
Network following Eq. 9-10 of the design doc, over the local per-query graph
`G_q = {q} + C2(q) + C3(q)` (Section 7.1).

## Status: optional, not wired into the default pipeline
This module:
- Requires `torch` (not in the default `requirements.txt`; uncomment the
  line there to enable it).
- Is never imported by `src/pipeline.py`.
- Exposes `GraphAttentionScorer` and `build_local_graph` so it can be
  exercised directly for the mandatory ablation (Section 8.1):

  1. Gradient-boosted model with pairwise features (`PairwiseMatcher` as-is)
  2. Same model plus cross-source support features (`PairwiseMatcher` with
     `add_cross_source_support` -- this is the current default)
  3. GAT without cross-source support
  4. GAT with cross-source support

## Initial hyperparameters (candidates, not requirements)
2 layers, up to 8 attention heads, hidden dim 128, output dim 64, LeakyReLU,
dropout ~0.3 (`GraphConfig` defaults).

## Retention criterion
The GAT is retained in the final pipeline only if variants 3/4 beat variant
2 by a reproducible, compute-justified margin on `reports/graph_ablation.csv`.
Given the challenge's 8B-parameter limit and the value of a fast, auditable
pipeline for judges, the default recommendation is to keep the GAT as a
documented experiment rather than a production dependency unless the ablation
result is unambiguous.
