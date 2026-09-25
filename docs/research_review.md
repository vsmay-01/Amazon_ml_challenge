# Research Review

Short summaries of the literature that motivates (but does not guarantee)
each experimental component. Every claim below is a hypothesis to be tested
against this challenge's own validation data, not an assumed result.

- **Blocking surveys** [1, 2] describe blocking as the principal mechanism
  for reducing the quadratic comparison space in entity resolution, and
  document the limits of rigid, fixed-schema blocking assumptions in
  heterogeneous data. Motivates `src/blocking.py`'s multi-pass, multi-view
  design instead of a single blocking key.
- **WDC Products** [3] reports that contrastive learning can be more
  training-data efficient than cross-encoder entity matching under its
  benchmark conditions. Motivates the InfoNCE-style hard-negative sampling
  objective described in Section 9 of the design doc (implemented as
  sampling logic in `src/hard_negatives.py`; the contrastive loss itself is
  an optional training objective for `src/model.py`).
- **Selective entity matching with contrastive margin ranking** [4] targets
  semantically similar hard negatives specifically, reinforcing the
  hard-negative mining strategy.
- **Confidence calibration for LLM-based entity matching** [5] evaluates
  temperature scaling, MC-Dropout and ensembles, and reports reduced
  calibration error after calibration on its studied datasets. Motivates
  `src/calibration.py`, though our objective is validation macro-F0.5, not
  minimum calibration error per se.
- **Contextual-semantics graph attention for entity resolution** [6]
  motivates testing a GAT for relational context pairwise models miss, but
  does not establish that a GAT will beat a boosted-tree feature model on
  this specific challenge -- hence the mandatory ablation gating
  `src/graph_reasoning.py`.

## References
[1] Li et al., "A Survey on Blocking Technology of Entity Resolution,"
    J. Comput. Sci. Technol. 35(4), 2020.
[2] Papadakis et al., "Blocking and Filtering Techniques for Entity
    Resolution: A Survey," ACM Comput. Surv. 53(2), 2020.
[3] Peeters, Der, Bizer, "WDC Products: A Multi-Dimensional Entity Matching
    Benchmark," arXiv:2301.09521, EDBT 2024.
[4] Ruan, Shi, Bauernhansl, "Fine-tuning LLMs with contrastive margin
    ranking loss for selective entity matching," Adv. Eng. Inform. 67, 2025.
[5] Kamsteeg et al., "Confidence Calibration in LLM-Based Entity Matching,"
    2nd Workshop on Uncertainty-Aware NLP, 2025.
[6] Li, Fan, Yao, Sun, "Contextual semantics graph attention network model
    for entity resolution," Scientific Reports 15, 27093, 2025.
