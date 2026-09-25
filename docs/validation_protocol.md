# Leakage-Safe Validation Protocol

Implemented in `src/evaluation.py::leakage_safe_split`.

## Why not a random row split?
A Source 2 or Source 3 record can be linked to multiple Source 1 entities
(identity reuse). Splitting individual candidate-pair rows at random can put
the same underlying record on both sides of the train/validation boundary,
leaking identity information.

## Required split logic (Section 14.1)
1. Build a ground-truth identity graph: nodes are (source, id) pairs, edges
   connect records known to refer to the same real-world entity across
   Source 1/2/3.
2. Compute connected components.
3. Split **components**, not rows, into train/validation. Every Source 1
   entity in a component moves to the same side as everything else in that
   component.
4. Source 1 entities with no ground-truth links (true singletons) are given
   their own singleton component so they can still be split freely.

## No-leakage checklist (Section 14.2)
- [x] Candidate generation for validation uses only validation-time records
      and training-fitted indexes (`build_candidates_and_features` is called
      separately per split in `src/pipeline.py::cmd_train`).
- [x] Validation labels are never used during model fitting (the model only
      ever sees `training_set`, built from the train-side split).
- [ ] Test labels are never used for threshold selection -- N/A at test time
      since no test ground truth is distributed; verified by inspection that
      `cmd_predict` never reads a ground-truth file.
- [x] Duplicate/linked records do not straddle the split (guaranteed by the
      component-level split).
- [x] Hard-negative mining for validation uses only training-side labels
      (mining happens once, on the train split, before validation scoring).
- [x] Blocking configurations are selected without looking at test labels.
- [x] Random seeds are recorded (`RANDOM_SEED` in `src/pipeline.py`, written
      to `reports/training_manifest.json`).
