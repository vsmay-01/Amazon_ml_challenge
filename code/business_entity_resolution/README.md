# TriVote-ER++ — Multi-Source Business Entity Resolution

Reference implementation for the Amazon Business Entity Resolution Challenge,
following the design in `docs/` (mirrored from the technical design document).

## Quick start

```bash
cd code/business_entity_resolution
pip install -r requirements.txt

# Train + calibrate on the training split, run leakage-safe validation
python -m src.pipeline train \
  --train-dir ../../dataset/train \
  --work-dir ../../reports \
  --model-out ../../reports/model.joblib

# Run inference on the test split and produce the submission files
python -m src.pipeline predict \
  --test-dir ../../dataset/test \
  --model-in ../../reports/model.joblib \
  --output-dir ../../output
```

This writes:

- `output/matching_results.tsv`
- `output/candidate_pairs.tsv`

Validate the submission contract with:

```bash
python ../../utils/validate_submission.py --output-dir ../../output --source1 ../../dataset/test/test_source1.tsv
```

## Design principles (see `docs/architecture_decision.md`)

1. Establish a reproducible baseline before adding complexity.
2. Measure candidate recall before optimizing the matcher.
3. Add complementary candidate generators one at a time (multi-pass blocking).
4. Fuse cross-source (S2<->S3) evidence as a *feature*, not an unconditional merge rule.
5. Mine hard negatives from the real candidate distribution.
6. Calibrate confidence, then optimize a query-level zero/one/many decision policy
   directly for macro-F0.5 (precision-weighted).
7. Keep the GAT / contrastive / MC-Dropout branches only if ablation proves them.

## Module map

| Module | Responsibility |
|---|---|
| `src/io.py` | Load/validate the three source TSVs and ground truth; write submission files |
| `src/normalization.py` | Multi-view name/address/country normalization |
| `src/blocking.py` | Multi-pass candidate generation + marginal recall/cost accounting |
| `src/features.py` | Pairwise + triadic (cross-source) feature engineering |
| `src/hard_negatives.py` | Hard-negative sampling from the real candidate distribution |
| `src/model.py` | Gradient-boosted pairwise scorer (baseline + primary model) |
| `src/graph_reasoning.py` | Optional GAT experimental module (Innovation 3), gated by ablation |
| `src/calibration.py` | Score calibration (Platt/isotonic) + MC-Dropout-style uncertainty |
| `src/decision.py` | Query-level 0/1/many selective decision policy |
| `src/evaluation.py` | Leakage-safe splitting + exact macro-F0.5 metric |
| `src/pipeline.py` | CLI: `train` and `predict` end-to-end orchestration |

## Compliance

No external business-identity lookups, commercial ER APIs, government-registration
lookups, geocoding APIs, or internet-based business-data augmentation are used
anywhere in this codebase (see `docs/limitations.md`).
