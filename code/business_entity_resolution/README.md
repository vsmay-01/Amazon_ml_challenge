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
python ../../utils/validate_submission.py \
  --matching ../../output/matching_results.tsv \
  --candidate ../../output/candidate_pairs.tsv \
  --test-dir ../../dataset/test
```

## Kaggle training

Attach the dataset to the Notebook with this layout:

```text
/kaggle/input/amazon-business-entity-resolution/train/train_source1.tsv
/kaggle/input/amazon-business-entity-resolution/train/train_source2.tsv
/kaggle/input/amazon-business-entity-resolution/train/train_source3.tsv
/kaggle/input/amazon-business-entity-resolution/train/train_ground_truth.tsv
/kaggle/input/amazon-business-entity-resolution/test/...
```

Make this repository's `code/business_entity_resolution` directory available
to the Notebook too (for example, attach a Kaggle Dataset containing the
project files). In a Notebook cell, install dependencies and configure the
input directory:

```python
%cd /kaggle/input/<project-code-dataset>/code/business_entity_resolution
!pip install -r requirements.txt
import os
os.environ["ER_DATASET_DIR"] = "/kaggle/input/amazon-business-entity-resolution"
```

Then start or resume training:

```python
!python -m src.pipeline train --dataset-dir "$ER_DATASET_DIR" --batch-size 32 --index-dir /kaggle/working/index-cache --work-dir /kaggle/working/reports --model-out /kaggle/working/model.joblib --checkpoint-path /kaggle/working/checkpoints/training.joblib
```

If the Notebook session is interrupted, run the same command with `--resume`.
Checkpoints and the final model are written under `/kaggle/working/`; download
or save them as a Kaggle Notebook output before the session ends. Kaggle
clears `/kaggle/working` between sessions, so for a new session attach the
saved checkpoint as an input Dataset and copy it back before resuming:

```python
!mkdir -p /kaggle/working/checkpoints
!cp /kaggle/input/<checkpoint-dataset>/training.joblib /kaggle/working/checkpoints/training.joblib
!python -m src.pipeline train --dataset-dir "$ER_DATASET_DIR" --batch-size 32 --index-dir /kaggle/working/index-cache --work-dir /kaggle/working/reports --model-out /kaggle/working/model.joblib --checkpoint-path /kaggle/working/checkpoints/training.joblib --resume
```

Set `ER_DATASET_DIR` to the actual Kaggle input slug, or pass `--dataset-dir`
directly. The default batch size is 32 and can be overridden with
`--batch-size`.

The production matcher is a scikit-learn incremental classifier, so its
training and the CPU-heavy blocking/feature generation run on CPU. Kaggle's
GPU accelerator, CUDA, and mixed precision do not apply to this existing
estimator; using them would require replacing the model implementation and
would no longer preserve this project's current ML behavior. The input TSVs
are read in chunks. Target-side blocking indexes are built once per source
in temporary SQLite files under `--index-dir` (default: the system temp
directory), then queried for each Source-1 batch; the index files are removed
when that pass finishes. Ensure the selected scratch directory has enough
free disk space for these indexes.

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
| `src/model.py` | Incremental scikit-learn pairwise scorer (baseline + primary model) |
| `src/graph_reasoning.py` | Optional GAT experimental module (Innovation 3), gated by ablation |
| `src/calibration.py` | Score calibration (Platt/isotonic) + MC-Dropout-style uncertainty |
| `src/decision.py` | Query-level 0/1/many selective decision policy |
| `src/evaluation.py` | Leakage-safe splitting + exact macro-F0.5 metric |
| `src/pipeline.py` | CLI: `train` and `predict` end-to-end orchestration |

## Compliance

No external business-identity lookups, commercial ER APIs, government-registration
lookups, geocoding APIs, or internet-based business-data augmentation are used
anywhere in this codebase (see `docs/limitations.md`).
