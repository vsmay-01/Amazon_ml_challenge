# Final Report (fill in after running experiments)

Use `Documentation_template.md` as the checklist while completing this file.
This is the report that should accompany the ZIP submission; it should read
as a narrative over the actual numbers in `reports/`, not a restatement of
the design doc.

## 1. What was built
_Summarize the architecture actually shipped (Candidate A/B/C -- see
`docs/architecture_decision.md`) and why._

## 2. Experiments and results
_Insert/summarize tables from:_
- `reports/blocking_experiments.csv`
- `reports/model_experiments.csv`
- `reports/threshold_experiments.csv`
- `reports/graph_ablation.csv` (if the GAT branch was exercised)
- `reports/ablation_results.csv`

## 3. Validation macro-F0.5
_Report the final validation score from
`reports/training_manifest.json`, and the singleton / multi-match /
precision / recall breakdown._

## 4. Error analysis
_Summarize the top 2-3 error categories from `docs/error_analysis.md` with
concrete counts, and what (if anything) was changed in response._

## 5. What was kept vs. rejected, and why
_For each component in Section 22's ablation plan (address features,
character features, TF-IDF features, hard negatives, cross-source support,
GAT, calibration, query-adaptive decision policy, individual blocking
passes): one line citing the experiment that justified keeping or dropping
it._

## 6. Reproducibility
_Python/dependency versions, random seed(s), the exact `train`/`predict`
commands used, and the split manifest._

## 7. Compliance
_Confirm: no external lookups anywhere in `src/`, no US/India-only country
logic, model license and parameter count, `utils/validate_submission.py`
passed on the real test output._
