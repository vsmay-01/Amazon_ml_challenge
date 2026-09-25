# Judge-Facing Pitch (Section 26)

## Narrative
> Most entity resolution pipelines answer a pairwise question: "Do these two
> records look similar?" TriVote-ER++ asks a more useful question: "Given
> all evidence available from the three sources, how strong is this
> candidate, how much independent cross-source support does it have, how
> uncertain is the decision, and should this query accept zero, one, or
> multiple links?"

The system starts with multi-pass blocking to avoid losing true matches,
uses hard negatives to focus learning on realistic confusions, optionally
learns cross-source evidence with a graph model, and finally calibrates a
selective decision policy against the actual macro-F0.5 objective.

## Suggested live demo flow (Section 26, Fig. 4)
1. Load Source 1 / Source 2 / Source 3.
2. Show the dataset audit and source statistics (`docs/dataset_audit.md`,
   `reports/dataset_statistics.json`).
3. Show blocking coverage and candidate cost
   (`reports/blocking_experiments.csv`).
4. Inspect name, address and cross-source evidence for a few example
   queries (pull directly from `src/features.py::build_feature_table` /
   `add_cross_source_support` output).
5. Explain accepted, rejected, and abstained candidates for one ambiguous
   query using `src/decision.py::DecisionPolicy.decide`.
6. Display validation macro-F0.5 and the error taxonomy breakdown
   (`docs/error_analysis.md`).
7. Export the required TSV files and run `utils/validate_submission.py`
   live to show the submission contract is satisfied.

## 48-Hour execution plan

| Time | Work | Deliverable |
|---|---|---|
| 0-4h | Audit all TSVs, schema, duplicates, missingness, S2/S3 reuse, cardinality | dataset audit |
| 4-8h | Exact + fuzzy + TF-IDF baselines | baseline scores |
| 8-14h | Blocking passes and marginal-recall analysis | blocking report |
| 14-20h | Feature model + hard negatives | first competitive matcher |
| 20-28h | Cross-source support + optional GAT | triadic experiment |
| 28-34h | Calibration and uncertainty experiments | decision policy |
| 34-39h | Threshold and query-level policy | optimized validation result |
| 39-43h | Error analysis + ablations | scientific evidence |
| 43-46h | Test inference + output generation + validator | submission artifacts |
| 46-48h | Documentation, pitch, packaging | final ZIP + judge demo |

Fallback order if time runs out: **Blocking -> Strong feature model -> Hard
negatives -> Calibration -> Triadic support -> GAT.** Everything left of the
cut is still a complete, compliant submission; everything right of the cut
is a bonus that must have already passed its ablation to be worth keeping.
