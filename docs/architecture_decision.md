# Architecture Decision

## Decision framework (Section 17)

| Criterion | Priority | A | B | C |
|---|---|---|---|---|
| Macro F0.5 | Primary | measured | measured | measured |
| Precision | Secondary | measured | measured | measured |
| Recall | Secondary | measured | measured | measured |
| Candidate recall | Secondary | measured | measured | measured |
| Singleton accuracy | Secondary | measured | measured | measured |
| Multi-match quality | Secondary | measured | measured | measured |
| Runtime | Secondary | measured | measured | measured |
| Memory | Secondary | measured | measured | measured |
| Reproducibility | Gate | checked | checked | checked |
| Compliance | Gate | checked | checked | checked |

Populate this table from `reports/model_experiments.csv`,
`reports/ablation_results.csv` and `reports/graph_ablation.csv` before
writing `docs/final_report.md`.

## Default choice and rationale
The codebase ships Candidate C/B by default: it satisfies every gate
(reproducible, compliant, no external lookups) with the least implementation
risk, and every advanced component (cross-source support, calibration,
selective decision policy) is already wired in and independently ablatable.
Candidate A (GAT) is available as an opt-in experiment
(`src/graph_reasoning.py`) but is **not** enabled by default, per the
"measure, ablate, retain" rule in Section 29 of the design doc.

The selected architecture should minimize unnecessary complexity subject to
competitive validation performance -- this decision should be revisited only
if `reports/graph_ablation.csv` shows a reproducible, compute-justified gain.
