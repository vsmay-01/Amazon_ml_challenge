# Documentation Template

Use this checklist when filling in `docs/final_report.md` before submission.

## 1. Problem framing
- [ ] Restate the scoring objective (macro-F0.5, precision weighted 2x recall)
- [ ] Restate the submission contract (files, schema, compliance gates)

## 2. What was measured
- [ ] Candidate recall per blocking pass (`reports/blocking_experiments.csv`)
- [ ] Model experiments (`reports/model_experiments.csv`)
- [ ] Threshold / decision-policy sweep (`reports/threshold_experiments.csv`)
- [ ] Graph ablation, if attempted (`reports/graph_ablation.csv`)
- [ ] Full ablation table (`reports/ablation_results.csv`)

## 3. What was kept and why
- [ ] For each retained component, cite the experiment that justified it
- [ ] For each rejected component, cite the experiment that ruled it out

## 4. Reproducibility
- [ ] Python + dependency versions
- [ ] Random seeds
- [ ] Train/validation split manifest (component-level, not row-level)
- [ ] Exact commands to reproduce `output/matching_results.tsv`

## 5. Compliance
- [ ] No external lookups / geocoding / commercial ER APIs anywhere in `src/`
- [ ] No hard-coded US/India-only country logic
- [ ] License + parameter-count statement for any learned model used
