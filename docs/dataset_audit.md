# Dataset Audit (template)

Run this audit first (Hours 0-4 of the plan in `docs/pitch.md`) once the real
`train_source1.tsv` / `train_source2.tsv` / `train_source3.tsv` /
`train_ground_truth.tsv` files are in `dataset/train/`. Suggested checks,
each of which should produce a row in `reports/dataset_statistics.json`:

## Schema and basic stats
- [ ] Row counts for Source 1 / Source 2 / Source 3
- [ ] Column-level null / empty-string rate for `business_name`,
      `business_address`, `country`
- [ ] Distinct country values (confirm US/India in training; watch for any
      unexpected values that would signal a labeling issue)
- [ ] Duplicate `entity_id` values within a source (should be zero)

## Identity structure
- [ ] Does any Source 2 or Source 3 record participate in ground-truth links
      with multiple Source 1 entities? (Required to know before building
      `src/evaluation.py::leakage_safe_split`'s identity graph.)
- [ ] Distribution of match cardinality per Source 1 entity: how many are
      true singletons vs. 1 match vs. many matches, split by target source.
- [ ] Ground-truth connected-component size distribution
      (`src/evaluation.py::build_identity_graph`).

## Text quality signals
- [ ] Fraction of names containing a legal suffix (Inc/LLC/Ltd/...)
- [ ] Fraction of addresses with fewer than 3 tokens (likely partial)
- [ ] Rough transliteration signal: fraction of names with non-ASCII
      characters before normalization
- [ ] Sample of name pairs that are ground-truth matches but have low
      character n-gram overlap (candidates for the "missed by exact/token
      blocking, needs char n-gram or TF-IDF pass" case)

## Output
Populate `reports/dataset_statistics.json` with the above, and link the
notable findings (e.g. "12% of Source 3 addresses are landmark descriptions
with no street token") into `docs/error_analysis.md` and
`docs/blocking_strategy.md` so the blocking-pass and feature design stay
grounded in the actual data rather than only the spec's prose description.
