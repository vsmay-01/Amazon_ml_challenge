# Problem Analysis

## Task
Source 1 is the deduplicated reference source. For every Source 1 record,
find all corresponding Source 2 and Source 3 records. A Source 1 entity may
match zero, one, or many records in each of the other sources.

All files are TSV: `(entity_id, business_name, business_address, country)`.

## Data challenges named in the spec
- Abbreviations and legal-suffix variation in business names
- Punctuation changes, typos, word-order changes
- Transliteration
- Missing address components, landmark descriptions, reordered address parts
- Country coverage: **US + India in training, France added at test time** ->
  country must be treated as an open-set string attribute; no US/India-only
  whitelist anywhere in the codebase (`src/normalization.py` enforces this).

## Scoring objective
Macro-averaged F0.5, computed per Source 1 entity then averaged:

    F0.5 = 1.25 * P * R / (0.25 * P + R)

Precision is weighted 2x recall. Singletons are scored too: predicting an
empty match list for a true singleton gets full credit; predicting any match
for a true singleton gets zero credit. This is why the design principle is:

> The objective is not "maximize similarity"; it is "make the best
> zero/one/many decision per query."

## Submission contract
- `output/matching_results.tsv`
- `output/candidate_pairs.tsv` (must be a superset of the final predictions,
  representing the actual final candidate set used by the matcher)
- Runnable source code, pinned dependencies, methodology docs, reproducible
  execution path
- MIT/Apache-2.0 license, <= 8B parameters for any learned model
- No external business-identity lookups, commercial ER APIs, government
  registration lookups, geocoding APIs, or internet-based business-data
  augmentation anywhere in the pipeline

See `docs/validation_protocol.md` and `docs/error_analysis.md` for how these
constraints shape the design.
