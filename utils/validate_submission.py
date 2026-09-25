#!/usr/bin/env python3
"""
validate_submission.py -- Checks the submission contract and hard compliance
gates from Section 2.2 / Section 24.1 before a ZIP is packaged.

Usage:
    python validate_submission.py --output-dir output --source1 dataset/test/test_source1.tsv
"""
from __future__ import annotations

import argparse
import sys

import pandas as pd


def fail(msg: str, errors: list[str]) -> None:
    errors.append(msg)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--source1", required=True, help="Path to test_source1.tsv")
    args = parser.parse_args()

    errors: list[str] = []

    try:
        matches = pd.read_csv(f"{args.output_dir}/matching_results.tsv", sep="\t", dtype=str)
    except FileNotFoundError:
        fail("matching_results.tsv is missing from the output directory.", errors)
        matches = None

    try:
        candidates = pd.read_csv(f"{args.output_dir}/candidate_pairs.tsv", sep="\t", dtype=str)
    except FileNotFoundError:
        fail("candidate_pairs.tsv is missing from the output directory.", errors)
        candidates = None

    source1 = pd.read_csv(args.source1, sep="\t", dtype=str)
    all_ids = set(source1["entity_id"])

    if matches is not None:
        required_match_cols = {"source1_id", "matched_source", "matched_id", "score"}
        if not required_match_cols.issubset(matches.columns):
            fail(f"matching_results.tsv missing columns: {required_match_cols - set(matches.columns)}", errors)
        else:
            present_ids = set(matches["source1_id"])
            missing = all_ids - present_ids
            if missing:
                fail(f"{len(missing)} Source 1 test entities never appear in matching_results.tsv "
                     f"(every entity must appear at least once, even as a singleton row): "
                     f"{sorted(list(missing))[:5]}...", errors)

            real_matches = matches[matches["matched_id"].fillna("") != ""]
            dupe = real_matches.duplicated(subset=["source1_id", "matched_source", "matched_id"])
            if dupe.any():
                fail(f"{int(dupe.sum())} duplicate (source1_id, matched_source, matched_id) rows found.", errors)

            if candidates is not None and not real_matches.empty:
                cand_keys = set(
                    zip(candidates["source1_id"], candidates["candidate_source"], candidates["candidate_id"])
                )
                match_keys = set(
                    zip(real_matches["source1_id"], real_matches["matched_source"], real_matches["matched_id"])
                )
                not_in_candidates = match_keys - cand_keys
                if not_in_candidates:
                    fail(
                        f"{len(not_in_candidates)} predicted matches are not present in "
                        f"candidate_pairs.tsv (final predictions must be a subset of the final "
                        f"candidate set): {list(not_in_candidates)[:5]}...",
                        errors,
                    )

    if candidates is not None:
        required_cand_cols = {"source1_id", "candidate_source", "candidate_id"}
        if not required_cand_cols.issubset(candidates.columns):
            fail(f"candidate_pairs.tsv missing columns: {required_cand_cols - set(candidates.columns)}", errors)

    if errors:
        print("SUBMISSION VALIDATION FAILED:\n")
        for e in errors:
            print(f"  - {e}")
        return 1

    print("Submission validation passed: files present, schema correct, "
          "every Source 1 entity covered, predictions are a subset of candidates.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
