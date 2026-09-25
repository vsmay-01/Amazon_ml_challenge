"""
io.py -- Loading and writing challenge data.

Handles the three source TSVs (Source 1 / Source 2 / Source 3), the training
ground-truth file, and the two required submission files:

    output/matching_results.tsv
    output/candidate_pairs.tsv

All files are TSV with columns: entity_id, business_name, business_address, country.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

import pandas as pd

SOURCE_COLUMNS = ["entity_id", "business_name", "business_address", "country"]
GROUND_TRUTH_COLUMNS = ["source1_id", "source2_id", "source3_id"]


@dataclass
class ChallengeData:
    source1: pd.DataFrame
    source2: pd.DataFrame
    source3: pd.DataFrame
    ground_truth: Optional[pd.DataFrame] = None


def _read_tsv(path: str, expected_columns: list[str]) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)
    missing = [c for c in expected_columns if c not in df.columns]
    if missing:
        raise ValueError(
            f"{path}: missing expected columns {missing}. Found {list(df.columns)}"
        )
    for c in expected_columns:
        df[c] = df[c].fillna("").astype(str)
    return df


def load_sources(data_dir: str, prefix: str, with_ground_truth: bool) -> ChallengeData:
    """
    Load source1/source2/source3 (and optionally ground_truth) from `data_dir`.
    `prefix` is e.g. "train" or "test" and files are expected to be named
    f"{prefix}_source1.tsv", etc.
    """
    s1 = _read_tsv(os.path.join(data_dir, f"{prefix}_source1.tsv"), SOURCE_COLUMNS)
    s2 = _read_tsv(os.path.join(data_dir, f"{prefix}_source2.tsv"), SOURCE_COLUMNS)
    s3 = _read_tsv(os.path.join(data_dir, f"{prefix}_source3.tsv"), SOURCE_COLUMNS)

    gt = None
    if with_ground_truth:
        gt_path = os.path.join(data_dir, f"{prefix}_ground_truth.tsv")
        if os.path.exists(gt_path):
            gt = pd.read_csv(gt_path, sep="\t", dtype=str, keep_default_na=False)

    return ChallengeData(source1=s1, source2=s2, source3=s3, ground_truth=gt)


def write_submission(
    matches: pd.DataFrame,
    candidate_pairs: pd.DataFrame,
    output_dir: str,
) -> None:
    """
    Write the two required submission files.

    `matches` must have columns: source1_id, source2_id, source3_id
        (source2_id / source3_id empty string when not applicable -- a match
        row always links a Source 1 id to exactly one Source 2 OR Source 3 id;
        represent each link as its own row so a query with multiple matches
        produces multiple rows.)
    `candidate_pairs` must be a superset of `matches` and represent the actual
        final candidate set scored by the matcher (not a discarded earlier
        blocking stage).
    """
    os.makedirs(output_dir, exist_ok=True)

    match_cols = ["source1_id", "matched_source", "matched_id", "score"]
    for c in match_cols:
        if c not in matches.columns:
            raise ValueError(f"matches is missing required column: {c}")

    cand_cols = ["source1_id", "candidate_source", "candidate_id"]
    for c in cand_cols:
        if c not in candidate_pairs.columns:
            raise ValueError(f"candidate_pairs is missing required column: {c}")

    matches.to_csv(
        os.path.join(output_dir, "matching_results.tsv"), sep="\t", index=False
    )
    candidate_pairs.to_csv(
        os.path.join(output_dir, "candidate_pairs.tsv"), sep="\t", index=False
    )
