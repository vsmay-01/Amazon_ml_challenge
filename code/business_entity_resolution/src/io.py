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


def iter_tsv_batches(path: str, expected_columns: list[str], batch_size: int):
    """Yield bounded DataFrame batches without retaining earlier batches."""
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    for df in pd.read_csv(
        path,
        sep="\t",
        dtype=str,
        keep_default_na=False,
        chunksize=batch_size,
    ):
        missing = [c for c in expected_columns if c not in df.columns]
        if missing:
            raise ValueError(f"{path}: missing expected columns {missing}. Found {list(df.columns)}")
        for c in expected_columns:
            df[c] = df[c].fillna("").astype(str)
        yield df[expected_columns]


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
    append: bool = False,
) -> None:
    """Write challenge-format rows, aggregating pair tables into comma lists."""
    os.makedirs(output_dir, exist_ok=True)

    match_cols = ["source1_id", "matched_source", "matched_id"]
    for c in match_cols:
        if c not in matches.columns:
            raise ValueError(f"matches is missing required column: {c}")

    cand_cols = ["source1_id", "candidate_source", "candidate_id"]
    for c in cand_cols:
        if c not in candidate_pairs.columns:
            raise ValueError(f"candidate_pairs is missing required column: {c}")

    match_ids: dict[str, list[str]] = {}
    for row in matches.itertuples(index=False):
        source1_id = str(row.source1_id)
        matched_id = str(row.matched_id)
        if not matched_id:
            continue
        if matched_id not in match_ids.setdefault(source1_id, []):
            match_ids[source1_id].append(matched_id)

    candidate_ids: dict[str, list[str]] = {}
    for row in candidate_pairs.itertuples(index=False):
        source1_id = str(row.source1_id)
        candidate_id = str(row.candidate_id)
        if not candidate_id:
            continue
        if candidate_id not in candidate_ids.setdefault(source1_id, []):
            candidate_ids[source1_id].append(candidate_id)

    source1_ids = list(dict.fromkeys(
        [str(source1_id) for source1_id in matches["source1_id"]]
        + [str(source1_id) for source1_id in candidate_pairs["source1_id"]]
    ))
    match_output = pd.DataFrame(
        {
            "source1_entity_id": source1_ids,
            "matched_entity_ids": [",".join(match_ids.get(source1_id, [])) for source1_id in source1_ids],
        }
    )
    candidate_output = pd.DataFrame(
        {
            "source1_entity_id": source1_ids,
            "candidate_entity_ids": [",".join(candidate_ids.get(source1_id, [])) for source1_id in source1_ids],
        }
    )
    mode = "a" if append else "w"
    header = not append
    match_output.to_csv(
        os.path.join(output_dir, "matching_results.tsv"),
        sep="\t", index=False, mode=mode, header=header,
    )
    candidate_output.to_csv(
        os.path.join(output_dir, "candidate_pairs.tsv"),
        sep="\t", index=False, mode=mode, header=header,
    )
