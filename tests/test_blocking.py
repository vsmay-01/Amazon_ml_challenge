"""
Tests for multi-pass blocking (Section 6): exact-name recall on clean data,
and that candidate unions never drop a candidate found by any single pass.
"""
import sys
from pathlib import Path

import pandas as pd
from pandas.testing import assert_frame_equal

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code" / "business_entity_resolution"))

from src.blocking import DiskTargetIndex, run_blocking
from src.pipeline import _iter_feature_batches, build_candidates_and_features


def _s1():
    return pd.DataFrame(
        [
            {"entity_id": "s1_1", "business_name": "Acme Corp", "business_address": "1 Main St", "country": "US"},
            {"entity_id": "s1_2", "business_name": "Globex Ltd", "business_address": "9 Elm Ave", "country": "IN"},
        ]
    )


def _s2():
    return pd.DataFrame(
        [
            {"entity_id": "s2_1", "business_name": "ACME Corp.", "business_address": "1 Main Street", "country": "US"},
            {"entity_id": "s2_2", "business_name": "Totally Different Co", "business_address": "5 Oak Rd", "country": "US"},
        ]
    )


def test_exact_name_pass_finds_case_and_suffix_variant():
    result = run_blocking(_s1(), _s2(), "source2", passes=["exact_name"])
    cands = result.candidates.get("s1_1", set())
    assert ("source2", "s2_1") in cands


def test_union_is_superset_of_each_pass():
    result = run_blocking(_s1(), _s2(), "source2")
    union = result.candidates.get("s1_1", set())
    for pass_name, pass_cands in result.per_pass_candidates.items():
        assert pass_cands.get("s1_1", set()).issubset(union)


def test_unrelated_entity_not_falsely_blocked_by_exact_pass():
    result = run_blocking(_s1(), _s2(), "source2", passes=["exact_name"])
    cands = result.candidates.get("s1_2", set())
    assert ("source2", "s2_2") not in cands


def test_disk_target_index_matches_full_table_blocking_across_ingest_batches(tmp_path):
    query = _s1()
    target = _s2()
    expected = run_blocking(query, target, "source2").candidates

    index = DiskTargetIndex(str(tmp_path / "target.sqlite"))
    index.add_batch(target.iloc[:1])
    index.add_batch(target.iloc[1:])
    index.finalize()
    try:
        actual = {
            row["entity_id"]: {
                ("source2", candidate_id)
                for candidate_id in index.candidate_ids(row)
            }
            for _, row in query.iterrows()
        }
    finally:
        index.close()

    assert actual == expected


def test_streamed_features_match_full_table_features(tmp_path):
    source1 = _s1()
    source2 = _s2()
    source3 = pd.DataFrame(
        [
            {"entity_id": "s3_1", "business_name": "Acme Corporation", "business_address": "1 Main St", "country": "US"},
            {"entity_id": "s3_2", "business_name": "Globex Limited", "business_address": "9 Elm Avenue", "country": "IN"},
        ]
    )
    for name, frame in (
        ("train_source1.tsv", source1),
        ("train_source2.tsv", source2),
        ("train_source3.tsv", source3),
    ):
        frame.to_csv(tmp_path / name, sep="\t", index=False)

    _, expected = build_candidates_and_features(source1, source2, source3)
    streamed = [
        feature_batch
        for _, feature_batch in _iter_feature_batches(str(tmp_path), "train", batch_size=1)
    ]
    actual = pd.concat(streamed, ignore_index=True)
    sort_columns = ["source1_id", "candidate_source", "candidate_id"]

    assert_frame_equal(
        actual.sort_values(sort_columns).reset_index(drop=True),
        expected.sort_values(sort_columns).reset_index(drop=True),
        check_dtype=False,
        check_exact=False,
        rtol=1e-12,
        atol=1e-12,
    )
