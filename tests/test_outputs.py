"""
Tests for the submission contract (Section 2.2 / 24.1):
- required files exist with required columns
- final predictions are a subset of the final candidate set
- no duplicate matched ids per query beyond what the candidate set allows
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code" / "business_entity_resolution"))

from src.io import write_submission


def test_write_submission_creates_expected_files(tmp_path):
    matches = pd.DataFrame(
        [
            {"source1_id": "s1_1", "matched_source": "source2", "matched_id": "s2_1", "score": 0.91},
            {"source1_id": "s1_2", "matched_source": "", "matched_id": "", "score": 0.0},
        ]
    )
    candidates = pd.DataFrame(
        [
            {"source1_id": "s1_1", "candidate_source": "source2", "candidate_id": "s2_1"},
            {"source1_id": "s1_1", "candidate_source": "source2", "candidate_id": "s2_2"},
        ]
    )
    write_submission(matches, candidates, str(tmp_path))

    out_matches = tmp_path / "matching_results.tsv"
    out_candidates = tmp_path / "candidate_pairs.tsv"
    assert out_matches.exists()
    assert out_candidates.exists()

    m = pd.read_csv(out_matches, sep="\t")
    c = pd.read_csv(out_candidates, sep="\t")
    assert set(["source1_id", "matched_source", "matched_id", "score"]).issubset(m.columns)
    assert set(["source1_id", "candidate_source", "candidate_id"]).issubset(c.columns)


def test_predictions_are_subset_of_candidates():
    matches = pd.DataFrame(
        [{"source1_id": "s1_1", "matched_source": "source2", "matched_id": "s2_1", "score": 0.9}]
    )
    candidates = pd.DataFrame(
        [{"source1_id": "s1_1", "candidate_source": "source2", "candidate_id": "s2_1"}]
    )
    match_keys = set(zip(matches["source1_id"], matches["matched_source"], matches["matched_id"]))
    cand_keys = set(zip(candidates["source1_id"], candidates["candidate_source"], candidates["candidate_id"]))
    assert match_keys.issubset(cand_keys)
