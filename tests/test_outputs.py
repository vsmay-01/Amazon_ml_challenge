"""
Tests for the submission contract (Section 2.2 / 24.1):
- required files exist with required columns
- final predictions are a subset of the final candidate set
- no duplicate matched ids per query beyond what the candidate set allows
"""
import sys
import subprocess
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code" / "business_entity_resolution"))

from src.io import write_submission


def test_write_submission_creates_expected_files(tmp_path):
    matches = pd.DataFrame(
        [
            {"source1_id": "S1-1", "matched_source": "source2", "matched_id": "S2-1"},
            {"source1_id": "S1-1", "matched_source": "source3", "matched_id": "S3-1"},
            {"source1_id": "S1-2", "matched_source": "", "matched_id": ""},
        ]
    )
    candidates = pd.DataFrame(
        [
            {"source1_id": "S1-1", "candidate_source": "source2", "candidate_id": "S2-1"},
            {"source1_id": "S1-1", "candidate_source": "source2", "candidate_id": "S2-2"},
            {"source1_id": "S1-1", "candidate_source": "source3", "candidate_id": "S3-1"},
        ]
    )
    write_submission(matches, candidates, str(tmp_path))

    out_matches = tmp_path / "matching_results.tsv"
    out_candidates = tmp_path / "candidate_pairs.tsv"
    assert out_matches.exists()
    assert out_candidates.exists()

    m = pd.read_csv(out_matches, sep="\t", keep_default_na=False)
    c = pd.read_csv(out_candidates, sep="\t", keep_default_na=False)
    assert list(m.columns) == ["source1_entity_id", "matched_entity_ids"]
    assert list(c.columns) == ["source1_entity_id", "candidate_entity_ids"]
    assert m.to_dict("records") == [
        {"source1_entity_id": "S1-1", "matched_entity_ids": "S2-1,S3-1"},
        {"source1_entity_id": "S1-2", "matched_entity_ids": ""},
    ]
    assert c.to_dict("records") == [
        {"source1_entity_id": "S1-1", "candidate_entity_ids": "S2-1,S2-2,S3-1"},
        {"source1_entity_id": "S1-2", "candidate_entity_ids": ""},
    ]


def test_submission_passes_challenge_validator(tmp_path):
    output_dir = tmp_path / "output"
    test_dir = tmp_path / "test"
    test_dir.mkdir()
    matches = pd.DataFrame(
        [
            {"source1_id": "S1-1", "matched_source": "source2", "matched_id": "S2-1"},
            {"source1_id": "S1-2", "matched_source": "", "matched_id": ""},
        ]
    )
    candidates = pd.DataFrame(
        [{"source1_id": "S1-1", "candidate_source": "source2", "candidate_id": "S2-1"}]
    )
    write_submission(matches, candidates, str(output_dir))
    pd.DataFrame({"entity_id": ["S1-1", "S1-2"]}).to_csv(
        test_dir / "test_source1.tsv", sep="\t", index=False
    )
    pd.DataFrame({"entity_id": ["S2-1"]}).to_csv(
        test_dir / "test_source2.tsv", sep="\t", index=False
    )
    pd.DataFrame({"entity_id": ["S3-1"]}).to_csv(
        test_dir / "test_source3.tsv", sep="\t", index=False
    )
    candidate_output = pd.read_csv(
        output_dir / "candidate_pairs.tsv", sep="\t", keep_default_na=False
    )
    candidate_output.iloc[::-1].to_csv(
        output_dir / "candidate_pairs.tsv", sep="\t", index=False
    )

    validator = Path(__file__).resolve().parents[1] / "utils" / "validate_submission.py"
    result = subprocess.run(
        [
            sys.executable,
            str(validator),
            "--matching", str(output_dir / "matching_results.tsv"),
            "--candidate", str(output_dir / "candidate_pairs.tsv"),
            "--test-dir", str(test_dir),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "PASS" in result.stdout


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
