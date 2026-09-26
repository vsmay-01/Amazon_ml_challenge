import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code" / "business_entity_resolution"))

from src.pipeline import _ground_truth_pair_set, _read_ground_truth


def test_reads_prefixed_comma_separated_ground_truth(tmp_path):
    ground_truth_path = tmp_path / "train_ground_truth.tsv"
    pd.DataFrame(
        {
            "source1_entity_id": ["S1-1", "S1-2"],
            "matched_entity_ids": ["S2-2,S3-3,S2-4", ""],
        }
    ).to_csv(ground_truth_path, sep="\t", index=False)

    ground_truth = _read_ground_truth(str(tmp_path), "train")

    assert _ground_truth_pair_set(ground_truth) == {
        ("S1-1", "source2", "S2-2"),
        ("S1-1", "source3", "S3-3"),
        ("S1-1", "source2", "S2-4"),
    }


def test_rejects_unrecognized_matched_entity_prefix(tmp_path):
    ground_truth_path = tmp_path / "train_ground_truth.tsv"
    pd.DataFrame(
        {"source1_entity_id": ["S1-1"], "matched_entity_ids": ["UNKNOWN-2"]}
    ).to_csv(ground_truth_path, sep="\t", index=False)

    with pytest.raises(ValueError, match="S2- or S3-"):
        _read_ground_truth(str(tmp_path), "train")