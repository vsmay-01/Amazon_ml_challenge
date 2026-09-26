import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code" / "business_entity_resolution"))

from src.features import FULL_FEATURE_COLUMNS
from src.model import PairwiseMatcher


def test_pairwise_matcher_fits_and_checkpoints_across_batches(tmp_path):
    batches = []
    for labels in ([0, 0], [1, 1]):
        batch = pd.DataFrame(0.0, index=range(2), columns=FULL_FEATURE_COLUMNS)
        batch["label"] = labels
        batches.append(batch)

    saved_batch_numbers = []
    matcher = PairwiseMatcher().fit_batches(
        batches, on_batch_end=saved_batch_numbers.append
    )

    assert saved_batch_numbers == [1, 2]
    assert matcher.predict_proba(batches[0]).shape == (2,)

    model_path = tmp_path / "matcher.joblib"
    matcher.save(str(model_path))
    restored = PairwiseMatcher.load(str(model_path))
    assert restored.predict_proba(batches[0]).shape == (2,)