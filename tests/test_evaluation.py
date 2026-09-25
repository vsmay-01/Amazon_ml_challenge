"""
Regression tests for the exact macro-F0.5 metric (Section 15), including
the six hand-test cases from Section 15.1.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code" / "business_entity_resolution"))

from src.evaluation import macro_f_beta, query_score, run_hand_test_cases


def test_hand_test_cases():
    df = run_hand_test_cases()
    expected = {
        "Empty prediction": 1.0,
        "False merge": 0.0,
        "Single true match": 1.0,
        "Missed multi-match": None,  # partial credit, just check bounds below
        "Extra false merge": None,
        "Multi-match preservation": 1.0,
    }
    for _, row in df.iterrows():
        exp = expected[row["purpose"]]
        if exp is not None:
            assert abs(row["score"] - exp) < 1e-9, row["purpose"]
        else:
            assert 0.0 < row["score"] < 1.0, row["purpose"]


def test_precision_weighted_over_recall():
    # Same tp, but the "false merge" variant with an extra false positive
    # should be penalized harder than the variant with an extra false negative,
    # since precision is weighted 2x recall in F0.5.
    true_set = {"x", "y"}
    missed = query_score(true_set, {"x"}, beta=0.5)        # recall 0.5, precision 1.0
    extra = query_score(true_set, {"x", "y", "z"}, beta=0.5)  # recall 1.0, precision 0.667
    # Both should be > 0 and < 1; F0.5 favors the higher-precision (missed) case
    assert missed > 0.0 and extra > 0.0
    assert missed > extra


def test_macro_average():
    true_by_query = {"q1": {"a"}, "q2": set()}
    pred_by_query = {"q1": {"a"}, "q2": set()}
    assert macro_f_beta(true_by_query, pred_by_query) == 1.0

    pred_by_query_bad = {"q1": set(), "q2": {"z"}}
    assert macro_f_beta(true_by_query, pred_by_query_bad) == 0.0
