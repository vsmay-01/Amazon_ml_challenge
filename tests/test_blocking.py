"""
Tests for multi-pass blocking (Section 6): exact-name recall on clean data,
and that candidate unions never drop a candidate found by any single pass.
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code" / "business_entity_resolution"))

from src.blocking import run_blocking


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
