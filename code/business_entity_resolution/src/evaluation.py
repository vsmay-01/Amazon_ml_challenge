"""
evaluation.py -- Leakage-safe validation split + exact macro-F0.5 metric
(Sections 14-15).
"""
from __future__ import annotations

from collections import defaultdict

import networkx as nx  # lightweight, only used for connected components
import numpy as np
import pandas as pd


def f_beta(precision: float, recall: float, beta: float = 0.5) -> float:
    if precision == 0.0 and recall == 0.0:
        return 0.0
    b2 = beta ** 2
    denom = b2 * precision + recall
    if denom == 0:
        return 0.0
    return (1 + b2) * precision * recall / denom


def query_score(true_set: set, pred_set: set, beta: float = 0.5) -> float:
    """
    Section 15 exact metric, including the organizer's singleton convention:
    true empty + predicted empty -> full credit; true empty + nonempty
    prediction -> zero credit.
    """
    if not true_set and not pred_set:
        return 1.0
    if not true_set and pred_set:
        return 0.0
    if true_set and not pred_set:
        return 0.0  # zero recall -> zero F-beta
    tp = len(true_set & pred_set)
    precision = tp / len(pred_set)
    recall = tp / len(true_set)
    return f_beta(precision, recall, beta=beta)


def macro_f_beta(
    true_by_query: dict[str, set],
    pred_by_query: dict[str, set],
    beta: float = 0.5,
) -> float:
    scores = []
    for qid, true_set in true_by_query.items():
        pred_set = pred_by_query.get(qid, set())
        scores.append(query_score(true_set, pred_set, beta=beta))
    return float(np.mean(scores)) if scores else 0.0


def run_hand_test_cases(beta: float = 0.5) -> pd.DataFrame:
    """Section 15.1 hand-test cases, as an executable regression check."""
    cases = [
        (set(), set(), "Empty prediction", "full singleton credit"),
        (set(), {"x"}, "False merge", "zero singleton credit"),
        ({"x"}, {"x"}, "Single true match", "perfect"),
        ({"x", "y"}, {"x"}, "Missed multi-match", "partial recall"),
        ({"x"}, {"x", "y"}, "Extra false merge", "precision penalty"),
        ({"x", "y"}, {"x", "y"}, "Multi-match preservation", "perfect"),
    ]
    rows = []
    for true_set, pred_set, purpose, expected in cases:
        rows.append(
            {
                "true_set": true_set,
                "pred_set": pred_set,
                "purpose": purpose,
                "expected": expected,
                "score": query_score(true_set, pred_set, beta=beta),
            }
        )
    return pd.DataFrame(rows)


def build_identity_graph(ground_truth: pd.DataFrame) -> nx.Graph:
    """
    Build the connected-components identity graph (Section 14.1): edges
    connect records known to refer to the same real-world entity, across
    all three sources. `ground_truth` has columns source1_id, source2_id,
    source3_id (either may be empty string for a given row).
    """
    g = nx.Graph()
    for _, row in ground_truth.iterrows():
        nodes = [
            ("s1", row.get("source1_id", "")),
            ("s2", row.get("source2_id", "")),
            ("s3", row.get("source3_id", "")),
        ]
        nodes = [n for n in nodes if n[1]]
        for n in nodes:
            g.add_node(n)
        for i in range(len(nodes)):
            for j in range(i + 1, len(nodes)):
                g.add_edge(nodes[i], nodes[j])
    return g


def leakage_safe_split(
    ground_truth: pd.DataFrame,
    source1_ids: list[str],
    val_fraction: float = 0.2,
    random_state: int = 42,
) -> tuple[list[str], list[str]]:
    """
    Split by connected component (Section 14.1), not by row, so that no
    Source 2/3 record that is linked to multiple Source 1 entities straddles
    the train/validation boundary. Returns (train_source1_ids, val_source1_ids).
    """
    g = build_identity_graph(ground_truth)
    components = list(nx.connected_components(g))

    s1_to_component: dict[str, int] = {}
    for comp_idx, comp in enumerate(components):
        for (source, sid) in comp:
            if source == "s1":
                s1_to_component[sid] = comp_idx

    # Singleton S1 entities (no ground-truth links, or not in any component)
    # get their own component so they can be split freely too.
    next_comp = len(components)
    for sid in source1_ids:
        if sid not in s1_to_component:
            s1_to_component[sid] = next_comp
            next_comp += 1

    comp_to_ids: dict[int, list[str]] = defaultdict(list)
    for sid, comp_idx in s1_to_component.items():
        comp_to_ids[comp_idx].append(sid)

    comp_ids = list(comp_to_ids.keys())
    rng = np.random.default_rng(random_state)
    rng.shuffle(comp_ids)

    val_ids: list[str] = []
    train_ids: list[str] = []
    target_val_count = int(round(len(source1_ids) * val_fraction))

    for comp_idx in comp_ids:
        ids = comp_to_ids[comp_idx]
        if len(val_ids) < target_val_count:
            val_ids.extend(ids)
        else:
            train_ids.extend(ids)

    return train_ids, val_ids
