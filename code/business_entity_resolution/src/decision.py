"""
decision.py -- Query-level 0/1/many decision policy (Innovation 6, Section 11).

The objective is not "maximize similarity"; it is "make the best zero/one/many
decision per query" (Eq. 2). This module converts per-candidate calibrated
scores into a final accepted set per Source 1 query, using query-context
features (rank, margin to runner-up, candidate count, score dispersion,
cross-source support/contradiction).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class DecisionPolicy:
    """
    Policy 2 from Section 10 ("confidence plus variance/margin") implemented
    as the default: accept candidate i for query q iff
        calibrated_score_i > theta1
        AND (margin to next-best candidate > theta_margin OR score_i > theta_high)
    theta1 is the primary lever for the precision/recall tradeoff under
    macro-F0.5 (precision weighted 2x); it should be swept on validation
    data and set via `fit_threshold`.
    """
    theta1: float = 0.5
    theta_margin: float = 0.05
    theta_high: float = 0.9
    max_matches_per_query: int | None = None

    def decide(self, query_group: pd.DataFrame) -> pd.DataFrame:
        """
        query_group: rows for a single source1_id, with a `calibrated_score`
        column. Returns the accepted subset (possibly empty), with
        query-context columns added.
        """
        g = query_group.sort_values("calibrated_score", ascending=False).reset_index(drop=True)
        g["rank"] = np.arange(1, len(g) + 1)
        g["cand_count"] = len(g)
        g["score_dispersion"] = g["calibrated_score"].std() if len(g) > 1 else 0.0

        second_best = g["calibrated_score"].iloc[1] if len(g) > 1 else 0.0
        g["margin_to_next"] = g["calibrated_score"] - second_best
        # For the top row the "next" is the true runner-up; for lower rows,
        # margin_to_next is not meaningful for acceptance, only diagnostics.

        accept_mask = g["calibrated_score"] > self.theta1
        # Ambiguous top cluster: if the top-2 scores are within theta_margin
        # of each other and both are below theta_high, demand stronger
        # evidence (do not accept either) -- protects singleton precision.
        if len(g) > 1:
            top_two_close = abs(g["calibrated_score"].iloc[0] - g["calibrated_score"].iloc[1]) < self.theta_margin
            neither_high = g["calibrated_score"].iloc[0] < self.theta_high
            if top_two_close and neither_high:
                accept_mask = pd.Series([False] * len(g))

        accepted = g[accept_mask].copy()
        if self.max_matches_per_query is not None:
            accepted = accepted.sort_values("calibrated_score", ascending=False).head(
                self.max_matches_per_query
            )
        return accepted


def apply_policy(
    scored_table: pd.DataFrame,
    policy: DecisionPolicy,
) -> pd.DataFrame:
    """
    scored_table: [source1_id, candidate_source, candidate_id, calibrated_score, ...]
    Returns the accepted rows across all queries, with rank/margin/etc. columns.
    """
    results = []
    for _, group in scored_table.groupby("source1_id"):
        results.append(policy.decide(group))
    if not results:
        return scored_table.iloc[0:0]
    return pd.concat(results, ignore_index=True)


def sweep_theta1(
    scored_table: pd.DataFrame,
    ground_truth_pairs: set[tuple[str, str, str]],
    all_source1_ids: list[str],
    thresholds: list[float] | None = None,
) -> pd.DataFrame:
    """
    Sweep theta1 and report validation macro-F0.5 for each, holding the
    rest of the policy fixed. Used to pick the operating point (Section 20,
    "Threshold experiments").
    """
    from .evaluation import macro_f_beta  # local import to avoid a cycle

    thresholds = thresholds or [round(x, 2) for x in np.arange(0.1, 0.96, 0.05)]
    rows = []
    for theta in thresholds:
        policy = DecisionPolicy(theta1=theta)
        accepted = apply_policy(scored_table, policy)
        pred_by_query: dict[str, set[tuple[str, str, str]]] = {qid: set() for qid in all_source1_ids}
        for _, r in accepted.iterrows():
            pred_by_query[r["source1_id"]].add(
                (r["source1_id"], r["candidate_source"], r["candidate_id"])
            )
        true_by_query: dict[str, set[tuple[str, str, str]]] = {qid: set() for qid in all_source1_ids}
        for key in ground_truth_pairs:
            true_by_query.setdefault(key[0], set()).add(key)

        score = macro_f_beta(true_by_query, pred_by_query, beta=0.5)
        rows.append({"theta1": theta, "macro_f0.5": score})
    return pd.DataFrame(rows)
