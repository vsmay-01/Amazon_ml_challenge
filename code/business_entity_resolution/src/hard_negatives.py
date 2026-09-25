"""
hard_negatives.py -- Hard-negative-aware sampling (Innovation 4, Section 9).

Given the real candidate distribution (post-blocking) and ground truth,
split negatives into "easy" and "hard" based on similarity signals that
correlate with false merges:
  - high normalized-name similarity but incorrect identity
  - high address similarity but incorrect identity
  - same country and similar business type (proxied by name similarity)
  - high TF-IDF / retrieval score but negative ground truth

The InfoNCE-style contrastive loss described in the design doc (Eq. 11) is
provided as an optional training objective in model.py; this module only
handles *sampling*, which is the part that matters regardless of which
downstream loss/model is ultimately used.
"""
from __future__ import annotations

import pandas as pd

HARD_NEGATIVE_THRESHOLDS = {
    "name_edit_sim": 0.75,
    "name_token_jaccard": 0.5,
    "address_token_overlap": 0.5,
    "name_tfidf_cosine": 0.7,
}


def label_pairs(feature_table: pd.DataFrame, ground_truth_pairs: set[tuple[str, str, str]]) -> pd.DataFrame:
    """
    ground_truth_pairs: set of (source1_id, candidate_source, candidate_id)
    Adds a binary `label` column.
    """
    out = feature_table.copy()

    def is_match(row) -> int:
        key = (row["source1_id"], row["candidate_source"], row["candidate_id"])
        return int(key in ground_truth_pairs)

    out["label"] = out.apply(is_match, axis=1)
    return out


def is_hard_negative(row: pd.Series) -> bool:
    if row.get("label", 0) == 1:
        return False
    score = 0
    for col, thresh in HARD_NEGATIVE_THRESHOLDS.items():
        if col in row and row[col] >= thresh:
            score += 1
    return score >= 2  # at least two strong-similarity signals but wrong identity


def split_easy_hard_negatives(labeled_table: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    negatives = labeled_table[labeled_table["label"] == 0].copy()
    hard_mask = negatives.apply(is_hard_negative, axis=1)
    hard = negatives[hard_mask]
    easy = negatives[~hard_mask]
    return easy, hard


def build_training_set(
    labeled_table: pd.DataFrame,
    hard_negative_ratio: float = 0.6,
    easy_to_positive_ratio: float = 1.0,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Construct a training set of positives + a mix of hard and easy negatives.
    `hard_negative_ratio` controls what fraction of sampled negatives per
    positive are hard negatives (rounded).
    """
    positives = labeled_table[labeled_table["label"] == 1]
    easy, hard = split_easy_hard_negatives(labeled_table)

    n_pos = len(positives)
    n_neg_total = int(round(n_pos * (1 + easy_to_positive_ratio)))
    n_hard = int(round(n_neg_total * hard_negative_ratio))
    n_easy = max(n_neg_total - n_hard, 0)

    hard_sample = hard.sample(n=min(n_hard, len(hard)), random_state=random_state) if len(hard) else hard
    easy_sample = easy.sample(n=min(n_easy, len(easy)), random_state=random_state) if len(easy) else easy

    training = pd.concat([positives, hard_sample, easy_sample], ignore_index=True)
    return training.sample(frac=1.0, random_state=random_state).reset_index(drop=True)
