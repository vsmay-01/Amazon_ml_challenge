"""
features.py -- Pairwise + triadic (cross-source) feature engineering
(Section 12: Feature Architecture).

Feature families: Name, Address, Country, Source, Missingness, Cross-source,
Query-context. Query-context features are added later in decision.py once
pair scores exist; this module produces the pair-level feature matrix.
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from rapidfuzz import fuzz
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .normalization import normalize_address, normalize_country, normalize_name

PAIR_FEATURE_COLUMNS = [
    # Name
    "name_exact", "name_token_jaccard", "name_edit_sim", "name_char_ngram_sim",
    "name_tfidf_cosine", "name_token_order_diff", "name_legal_suffix_match",
    # Address
    "address_exact", "address_token_overlap", "address_char_sim",
    "address_tfidf_cosine", "address_component_agreement", "address_completeness",
    # Country
    "country_agreement", "country_unknown",
    # Source
    "is_source2", "is_source3",
    # Missingness
    "name_missing", "address_missing", "address_partial",
]


def _jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def _tfidf_cosine(a: str, b: str) -> float:
    if not a.strip() and not b.strip():
        return 1.0
    if not a.strip() or not b.strip():
        return 0.0
    try:
        vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4))
        m = vec.fit_transform([a, b])
        return float(cosine_similarity(m[0], m[1])[0][0])
    except ValueError:
        return 0.0


@dataclass
class PairRecord:
    source1_id: str
    candidate_source: str  # "source2" or "source3"
    candidate_id: str


def compute_pair_features(
    row1: pd.Series,
    row_cand: pd.Series,
    candidate_source: str,
) -> dict:
    n1 = normalize_name(row1["business_name"])
    n2 = normalize_name(row_cand["business_name"])
    a1 = normalize_address(row1["business_address"])
    a2 = normalize_address(row_cand["business_address"])
    c1 = normalize_country(row1["country"])
    c2 = normalize_country(row_cand["country"])

    name_token_set1 = set(n1.no_suffix.split())
    name_token_set2 = set(n2.no_suffix.split())

    feats = {
        # Name
        "name_exact": float(n1.no_suffix == n2.no_suffix and n1.no_suffix != ""),
        "name_token_jaccard": _jaccard(name_token_set1, name_token_set2),
        "name_edit_sim": fuzz.ratio(n1.no_suffix, n2.no_suffix) / 100.0,
        "name_char_ngram_sim": _jaccard(n1.char_ngrams, n2.char_ngrams),
        "name_tfidf_cosine": _tfidf_cosine(n1.no_suffix, n2.no_suffix),
        "name_token_order_diff": float(n1.no_suffix != n2.no_suffix and n1.token_sorted == n2.token_sorted),
        "name_legal_suffix_match": float(n1.norm != n1.no_suffix and n2.norm != n2.no_suffix),
        # Address
        "address_exact": float(a1.norm == a2.norm and a1.norm != ""),
        "address_token_overlap": _jaccard(set(a1.tokens), set(a2.tokens)),
        "address_char_sim": _jaccard(a1.char_ngrams, a2.char_ngrams),
        "address_tfidf_cosine": _tfidf_cosine(a1.norm, a2.norm),
        "address_component_agreement": _jaccard(set(a1.abbreviated.split()), set(a2.abbreviated.split())),
        "address_completeness": float(bool(a1.norm) and bool(a2.norm)),
        # Country
        "country_agreement": float(c1 == c2 and c1 != ""),
        "country_unknown": float(c1 == "" or c2 == ""),
        # Source
        "is_source2": float(candidate_source == "source2"),
        "is_source3": float(candidate_source == "source3"),
        # Missingness
        "name_missing": float(n1.norm == "" or n2.norm == ""),
        "address_missing": float(a1.norm == "" or a2.norm == ""),
        "address_partial": float(0 < len(a1.tokens) < 3 or 0 < len(a2.tokens) < 3),
    }
    return feats


def build_feature_table(
    source1: pd.DataFrame,
    source2: pd.DataFrame,
    source3: pd.DataFrame,
    candidate_pairs: pd.DataFrame,
) -> pd.DataFrame:
    """
    candidate_pairs: columns [source1_id, candidate_source, candidate_id]
    Returns a DataFrame with PAIR_FEATURE_COLUMNS plus the id columns.
    """
    s1_index = source1.set_index("entity_id")
    s2_index = source2.set_index("entity_id")
    s3_index = source3.set_index("entity_id")

    rows = []
    for _, pair in candidate_pairs.iterrows():
        row1 = s1_index.loc[pair["source1_id"]]
        target_index = s2_index if pair["candidate_source"] == "source2" else s3_index
        if pair["candidate_id"] not in target_index.index:
            continue
        row_cand = target_index.loc[pair["candidate_id"]]
        feats = compute_pair_features(row1, row_cand, pair["candidate_source"])
        feats["source1_id"] = pair["source1_id"]
        feats["candidate_source"] = pair["candidate_source"]
        feats["candidate_id"] = pair["candidate_id"]
        rows.append(feats)

    cols = ["source1_id", "candidate_source", "candidate_id"] + PAIR_FEATURE_COLUMNS
    if not rows:
        return pd.DataFrame(columns=cols)
    return pd.DataFrame(rows)[cols]


def add_cross_source_support(
    feature_table: pd.DataFrame,
    source2: pd.DataFrame,
    source3: pd.DataFrame,
) -> pd.DataFrame:
    """
    Triadic evidence (Section 7): for each S1<->S2 candidate, find the
    strongest independent S2<->S3 support among the *other* S3 candidates
    already proposed for the same S1 query, and vice versa. This is used as
    evidence (a feature), never as an unconditional merge rule.
    """
    s2_index = source2.set_index("entity_id")
    s3_index = source3.set_index("entity_id")

    out = feature_table.copy()
    out["cross_support"] = 0.0
    out["cross_support_count"] = 0

    for q_id, group in out.groupby("source1_id"):
        s2_rows = group[group["candidate_source"] == "source2"]
        s3_rows = group[group["candidate_source"] == "source3"]
        if s2_rows.empty or s3_rows.empty:
            continue

        for idx2, r2 in s2_rows.iterrows():
            if r2["candidate_id"] not in s2_index.index:
                continue
            name2 = normalize_name(s2_index.loc[r2["candidate_id"], "business_name"]).no_suffix
            best_sim = 0.0
            support_count = 0
            for idx3, r3 in s3_rows.iterrows():
                if r3["candidate_id"] not in s3_index.index:
                    continue
                name3 = normalize_name(s3_index.loc[r3["candidate_id"], "business_name"]).no_suffix
                sim = fuzz.ratio(name2, name3) / 100.0
                best_sim = max(best_sim, sim)
                if sim > 0.85:
                    support_count += 1
            out.loc[idx2, "cross_support"] = best_sim
            out.loc[idx2, "cross_support_count"] = support_count

        for idx3, r3 in s3_rows.iterrows():
            if r3["candidate_id"] not in s3_index.index:
                continue
            name3 = normalize_name(s3_index.loc[r3["candidate_id"], "business_name"]).no_suffix
            best_sim = 0.0
            support_count = 0
            for idx2, r2 in s2_rows.iterrows():
                if r2["candidate_id"] not in s2_index.index:
                    continue
                name2 = normalize_name(s2_index.loc[r2["candidate_id"], "business_name"]).no_suffix
                sim = fuzz.ratio(name2, name3) / 100.0
                best_sim = max(best_sim, sim)
                if sim > 0.85:
                    support_count += 1
            out.loc[idx3, "cross_support"] = best_sim
            out.loc[idx3, "cross_support_count"] = support_count

    return out


FULL_FEATURE_COLUMNS = PAIR_FEATURE_COLUMNS + ["cross_support", "cross_support_count"]
