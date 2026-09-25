"""
blocking.py -- Multi-pass adaptive blocking (Innovation 1).

Implements the blocking views from Section 6 of the design doc:
  1. Exact normalized name
  2. Name token overlap
  3. Character n-gram retrieval
  4. Address token retrieval
  5. Address character retrieval
  6. TF-IDF retrieval
  (7. Optional embedding retrieval is left as an extension point.)

Also implements the marginal-recall accounting (Section 6.2) used to decide
whether a blocking pass earns its place in the final union.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Callable, Iterable

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors

from .normalization import normalize_address, normalize_name

BlockKey = str


@dataclass
class BlockingResult:
    # candidate[q_id] = set of (source_name, candidate_id)
    candidates: dict[str, set[tuple[str, str]]]
    per_pass_candidates: dict[str, dict[str, set[tuple[str, str]]]]


def _index_by_key(df: pd.DataFrame, key_fn: Callable[[pd.Series], Iterable[str]]) -> dict[str, list[str]]:
    index: dict[str, list[str]] = defaultdict(list)
    for _, row in df.iterrows():
        for key in key_fn(row):
            if key:
                index[key].append(row["entity_id"])
    return index


def _exact_name_pass(query_df: pd.DataFrame, target_df: pd.DataFrame) -> dict[str, set[str]]:
    target_index = _index_by_key(
        target_df, lambda r: [normalize_name(r["business_name"]).no_suffix]
    )
    result: dict[str, set[str]] = {}
    for _, row in query_df.iterrows():
        key = normalize_name(row["business_name"]).no_suffix
        result[row["entity_id"]] = set(target_index.get(key, []))
    return result


def _token_overlap_pass(query_df: pd.DataFrame, target_df: pd.DataFrame) -> dict[str, set[str]]:
    token_index: dict[str, list[str]] = defaultdict(list)
    for _, row in target_df.iterrows():
        for tok in normalize_name(row["business_name"]).no_suffix.split():
            if len(tok) >= 2:
                token_index[tok].append(row["entity_id"])

    result: dict[str, set[str]] = {}
    for _, row in query_df.iterrows():
        cands: set[str] = set()
        for tok in normalize_name(row["business_name"]).no_suffix.split():
            if len(tok) >= 2:
                cands.update(token_index.get(tok, []))
        result[row["entity_id"]] = cands
    return result


def _char_ngram_pass(query_df: pd.DataFrame, target_df: pd.DataFrame, n: int = 3, min_shared: int = 2) -> dict[str, set[str]]:
    ngram_index: dict[str, list[str]] = defaultdict(list)
    for _, row in target_df.iterrows():
        for g in normalize_name(row["business_name"]).char_ngrams:
            ngram_index[g].append(row["entity_id"])

    result: dict[str, set[str]] = {}
    for _, row in query_df.iterrows():
        counts: dict[str, int] = defaultdict(int)
        for g in normalize_name(row["business_name"]).char_ngrams:
            for tid in ngram_index.get(g, []):
                counts[tid] += 1
        result[row["entity_id"]] = {tid for tid, c in counts.items() if c >= min_shared}
    return result


def _address_token_pass(query_df: pd.DataFrame, target_df: pd.DataFrame) -> dict[str, set[str]]:
    token_index: dict[str, list[str]] = defaultdict(list)
    for _, row in target_df.iterrows():
        for tok in normalize_address(row["business_address"]).tokens:
            if len(tok) >= 3:
                token_index[tok].append(row["entity_id"])

    result: dict[str, set[str]] = {}
    for _, row in query_df.iterrows():
        cands: set[str] = set()
        for tok in normalize_address(row["business_address"]).tokens:
            if len(tok) >= 3:
                cands.update(token_index.get(tok, []))
        result[row["entity_id"]] = cands
    return result


def _address_char_pass(query_df: pd.DataFrame, target_df: pd.DataFrame, min_shared: int = 3) -> dict[str, set[str]]:
    ngram_index: dict[str, list[str]] = defaultdict(list)
    for _, row in target_df.iterrows():
        for g in normalize_address(row["business_address"]).char_ngrams:
            ngram_index[g].append(row["entity_id"])

    result: dict[str, set[str]] = {}
    for _, row in query_df.iterrows():
        counts: dict[str, int] = defaultdict(int)
        for g in normalize_address(row["business_address"]).char_ngrams:
            for tid in ngram_index.get(g, []):
                counts[tid] += 1
        result[row["entity_id"]] = {tid for tid, c in counts.items() if c >= min_shared}
    return result


def _tfidf_pass(query_df: pd.DataFrame, target_df: pd.DataFrame, top_k: int = 20) -> dict[str, set[str]]:
    def doc(row):
        return normalize_name(row["business_name"]).no_suffix + " " + normalize_address(row["business_address"]).norm

    target_docs = [doc(r) for _, r in target_df.iterrows()]
    target_ids = target_df["entity_id"].tolist()
    if not target_docs or all(not d.strip() for d in target_docs):
        return {qid: set() for qid in query_df["entity_id"]}

    vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), min_df=1)
    target_matrix = vectorizer.fit_transform(target_docs)

    k = min(top_k, len(target_ids))
    nn = NearestNeighbors(n_neighbors=k, metric="cosine")
    nn.fit(target_matrix)

    query_docs = [doc(r) for _, r in query_df.iterrows()]
    query_matrix = vectorizer.transform(query_docs)
    _, indices = nn.kneighbors(query_matrix)

    result: dict[str, set[str]] = {}
    for qid, idx_row in zip(query_df["entity_id"], indices):
        result[qid] = {target_ids[i] for i in idx_row}
    return result


BLOCKING_PASSES = {
    "exact_name": _exact_name_pass,
    "name_token": _token_overlap_pass,
    "name_char_ngram": _char_ngram_pass,
    "address_token": _address_token_pass,
    "address_char": _address_char_pass,
    "tfidf": _tfidf_pass,
}


def run_blocking(
    source1: pd.DataFrame,
    target: pd.DataFrame,
    target_source_name: str,
    passes: list[str] | None = None,
) -> BlockingResult:
    """
    Run the requested blocking passes of Source 1 against one target source
    (Source 2 or Source 3) and union the results.
    """
    passes = passes or list(BLOCKING_PASSES.keys())
    per_pass: dict[str, dict[str, set[tuple[str, str]]]] = {}
    union: dict[str, set[tuple[str, str]]] = defaultdict(set)

    for pass_name in passes:
        fn = BLOCKING_PASSES[pass_name]
        raw = fn(source1, target)
        tagged = {
            qid: {(target_source_name, cid) for cid in cids} for qid, cids in raw.items()
        }
        per_pass[pass_name] = tagged
        for qid, cids in tagged.items():
            union[qid].update(cids)

    return BlockingResult(candidates=dict(union), per_pass_candidates=per_pass)


def marginal_recall(
    per_pass_candidates: dict[str, dict[str, set[tuple[str, str]]]],
    pass_order: list[str],
    ground_truth: dict[str, set[tuple[str, str]]],
) -> pd.DataFrame:
    """
    Compute marginal candidate-recall contribution (Section 6.2) of each
    blocking pass, in the given order, plus AvgCand / P95Cand / cost stats
    for the cumulative union up to that pass.
    """
    rows = []
    cumulative: dict[str, set[tuple[str, str]]] = defaultdict(set)
    prev_recall = 0.0

    all_qids = list(ground_truth.keys())
    total_true = sum(len(v) for v in ground_truth.values())

    for pass_name in pass_order:
        pass_cands = per_pass_candidates.get(pass_name, {})
        for qid, cids in pass_cands.items():
            cumulative[qid].update(cids)

        found = 0
        cand_counts = []
        for qid in all_qids:
            true_set = ground_truth[qid]
            cand_set = cumulative.get(qid, set())
            found += len(true_set & cand_set)
            cand_counts.append(len(cand_set))

        recall = found / total_true if total_true else 0.0
        cand_series = pd.Series(cand_counts) if cand_counts else pd.Series([0])

        rows.append(
            {
                "pass": pass_name,
                "cumulative_recall": recall,
                "marginal_recall": recall - prev_recall,
                "avg_candidates": cand_series.mean(),
                "p95_candidates": cand_series.quantile(0.95),
            }
        )
        prev_recall = recall

    return pd.DataFrame(rows)
