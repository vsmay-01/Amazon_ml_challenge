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

import math
import sqlite3
from collections import Counter
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


class DiskTargetIndex:
    """SQLite-backed blocking index built in a single streaming target pass."""

    def __init__(self, path: str):
        self.connection = sqlite3.connect(path)
        self.connection.execute("PRAGMA journal_mode=WAL")
        self.connection.execute("PRAGMA synchronous=NORMAL")
        self.connection.executescript(
            """
            CREATE TABLE records (
                entity_id TEXT PRIMARY KEY,
                business_name TEXT NOT NULL,
                business_address TEXT NOT NULL,
                country TEXT NOT NULL,
                ordinal INTEGER NOT NULL,
                tfidf_norm REAL NOT NULL DEFAULT 0
            );
            CREATE TABLE postings (
                kind TEXT NOT NULL,
                key TEXT NOT NULL,
                entity_id TEXT NOT NULL,
                PRIMARY KEY (kind, key, entity_id)
            );
            CREATE INDEX postings_lookup ON postings(kind, key);
            CREATE TABLE tfidf_terms (
                entity_id TEXT NOT NULL,
                term TEXT NOT NULL,
                tf INTEGER NOT NULL,
                PRIMARY KEY (entity_id, term)
            );
            CREATE INDEX tfidf_term_lookup ON tfidf_terms(term, entity_id);
            CREATE TABLE term_df (
                term TEXT PRIMARY KEY,
                doc_freq INTEGER NOT NULL,
                idf REAL NOT NULL DEFAULT 0
            );
            CREATE TEMP TABLE query_keys (key TEXT PRIMARY KEY);
            CREATE TEMP TABLE query_terms (
                term TEXT PRIMARY KEY,
                tf INTEGER NOT NULL,
                idf REAL NOT NULL
            );
            CREATE TEMP TABLE query_ids (entity_id TEXT PRIMARY KEY);
            """
        )
        self._analyzer = TfidfVectorizer(
            analyzer="char_wb", ngram_range=(2, 4)
        ).build_analyzer()
        self._document_count = 0
        self._nonempty_document_count = 0

    def add_batch(self, target: pd.DataFrame) -> None:
        records = []
        postings = []
        tfidf_rows = []
        document_frequencies = Counter()

        for row in target.itertuples(index=False):
            entity_id = str(row.entity_id)
            name = str(row.business_name)
            address = str(row.business_address)
            country = str(row.country)
            ordinal = self._document_count + len(records)
            normalized_name = normalize_name(name)
            normalized_address = normalize_address(address)
            records.append((entity_id, name, address, country, ordinal))

            indexed_keys = (
                ("exact_name", [normalized_name.no_suffix]),
                ("name_token", [t for t in normalized_name.no_suffix.split() if len(t) >= 2]),
                ("name_char_ngram", normalized_name.char_ngrams),
                ("address_token", [t for t in normalized_address.tokens if len(t) >= 3]),
                ("address_char", normalized_address.char_ngrams),
            )
            for kind, keys in indexed_keys:
                postings.extend((kind, key, entity_id) for key in set(keys) if key)

            document = normalized_name.no_suffix + " " + normalized_address.norm
            term_counts = Counter(self._analyzer(document))
            self._nonempty_document_count += bool(term_counts)
            tfidf_rows.extend((entity_id, term, count) for term, count in term_counts.items())
            document_frequencies.update(term_counts.keys())

        try:
            with self.connection:
                self.connection.executemany(
                    "INSERT INTO records(entity_id, business_name, business_address, country, ordinal) "
                    "VALUES (?, ?, ?, ?, ?)",
                    records,
                )
                self.connection.executemany(
                    "INSERT INTO postings(kind, key, entity_id) VALUES (?, ?, ?)", postings
                )
                self.connection.executemany(
                    "INSERT INTO tfidf_terms(entity_id, term, tf) VALUES (?, ?, ?)", tfidf_rows
                )
                self.connection.executemany(
                    "INSERT INTO term_df(term, doc_freq) VALUES (?, ?) "
                    "ON CONFLICT(term) DO UPDATE SET doc_freq = doc_freq + excluded.doc_freq",
                    document_frequencies.items(),
                )
        except sqlite3.IntegrityError as error:
            raise ValueError("Target source contains duplicate entity_id values") from error
        self._document_count += len(records)

    def finalize(self) -> None:
        if not self._document_count:
            return
        self.connection.execute(
            "CREATE TEMP TABLE term_df_snapshot AS SELECT term, doc_freq FROM term_df"
        )
        cursor = self.connection.execute("SELECT term, doc_freq FROM term_df_snapshot")
        while terms := cursor.fetchmany(10000):
            with self.connection:
                self.connection.executemany(
                    "UPDATE term_df SET idf = ? WHERE term = ?",
                    (
                        (math.log((1 + self._document_count) / (1 + doc_freq)) + 1, term)
                        for term, doc_freq in terms
                    ),
                )
        self.connection.execute("DROP TABLE term_df_snapshot")
        self.connection.execute(
            "UPDATE records SET tfidf_norm = COALESCE(("
            "SELECT sqrt(SUM(tfidf_terms.tf * term_df.idf * tfidf_terms.tf * term_df.idf)) "
            "FROM tfidf_terms JOIN term_df USING(term) "
            "WHERE tfidf_terms.entity_id = records.entity_id), 0)"
        )
        self.connection.commit()

    def _lookup(self, kind: str, keys: Iterable[str], minimum_shared: int = 1) -> set[str]:
        unique_keys = {key for key in keys if key}
        if not unique_keys:
            return set()
        self.connection.execute("DELETE FROM query_keys")
        self.connection.executemany(
            "INSERT INTO query_keys(key) VALUES (?)", ((key,) for key in unique_keys)
        )
        rows = self.connection.execute(
            "SELECT postings.entity_id FROM postings "
            "JOIN query_keys ON postings.key = query_keys.key "
            "WHERE postings.kind = ? GROUP BY postings.entity_id "
            "HAVING COUNT(*) >= ?",
            (kind, minimum_shared),
        )
        return {row[0] for row in rows}

    def _tfidf_candidates(self, name: str, address: str, top_k: int = 20) -> set[str]:
        if not self._nonempty_document_count:
            return set()
        document = normalize_name(name).no_suffix + " " + normalize_address(address).norm
        term_counts = Counter(self._analyzer(document))
        self.connection.execute("DELETE FROM query_terms")
        self.connection.executemany(
            "INSERT INTO query_terms(term, tf, idf) "
            "SELECT ?, ?, idf FROM term_df WHERE term = ?",
            ((term, tf, term) for term, tf in term_counts.items()),
        )
        query_norm = self.connection.execute(
            "SELECT sqrt(COALESCE(SUM(tf * idf * tf * idf), 0)) FROM query_terms"
        ).fetchone()[0]
        if query_norm:
            rows = self.connection.execute(
                "WITH scores AS ("
                "SELECT tfidf_terms.entity_id, "
                "SUM(query_terms.tf * tfidf_terms.tf * term_df.idf * term_df.idf) "
                "/ (? * records.tfidf_norm) AS cosine "
                "FROM query_terms JOIN tfidf_terms USING(term) "
                "JOIN term_df USING(term) JOIN records USING(entity_id) "
                "WHERE records.tfidf_norm > 0 GROUP BY tfidf_terms.entity_id) "
                "SELECT records.entity_id, COALESCE(scores.cosine, 0) AS cosine "
                "FROM records LEFT JOIN scores USING(entity_id) "
                "ORDER BY cosine DESC, records.ordinal LIMIT ?",
                (query_norm, top_k),
            )
        else:
            rows = self.connection.execute(
                "SELECT entity_id, 0 FROM records ORDER BY ordinal LIMIT ?", (top_k,)
            )
        return {row[0] for row in rows}

    def candidate_ids(self, row: pd.Series) -> set[str]:
        name = normalize_name(row["business_name"])
        address = normalize_address(row["business_address"])
        candidates = set()
        candidates.update(self._lookup("exact_name", [name.no_suffix]))
        candidates.update(self._lookup("name_token", (t for t in name.no_suffix.split() if len(t) >= 2)))
        candidates.update(self._lookup("name_char_ngram", name.char_ngrams, minimum_shared=2))
        candidates.update(self._lookup("address_token", (t for t in address.tokens if len(t) >= 3)))
        candidates.update(self._lookup("address_char", address.char_ngrams, minimum_shared=3))
        candidates.update(self._tfidf_candidates(row["business_name"], row["business_address"]))
        return candidates

    def fetch_records(self, entity_ids: Iterable[str]) -> pd.DataFrame:
        ids = set(entity_ids)
        if not ids:
            return pd.DataFrame(columns=["entity_id", "business_name", "business_address", "country"])
        self.connection.execute("DELETE FROM query_ids")
        self.connection.executemany(
            "INSERT INTO query_ids(entity_id) VALUES (?)", ((entity_id,) for entity_id in ids)
        )
        return pd.read_sql_query(
            "SELECT records.entity_id, business_name, business_address, country "
            "FROM records JOIN query_ids USING(entity_id)",
            self.connection,
        )

    def close(self) -> None:
        self.connection.close()


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
