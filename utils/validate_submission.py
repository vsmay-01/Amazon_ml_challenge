#!/usr/bin/env python3
"""Validate challenge-format outputs using only the Python standard library."""
from __future__ import annotations

import argparse
import csv
import os
import sqlite3
import sys
import tempfile


MATCH_COLUMNS = ["source1_entity_id", "matched_entity_ids"]
CANDIDATE_COLUMNS = ["source1_entity_id", "candidate_entity_ids"]


def _read_header(path: str, expected: list[str], errors: list[str]):
    handle = open(path, "r", encoding="utf-8-sig", newline="")
    reader = csv.DictReader(handle, delimiter="\t")
    if reader.fieldnames != expected:
        errors.append(f"{path}: expected columns {expected}, found {reader.fieldnames}")
        handle.close()
        return None, None
    return handle, reader


def _split_ids(value: str) -> list[str]:
    if not value:
        return []
    return [entity_id.strip() for entity_id in value.split(",")]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--matching", required=True, help="Path to matching_results.tsv")
    parser.add_argument("--candidate", required=True, help="Path to candidate_pairs.tsv")
    parser.add_argument("--test-dir", required=True, help="Directory containing the three test TSVs")
    args = parser.parse_args()

    errors: list[str] = []
    paths = {
        "source1": os.path.join(args.test_dir, "test_source1.tsv"),
        "source2": os.path.join(args.test_dir, "test_source2.tsv"),
        "source3": os.path.join(args.test_dir, "test_source3.tsv"),
    }
    for path in [*paths.values(), args.matching, args.candidate]:
        if not os.path.isfile(path):
            errors.append(f"Required file not found: {path}")
    if errors:
        for number, error in enumerate(errors, 1):
            print(f"{number}. {error}")
        return 1

    with tempfile.TemporaryDirectory(prefix="submission_validation_") as temp_dir:
        connection = sqlite3.connect(os.path.join(temp_dir, "ids.sqlite"))
        connection.executescript(
            "CREATE TABLE valid_ids(entity_id TEXT PRIMARY KEY, source TEXT NOT NULL);"
            "CREATE TABLE seen_source1(entity_id TEXT PRIMARY KEY);"
            "CREATE TABLE seen_candidate_source1(entity_id TEXT PRIMARY KEY);"
            "CREATE TABLE predicted_ids(source1_id TEXT NOT NULL, entity_id TEXT NOT NULL, "
            "PRIMARY KEY(source1_id, entity_id));"
            "CREATE TABLE candidate_ids(source1_id TEXT NOT NULL, entity_id TEXT NOT NULL, "
            "PRIMARY KEY(source1_id, entity_id));"
        )
        source1_count = 0
        for source, path in paths.items():
            with open(path, "r", encoding="utf-8-sig", newline="") as handle:
                reader = csv.DictReader(handle, delimiter="\t")
                if not reader.fieldnames or "entity_id" not in reader.fieldnames:
                    errors.append(f"{path}: missing entity_id column")
                    continue
                for row in reader:
                    entity_id = (row.get("entity_id") or "").strip()
                    if not entity_id:
                        errors.append(f"{path}: empty entity_id found")
                        continue
                    try:
                        connection.execute(
                            "INSERT INTO valid_ids(entity_id, source) VALUES (?, ?)",
                            (entity_id, source),
                        )
                    except sqlite3.IntegrityError:
                        errors.append(f"Duplicate test entity_id: {entity_id}")
                    if source == "source1":
                        source1_count += 1
        connection.commit()

        matches_handle, matches_reader = _read_header(args.matching, MATCH_COLUMNS, errors)
        candidates_handle, candidates_reader = _read_header(args.candidate, CANDIDATE_COLUMNS, errors)

        def validate_output_rows(reader, list_column: str, label: str) -> None:
            seen_table = "seen_source1" if label == "matched" else "seen_candidate_source1"
            ids_table = "predicted_ids" if label == "matched" else "candidate_ids"
            for row_number, row in enumerate(reader, 1):
                source1_id = (row.get("source1_entity_id") or "").strip()
                if not source1_id:
                    errors.append(f"{label} output row {row_number}: source1_entity_id is empty")
                    continue
                if connection.execute(
                    "SELECT 1 FROM valid_ids WHERE entity_id = ? AND source = 'source1'",
                    (source1_id,),
                ).fetchone() is None:
                    errors.append(f"{label} output row {row_number}: unknown Source 1 ID {source1_id}")
                    continue
                try:
                    connection.execute(f"INSERT INTO {seen_table} VALUES (?)", (source1_id,))
                except sqlite3.IntegrityError:
                    errors.append(f"Duplicate {label} output row for Source 1 ID {source1_id}")

                entity_ids = _split_ids(row.get(list_column) or "")
                if any(not entity_id for entity_id in entity_ids):
                    errors.append(f"{label} output row {row_number}: empty ID in list")
                if len(entity_ids) != len(set(entity_ids)):
                    errors.append(f"{label} output row {row_number}: duplicate ID in list")
                for entity_id in entity_ids:
                    valid = connection.execute(
                        "SELECT source FROM valid_ids WHERE entity_id = ?", (entity_id,)
                    ).fetchone()
                    if valid is None or valid[0] not in ("source2", "source3"):
                        errors.append(f"{label} output row {row_number}: invalid ID {entity_id}")
                        continue
                    try:
                        connection.execute(
                            f"INSERT INTO {ids_table}(source1_id, entity_id) VALUES (?, ?)",
                            (source1_id, entity_id),
                        )
                    except sqlite3.IntegrityError:
                        errors.append(f"{label} output row {row_number}: duplicate ID {entity_id}")

        if matches_reader is not None:
            validate_output_rows(matches_reader, "matched_entity_ids", "matched")
        if candidates_reader is not None:
            validate_output_rows(candidates_reader, "candidate_entity_ids", "candidate")

        if matches_handle is not None:
            matches_handle.close()
        if candidates_handle is not None:
            candidates_handle.close()
        seen_count = connection.execute("SELECT COUNT(*) FROM seen_source1").fetchone()[0]
        if seen_count != source1_count:
            errors.append(
                f"Expected {source1_count} unique matching rows; found {seen_count}"
            )
        candidate_count = connection.execute("SELECT COUNT(*) FROM seen_candidate_source1").fetchone()[0]
        if candidate_count != source1_count:
            errors.append(
                f"Expected {source1_count} unique candidate rows; found {candidate_count}"
            )
        missing_matches = connection.execute(
            "SELECT entity_id FROM valid_ids WHERE source = 'source1' "
            "EXCEPT SELECT entity_id FROM seen_source1 LIMIT 5"
        ).fetchall()
        missing_candidates = connection.execute(
            "SELECT entity_id FROM valid_ids WHERE source = 'source1' "
            "EXCEPT SELECT entity_id FROM seen_candidate_source1 LIMIT 5"
        ).fetchall()
        if missing_matches:
            errors.append(f"Source 1 entities missing matching rows: {[row[0] for row in missing_matches]}")
        if missing_candidates:
            errors.append(f"Source 1 entities missing candidate rows: {[row[0] for row in missing_candidates]}")
        absent_candidates = connection.execute(
            "SELECT predicted_ids.source1_id, predicted_ids.entity_id FROM predicted_ids "
            "LEFT JOIN candidate_ids USING(source1_id, entity_id) "
            "WHERE candidate_ids.entity_id IS NULL LIMIT 5"
        ).fetchall()
        if absent_candidates:
            errors.append(f"Matches absent from candidate lists: {absent_candidates}")
        connection.close()

    if errors:
        print("SUBMISSION VALIDATION FAILED:")
        for number, error in enumerate(errors, 1):
            print(f"{number}. {error}")
        return 1

    print("PASS: submission files have the required schema; every Source 1 ID "
          "appears exactly once; IDs exist in the test sources; matches are candidates.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
