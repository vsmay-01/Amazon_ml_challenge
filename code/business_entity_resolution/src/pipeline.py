"""
pipeline.py -- End-to-end orchestration (CLI).

    python -m src.pipeline train   --train-dir ... --work-dir ... --model-out ...
    python -m src.pipeline predict --test-dir  ... --model-in  ... --output-dir ...

`train` runs: leakage-safe split -> multi-pass blocking -> features ->
cross-source support -> hard-negative sampling -> fit model -> calibrate ->
sweep decision threshold on the held-out validation split -> save model +
calibrator + chosen policy + all reports.

`predict` runs the identical inference-time feature/candidate pipeline on
the test split and writes output/matching_results.tsv + candidate_pairs.tsv.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
import tempfile

import numpy as np
import pandas as pd

from . import blocking, evaluation, features, hard_negatives
from .calibration import ScoreCalibrator
from .decision import DecisionPolicy, apply_policy, sweep_theta1
from .io import GROUND_TRUTH_COLUMNS, SOURCE_COLUMNS, iter_tsv_batches, write_submission
from .model import PairwiseMatcher

RANDOM_SEED = 42
BATCH_SIZE = 32
DEFAULT_DATASET_DIR = os.environ.get(
    "ER_DATASET_DIR", "/kaggle/input/amazon-business-entity-resolution"
)


def _set_seed(seed: int = RANDOM_SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)


def _candidate_pairs_df(blocking_candidates: dict[str, set[tuple[str, str]]]) -> pd.DataFrame:
    rows = []
    for qid, cands in blocking_candidates.items():
        for source_name, cid in cands:
            rows.append({"source1_id": qid, "candidate_source": source_name, "candidate_id": cid})
    return pd.DataFrame(rows, columns=["source1_id", "candidate_source", "candidate_id"])


def _ground_truth_pair_set(ground_truth: pd.DataFrame) -> set[tuple[str, str, str]]:
    pairs = set()
    for _, row in ground_truth.iterrows():
        s1 = row.get("source1_id", "")
        if not s1:
            continue
        if row.get("source2_id", ""):
            pairs.add((s1, "source2", row["source2_id"]))
        if row.get("source3_id", ""):
            pairs.add((s1, "source3", row["source3_id"]))
    return pairs


def _ground_truth_by_query(ground_truth: pd.DataFrame, all_ids: list[str]) -> dict[str, set[tuple[str, str, str]]]:
    by_q: dict[str, set[tuple[str, str, str]]] = {qid: set() for qid in all_ids}
    for pair in _ground_truth_pair_set(ground_truth):
        by_q.setdefault(pair[0], set()).add(pair)
    return by_q


def build_candidates_and_features(source1: pd.DataFrame, source2: pd.DataFrame, source3: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    block_s2 = blocking.run_blocking(source1, source2, "source2")
    block_s3 = blocking.run_blocking(source1, source3, "source3")

    merged: dict[str, set[tuple[str, str]]] = {}
    for qid in source1["entity_id"]:
        merged[qid] = block_s2.candidates.get(qid, set()) | block_s3.candidates.get(qid, set())

    candidate_pairs = _candidate_pairs_df(merged)
    feature_table = features.build_feature_table(source1, source2, source3, candidate_pairs)
    feature_table = features.add_cross_source_support(feature_table, source2, source3)
    return candidate_pairs, feature_table


def _iter_feature_batches(
    data_dir: str,
    prefix: str,
    query_ids: set[str] | None = None,
    batch_size: int = BATCH_SIZE,
    index_dir: str | None = None,
):
    """Build disk-backed target indexes once, then stream query batches against them."""
    source2_path = os.path.join(data_dir, f"{prefix}_source2.tsv")
    source3_path = os.path.join(data_dir, f"{prefix}_source3.tsv")
    empty_source = pd.DataFrame(columns=SOURCE_COLUMNS)
    if index_dir is not None:
        os.makedirs(index_dir, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix="entity_resolution_index_", dir=index_dir
    ) as temporary_index_dir:
        source2_index = blocking.DiskTargetIndex(os.path.join(temporary_index_dir, "source2.sqlite"))
        source3_index = blocking.DiskTargetIndex(os.path.join(temporary_index_dir, "source3.sqlite"))
        try:
            for batch in iter_tsv_batches(source2_path, SOURCE_COLUMNS, batch_size):
                source2_index.add_batch(batch)
            for batch in iter_tsv_batches(source3_path, SOURCE_COLUMNS, batch_size):
                source3_index.add_batch(batch)
            source2_index.finalize()
            source3_index.finalize()

            for source1 in iter_tsv_batches(
                os.path.join(data_dir, f"{prefix}_source1.tsv"), SOURCE_COLUMNS, batch_size
            ):
                if query_ids is not None:
                    source1 = source1[source1["entity_id"].isin(query_ids)].reset_index(drop=True)
                if source1.empty:
                    continue

                merged: dict[str, set[tuple[str, str]]] = {}
                candidate_ids2: set[str] = set()
                candidate_ids3: set[str] = set()
                for _, query in source1.iterrows():
                    query_id = query["entity_id"]
                    ids2 = source2_index.candidate_ids(query)
                    ids3 = source3_index.candidate_ids(query)
                    candidate_ids2.update(ids2)
                    candidate_ids3.update(ids3)
                    merged[query_id] = (
                        {("source2", candidate_id) for candidate_id in ids2}
                        | {("source3", candidate_id) for candidate_id in ids3}
                    )

                source2 = source2_index.fetch_records(candidate_ids2)
                source3 = source3_index.fetch_records(candidate_ids3)
                candidate_pairs = _candidate_pairs_df(merged)
                feature_table = features.build_feature_table(
                    source1, source2, source3, candidate_pairs
                )
                feature_table = features.add_cross_source_support(
                    feature_table, source2, source3
                )
                yield source1["entity_id"].tolist(), feature_table
        finally:
            source2_index.close()
            source3_index.close()


def _read_ground_truth(data_dir: str, prefix: str) -> pd.DataFrame:
    path = os.path.join(data_dir, f"{prefix}_ground_truth.tsv")
    ground_truth = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)
    if set(GROUND_TRUTH_COLUMNS).issubset(ground_truth.columns):
        return ground_truth[GROUND_TRUTH_COLUMNS].fillna("").astype(str)

    actual_columns = {"source1_entity_id", "matched_entity_ids"}
    if not actual_columns.issubset(ground_truth.columns):
        raise ValueError(
            f"{path}: expected columns {GROUND_TRUTH_COLUMNS} or "
            f"{sorted(actual_columns)}; found {list(ground_truth.columns)}"
        )

    rows = []
    for row in ground_truth[["source1_entity_id", "matched_entity_ids"]].itertuples(index=False):
        source1_id = str(row.source1_entity_id).strip()
        if not source1_id:
            raise ValueError(f"{path}: source1_entity_id cannot be empty")
        for matched_id in str(row.matched_entity_ids).split(","):
            matched_id = matched_id.strip()
            if not matched_id:
                continue
            if matched_id.startswith("S2-"):
                source2_id, source3_id = matched_id, ""
            elif matched_id.startswith("S3-"):
                source2_id, source3_id = "", matched_id
            else:
                raise ValueError(
                    f"{path}: cannot determine source for matched ID {matched_id!r}; "
                    "expected an S2- or S3- prefix"
                )
            rows.append((source1_id, source2_id, source3_id))

    return pd.DataFrame(rows, columns=GROUND_TRUTH_COLUMNS)


def _read_source1_ids(data_dir: str, prefix: str, batch_size: int = BATCH_SIZE) -> list[str]:
    return [entity_id for batch in iter_tsv_batches(
        os.path.join(data_dir, f"{prefix}_source1.tsv"), SOURCE_COLUMNS, batch_size
    ) for entity_id in batch["entity_id"]]


def _training_signature(data_dir: str, train_ids: list[str], batch_size: int) -> str:
    digest = hashlib.sha256()
    digest.update("\n".join(sorted(train_ids)).encode("utf-8"))
    digest.update(str(batch_size).encode("ascii"))
    for filename in (
        "train_source1.tsv", "train_source2.tsv", "train_source3.tsv",
        "train_ground_truth.tsv",
    ):
        path = os.path.join(data_dir, filename)
        stat = os.stat(path)
        digest.update(f"{filename}:{stat.st_size}".encode("utf-8"))
    return digest.hexdigest()


def _atomic_joblib_dump(payload: dict, path: str) -> None:
    import joblib

    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    temporary_path = f"{path}.tmp"
    joblib.dump(payload, temporary_path)
    os.replace(temporary_path, path)


def cmd_train(args: argparse.Namespace) -> None:
    _set_seed()
    os.makedirs(args.work_dir, exist_ok=True)

    ground_truth = _read_ground_truth(args.train_dir, "train")
    all_s1_ids = _read_source1_ids(args.train_dir, "train")
    train_ids, val_ids = evaluation.leakage_safe_split(ground_truth, all_s1_ids)
    print(f"[train] {len(train_ids)} train / {len(val_ids)} val Source-1 entities (component-level split)")
    print("[train] scikit-learn estimator selected; fitting on CPU (CUDA/AMP do not apply to this estimator)")

    # ---- candidate generation + features (train side) ----
    gt_pair_set = _ground_truth_pair_set(ground_truth)
    signature = _training_signature(args.train_dir, train_ids, args.batch_size)
    completed_batches = 0
    training_already_complete = False
    model = PairwiseMatcher()
    if args.resume and os.path.exists(args.checkpoint_path):
        import joblib

        checkpoint = joblib.load(args.checkpoint_path)
        if checkpoint.get("signature") != signature:
            raise ValueError("Checkpoint does not match this dataset, split, or batch size.")
        model = checkpoint["model"]
        completed_batches = int(checkpoint["next_batch"])
        training_already_complete = bool(checkpoint.get("training_complete", False))
        print(f"[train] resumed checkpoint after {completed_batches} batches")

    training_rows = 0
    positive_rows = 0

    def training_batches():
        nonlocal training_rows, positive_rows
        seen_batches = 0
        for _, train_features in _iter_feature_batches(
            args.train_dir, "train", set(train_ids), args.batch_size, args.index_dir
        ):
            labeled = hard_negatives.label_pairs(train_features, gt_pair_set)
            training_set = hard_negatives.build_training_set(labeled)
            if training_set.empty:
                continue
            seen_batches += 1
            training_rows += len(training_set)
            positive_rows += int(training_set["label"].sum())
            if seen_batches <= completed_batches:
                continue
            del labeled
            yield training_set

    trained_batches = completed_batches

    def save_checkpoint(fitted_batch_number: int) -> None:
        nonlocal trained_batches
        trained_batches = completed_batches + fitted_batch_number
        next_batch = trained_batches
        if next_batch % args.checkpoint_every_batches == 0:
            _atomic_joblib_dump(
                {"model": model, "next_batch": next_batch, "signature": signature},
                args.checkpoint_path,
            )

    if not training_already_complete:
        model.fit_batches(training_batches(), on_batch_end=save_checkpoint)
    _atomic_joblib_dump(
        {
            "model": model,
            "next_batch": trained_batches,
            "signature": signature,
            "training_complete": True,
        },
        args.checkpoint_path,
    )
    print(f"[train] training rows: {training_rows} (positives={positive_rows})")
    print("[train] feature importances:")
    print(model.feature_importances().head(10).to_string())

    # ---- candidate generation + features (validation side) ----
    validation_tables = []
    for _, validation_batch in _iter_feature_batches(
        args.train_dir, "train", set(val_ids), args.batch_size, args.index_dir
    ):
        if validation_batch.empty:
            continue
        raw_scores = model.predict_proba(validation_batch)
        labels = hard_negatives.label_pairs(validation_batch, gt_pair_set)["label"].to_numpy()
        validation_tables.append(pd.DataFrame({
            "source1_id": validation_batch["source1_id"].to_numpy(),
            "candidate_source": validation_batch["candidate_source"].to_numpy(),
            "candidate_id": validation_batch["candidate_id"].to_numpy(),
            "raw_score": raw_scores,
            "label": labels,
        }))
    val_features = pd.concat(validation_tables, ignore_index=True) if validation_tables else pd.DataFrame()
    calibrator = ScoreCalibrator(method="isotonic")
    calibrator.fit(val_features["raw_score"].values, val_features["label"].values)
    val_features["calibrated_score"] = calibrator.transform(val_features["raw_score"].values)

    sweep = sweep_theta1(val_features, gt_pair_set, val_ids)
    sweep.to_csv(os.path.join(args.work_dir, "threshold_experiments.csv"), index=False)
    best_row = sweep.loc[sweep["macro_f0.5"].idxmax()]
    best_theta1 = float(best_row["theta1"])
    print(f"[train] best theta1={best_theta1} -> validation macro-F0.5={best_row['macro_f0.5']:.4f}")

    hand_test = evaluation.run_hand_test_cases()
    hand_test.to_csv(os.path.join(args.work_dir, "hand_test_cases.csv"), index=False)

    payload = {
        "model": model,
        "calibrator": calibrator,
        "theta1": best_theta1,
        "feature_columns": model.feature_columns,
        "config": {
            "batch_size": args.batch_size,
            "random_seed": RANDOM_SEED,
            "dataset_dir": args.dataset_dir,
            "train_dir": args.train_dir,
            "estimator": type(model.model).__name__,
            "compute_device": "cpu",
        },
    }
    _atomic_joblib_dump(payload, args.model_out)
    print(f"[train] saved model bundle to {args.model_out}")

    with open(os.path.join(args.work_dir, "training_manifest.json"), "w") as f:
        json.dump(
            {
                "random_seed": RANDOM_SEED,
                "n_train_entities": len(train_ids),
                "n_val_entities": len(val_ids),
                "best_theta1": best_theta1,
                "val_macro_f0_5": float(best_row["macro_f0.5"]),
            },
            f,
            indent=2,
        )


def cmd_predict(args: argparse.Namespace) -> None:
    _set_seed()
    import joblib

    payload = joblib.load(args.model_in)
    model: PairwiseMatcher = payload["model"]
    calibrator: ScoreCalibrator = payload["calibrator"]
    theta1: float = payload["theta1"]

    os.makedirs(args.output_dir, exist_ok=True)
    matches_path = os.path.join(args.output_dir, "matching_results.tsv")
    candidates_path = os.path.join(args.output_dir, "candidate_pairs.tsv")
    for output_path in (matches_path, candidates_path):
        if os.path.exists(output_path):
            os.remove(output_path)
    wrote_matches = False
    policy = DecisionPolicy(theta1=theta1)

    for query_ids, feature_table in _iter_feature_batches(
        args.test_dir, "test", batch_size=args.batch_size, index_dir=args.index_dir
    ):
        if feature_table.empty:
            accepted = feature_table
        else:
            feature_table["raw_score"] = model.predict_proba(feature_table)
            feature_table["calibrated_score"] = calibrator.transform(feature_table["raw_score"].values)
            accepted = apply_policy(feature_table, policy)

        accepted_ids = set(accepted["source1_id"]) if not accepted.empty else set()
        missing = set(query_ids) - accepted_ids
        singleton_rows = pd.DataFrame({
            "source1_id": sorted(missing), "matched_source": "", "matched_id": "", "score": 0.0,
        })
        matches = accepted.rename(columns={
            "candidate_source": "matched_source", "candidate_id": "matched_id", "calibrated_score": "score",
        })
        matches = matches[["source1_id", "matched_source", "matched_id"]] if not matches.empty else singleton_rows.iloc[0:0]
        matches = pd.concat([matches, singleton_rows], ignore_index=True).sort_values(
            ["source1_id", "matched_source"]
        )
        candidates = feature_table[["source1_id", "candidate_source", "candidate_id"]]
        write_submission(matches, candidates, args.output_dir, append=wrote_matches)
        wrote_matches = True

    if not wrote_matches:
        write_submission(
            pd.DataFrame(columns=["source1_id", "matched_source", "matched_id"]),
            pd.DataFrame(columns=["source1_id", "candidate_source", "candidate_id"]),
            args.output_dir,
        )
    print(f"[predict] wrote submission files to {args.output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="TriVote-ER++ pipeline")
    sub = parser.add_subparsers(dest="command", required=True)

    p_train = sub.add_parser("train")
    p_train.add_argument("--dataset-dir", default=DEFAULT_DATASET_DIR)
    p_train.add_argument("--train-dir")
    p_train.add_argument("--work-dir", default="/kaggle/working/reports")
    p_train.add_argument("--model-out", default="/kaggle/working/model.joblib")
    p_train.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    p_train.add_argument("--index-dir", default=os.environ.get("ER_INDEX_DIR", tempfile.gettempdir()))
    p_train.add_argument("--checkpoint-path", default="/kaggle/working/checkpoints/training.joblib")
    p_train.add_argument("--checkpoint-every-batches", type=int, default=25)
    p_train.add_argument("--resume", action="store_true")
    p_train.set_defaults(func=cmd_train)

    p_predict = sub.add_parser("predict")
    p_predict.add_argument("--dataset-dir", default=DEFAULT_DATASET_DIR)
    p_predict.add_argument("--test-dir")
    p_predict.add_argument("--model-in", default="/kaggle/working/model.joblib")
    p_predict.add_argument("--output-dir", default="/kaggle/working/output")
    p_predict.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    p_predict.add_argument("--index-dir", default=os.environ.get("ER_INDEX_DIR", tempfile.gettempdir()))
    p_predict.set_defaults(func=cmd_predict)

    args = parser.parse_args()
    if args.batch_size <= 0:
        parser.error("--batch-size must be positive")
    if args.command == "train":
        args.train_dir = args.train_dir or os.path.join(args.dataset_dir, "train")
        if args.checkpoint_every_batches <= 0:
            parser.error("--checkpoint-every-batches must be positive")
    else:
        args.test_dir = args.test_dir or os.path.join(args.dataset_dir, "test")
    args.func(args)


if __name__ == "__main__":
    main()
