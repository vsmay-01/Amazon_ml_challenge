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
import json
import os
import random
import sys

import numpy as np
import pandas as pd

from . import blocking, evaluation, features, hard_negatives
from .calibration import ScoreCalibrator
from .decision import DecisionPolicy, apply_policy, sweep_theta1
from .io import load_sources, write_submission
from .model import PairwiseMatcher

RANDOM_SEED = 42


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


def cmd_train(args: argparse.Namespace) -> None:
    _set_seed()
    os.makedirs(args.work_dir, exist_ok=True)

    data = load_sources(args.train_dir, "train", with_ground_truth=True)
    if data.ground_truth is None:
        print("ERROR: no train_ground_truth.tsv found in train dir.", file=sys.stderr)
        sys.exit(1)

    all_s1_ids = data.source1["entity_id"].tolist()
    train_ids, val_ids = evaluation.leakage_safe_split(data.ground_truth, all_s1_ids)
    print(f"[train] {len(train_ids)} train / {len(val_ids)} val Source-1 entities (component-level split)")

    s1_train = data.source1[data.source1["entity_id"].isin(train_ids)].reset_index(drop=True)
    s1_val = data.source1[data.source1["entity_id"].isin(val_ids)].reset_index(drop=True)

    # ---- candidate generation + features (train side) ----
    train_pairs, train_features = build_candidates_and_features(s1_train, data.source2, data.source3)
    gt_pair_set = _ground_truth_pair_set(data.ground_truth)
    labeled = hard_negatives.label_pairs(train_features, gt_pair_set)
    training_set = hard_negatives.build_training_set(labeled)
    print(f"[train] training rows: {len(training_set)} (positives={int(training_set['label'].sum())})")

    model = PairwiseMatcher()
    model.fit(training_set)
    print("[train] feature importances:")
    print(model.feature_importances().head(10).to_string())

    # ---- candidate generation + features (validation side) ----
    val_pairs, val_features = build_candidates_and_features(s1_val, data.source2, data.source3)
    val_features["raw_score"] = model.predict_proba(val_features)

    calibrator = ScoreCalibrator(method="isotonic")
    calib_labels = hard_negatives.label_pairs(val_features, gt_pair_set)["label"].values
    calibrator.fit(val_features["raw_score"].values, calib_labels)
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
    }
    import joblib

    joblib.dump(payload, args.model_out)
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

    data = load_sources(args.test_dir, "test", with_ground_truth=False)

    candidate_pairs, feature_table = build_candidates_and_features(data.source1, data.source2, data.source3)
    if feature_table.empty:
        print("[predict] WARNING: no candidates generated.", file=sys.stderr)

    feature_table["raw_score"] = model.predict_proba(feature_table) if not feature_table.empty else []
    feature_table["calibrated_score"] = (
        calibrator.transform(feature_table["raw_score"].values) if not feature_table.empty else []
    )

    policy = DecisionPolicy(theta1=theta1)
    accepted = apply_policy(feature_table, policy) if not feature_table.empty else feature_table

    # Ensure every Source 1 test entity appears exactly once, per compliance gates.
    all_ids = set(data.source1["entity_id"])
    accepted_ids = set(accepted["source1_id"]) if not accepted.empty else set()
    missing = all_ids - accepted_ids
    singleton_rows = pd.DataFrame(
        {
            "source1_id": sorted(missing),
            "matched_source": "",
            "matched_id": "",
            "score": 0.0,
        }
    )

    matches = accepted.rename(
        columns={"candidate_source": "matched_source", "candidate_id": "matched_id", "calibrated_score": "score"}
    )
    matches = matches[["source1_id", "matched_source", "matched_id", "score"]] if not matches.empty else matches
    matches = pd.concat([matches, singleton_rows], ignore_index=True) if not matches.empty else singleton_rows
    matches = matches.sort_values(["source1_id", "matched_source"]).reset_index(drop=True)

    write_submission(matches, candidate_pairs.assign(**{"candidate_source": candidate_pairs["candidate_source"]}), args.output_dir)
    print(f"[predict] wrote submission files to {args.output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="TriVote-ER++ pipeline")
    sub = parser.add_subparsers(dest="command", required=True)

    p_train = sub.add_parser("train")
    p_train.add_argument("--train-dir", required=True)
    p_train.add_argument("--work-dir", required=True)
    p_train.add_argument("--model-out", required=True)
    p_train.set_defaults(func=cmd_train)

    p_predict = sub.add_parser("predict")
    p_predict.add_argument("--test-dir", required=True)
    p_predict.add_argument("--model-in", required=True)
    p_predict.add_argument("--output-dir", required=True)
    p_predict.set_defaults(func=cmd_predict)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
