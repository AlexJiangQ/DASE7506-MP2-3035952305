"""Stage 3 CREO development using train OOF and validation only.

No test input or reference is opened by this script.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
import shutil
import time

import numpy as np

from common import DATA, load_inputs, read_jsonl
from creo_features import (
    extract_group_features,
    gate_feature_matrices,
    group_argmax_choices,
    group_labels,
    group_to_candidate_choices,
    plurality_group_indices,
    shrinkage_pool_choices,
)
from creo_model import (
    fit_logistic,
    gate_group_choices,
    logistic_logits,
    logistic_probabilities,
    save_creo_bundle,
    score_group_model,
    train_group_model,
)
from selector_features import extract_feature_blocks, hard_group_mask, labels_from_references
from selector_model import load_model, score_model


SEED = 20261022
RUNS = Path("runs/stage3")
STAGE1_OOF = Path("runs/cv/hand_centered__h0.01__e1__listwise/oof_predictions.npz")
STAGE2_OOF = Path("runs/stage2/cv/hand_centered__uniform__pair0.1/oof_predictions.npz")
STAGE1_CHECKPOINT = Path("runs/provisional_model_bundle")
STAGE2_CHECKPOINT = Path("runs/final/frozen_checkpoint")


def dump_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def slice_blocks(blocks: dict, indices: np.ndarray) -> dict:
    result = {}
    for key, value in blocks.items():
        if isinstance(value, np.ndarray):
            result[key] = value[indices]
        elif isinstance(value, list):
            result[key] = [value[int(index)] for index in indices]
    return result


def fold_stats(values: list[float]) -> tuple[float, float]:
    array = np.asarray(values, dtype=np.float64)
    return float(array.mean()), float(array.std(ddof=1) / math.sqrt(len(array)))


def group_accuracy(labels: np.ndarray, choices: np.ndarray, mask: np.ndarray | None = None) -> float:
    if mask is None:
        mask = np.ones(len(labels), dtype=bool)
    indices = np.flatnonzero(mask)
    return 100.0 * float(np.mean(labels[indices, choices[indices]]))


def candidate_choices_to_groups(choices: list[int] | np.ndarray, group_data: dict) -> np.ndarray:
    result = np.zeros(len(choices), dtype=np.int64)
    for i, choice in enumerate(choices):
        result[i] = next(
            g for g, members in enumerate(group_data["memberships"][i])
            if int(choice) in members
        )
    return result


def masked_candidate_argmax(scores: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Candidate argmax that matches the official invalid-answer masking rule."""
    scores = np.asarray(scores, dtype=np.float64)
    valid = np.asarray(valid, dtype=bool)
    masked = np.where(valid, scores, -np.inf)
    choices = np.argmax(masked, axis=1).astype(np.int64)
    choices[~valid.any(axis=1)] = 0
    return choices


def decision_metrics(
    labels: np.ndarray,
    selected: np.ndarray,
    plurality: np.ndarray,
    hard_mask: np.ndarray | None = None,
) -> dict:
    if hard_mask is None:
        hard_mask = np.ones(len(labels), dtype=bool)
    selected_correct = labels[np.arange(len(labels)), selected]
    plurality_correct = labels[np.arange(len(labels)), plurality]
    overturn = selected != plurality
    good = overturn & ~plurality_correct & selected_correct
    bad = overturn & plurality_correct & ~selected_correct
    ineffective = overturn & ~plurality_correct & ~selected_correct
    trap = hard_mask & ~plurality_correct
    nontrap = hard_mask & plurality_correct
    flips = overturn & hard_mask
    return {
        "all_accuracy": group_accuracy(labels, selected),
        "hard_accuracy": group_accuracy(labels, selected, hard_mask),
        "hard_correct": int(selected_correct[hard_mask].sum()),
        "hard_n": int(hard_mask.sum()),
        "trap_accuracy": group_accuracy(labels, selected, trap) if trap.any() else None,
        "trap_correct": int(selected_correct[trap].sum()),
        "trap_n": int(trap.sum()),
        "nontrap_accuracy": group_accuracy(labels, selected, nontrap) if nontrap.any() else None,
        "nontrap_correct": int(selected_correct[nontrap].sum()),
        "nontrap_n": int(nontrap.sum()),
        "overturns": int(flips.sum()),
        "good_overturns": int((good & hard_mask).sum()),
        "bad_overturns": int((bad & hard_mask).sum()),
        "ineffective_overturns": int((ineffective & hard_mask).sum()),
        "net_corrected": int((good & hard_mask).sum() - (bad & hard_mask).sum()),
        "overturn_precision": (
            float((good & hard_mask).sum() / flips.sum()) if flips.any() else None
        ),
        "correct_overturn_recall": (
            float((good & hard_mask).sum() / trap.sum()) if trap.any() else None
        ),
    }


def local_folds_from_global(hard_indices: np.ndarray) -> list[np.ndarray]:
    payload = json.loads(Path("runs/folds.json").read_text(encoding="utf-8"))
    if payload.get("seed") != SEED:
        raise ValueError("Frozen fold seed mismatch")
    position = {int(index): pos for pos, index in enumerate(hard_indices)}
    folds = [
        np.asarray([position[int(index)] for index in fold], dtype=int)
        for fold in payload["fold_indices"]
    ]
    if set(np.concatenate(folds).tolist()) != set(range(len(hard_indices))):
        raise ValueError("Frozen folds do not cover train-hard exactly")
    return folds


def run_group_config(
    key: str,
    config: dict,
    features: np.ndarray,
    labels: np.ndarray,
    valid: np.ndarray,
    scalar_dim: int,
    folds: list[np.ndarray],
    train_bank: list[dict],
    train_group_data: dict,
    train_candidate_scores: np.ndarray,
    val_features: np.ndarray,
    val_valid: np.ndarray,
    val_bank: list[dict],
    val_group_data: dict,
    val_candidate_scores: np.ndarray,
    val_group_labels: np.ndarray,
    val_hard: np.ndarray,
) -> tuple[dict, dict, np.ndarray, np.ndarray]:
    output_dir = RUNS / "group_models" / key
    output_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    all_indices = set(range(len(labels)))
    oof_scores = np.full((len(labels), 4), -np.inf, dtype=np.float64)
    fold_results = []
    histories = []
    for fold_no, eval_indices in enumerate(folds):
        train_indices = np.asarray(sorted(all_indices - set(eval_indices.tolist())), dtype=int)
        model = train_group_model(
            features, labels, valid, train_indices, scalar_dim,
            scalar_l2=config["scalar_l2"],
            embedding_l2=config["embedding_l2"],
            pairwise_weight=config["pairwise_weight"],
            max_steps=110,
        )
        scores = score_group_model(model, features[eval_indices], valid[eval_indices])
        oof_scores[eval_indices] = scores
        _, candidate_choices = group_argmax_choices(
            [train_bank[int(i)] for i in eval_indices],
            scores,
            {
                "valid": train_group_data["valid"][eval_indices],
                "memberships": [train_group_data["memberships"][int(i)] for i in eval_indices],
            },
            train_candidate_scores[eval_indices],
        )
        group_choices = candidate_choices_to_groups(candidate_choices, {
            "memberships": [train_group_data["memberships"][int(i)] for i in eval_indices]
        })
        accuracy = group_accuracy(labels[eval_indices], group_choices)
        fold_results.append({"fold": fold_no, "n": len(eval_indices), "accuracy": accuracy})
        histories.append({"fold": fold_no, "history": model["loss_history"]})
    oof_groups, oof_candidate_choices = group_argmax_choices(
        train_bank, oof_scores, train_group_data, train_candidate_scores
    )
    fold_accuracy = [row["accuracy"] for row in fold_results]
    cv_mean, cv_se = fold_stats(fold_accuracy)
    full_indices = np.arange(len(labels), dtype=int)
    full_model = train_group_model(
        features, labels, valid, full_indices, scalar_dim,
        scalar_l2=config["scalar_l2"],
        embedding_l2=config["embedding_l2"],
        pairwise_weight=config["pairwise_weight"],
        max_steps=160,
    )
    val_scores = score_group_model(full_model, val_features, val_valid)
    val_groups, val_candidate_choices = group_argmax_choices(
        val_bank, val_scores, val_group_data, val_candidate_scores
    )
    val_plurality = plurality_group_indices(val_bank, val_group_data)
    metrics = decision_metrics(val_group_labels, val_groups, val_plurality, val_hard)
    train_plurality = plurality_group_indices(train_bank, train_group_data)
    train_metrics = decision_metrics(labels, oof_groups, train_plurality)
    result = {
        "key": key,
        **config,
        "cv_fold_accuracy": fold_accuracy,
        "cv_mean": cv_mean,
        "cv_se": cv_se,
        "train_oof_metrics": train_metrics,
        "validation_metrics": metrics,
        "elapsed_seconds": time.perf_counter() - started,
    }
    dump_json(output_dir / "result.json", result)
    dump_json(output_dir / "fold_results.json", fold_results)
    dump_json(output_dir / "loss_histories.json", histories)
    np.savez(
        output_dir / "predictions.npz",
        oof_scores=oof_scores,
        oof_groups=oof_groups,
        oof_candidate_choices=np.asarray(oof_candidate_choices),
        validation_scores=val_scores,
        validation_groups=val_groups,
        validation_candidate_choices=np.asarray(val_candidate_choices),
    )
    return result, full_model, oof_scores, val_scores


def binary_nll(labels: np.ndarray, probabilities: np.ndarray) -> float:
    p = np.clip(probabilities, 1e-8, 1 - 1e-8)
    y = np.asarray(labels, dtype=np.float64)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def cross_fitted_logistic(
    features: np.ndarray,
    labels: np.ndarray,
    folds: list[np.ndarray],
    l2: float,
    eligible: np.ndarray | None = None,
    class_balance: bool = False,
) -> tuple[np.ndarray, list[dict]]:
    if eligible is None:
        eligible = np.ones(len(labels), dtype=bool)
    all_indices = np.arange(len(labels))
    probabilities = np.zeros(len(labels), dtype=np.float64)
    models = []
    for eval_indices in folds:
        train_mask = eligible & ~np.isin(all_indices, eval_indices)
        model = fit_logistic(features, labels, train_mask, l2, class_balance=class_balance)
        probabilities[eval_indices] = logistic_probabilities(model, features[eval_indices])
        models.append(model)
    return probabilities, models


def main() -> None:
    RUNS.mkdir(parents=True, exist_ok=True)
    started_unix = time.time(); started_perf = time.perf_counter()
    dump_json(RUNS / "search_state.json", {
        "status": "running", "seed": SEED, "started_unix": started_unix,
        "test_touched": False, "historical_test_score_known": True,
        "selection_sources": ["train grouped CV", "validation development"],
    })

    train_bank_all, train_arrays = load_inputs("train", DATA)
    train_refs = read_jsonl(DATA / "gsm8k_train.jsonl")
    train_blocks_all = extract_feature_blocks(train_bank_all, train_arrays["embeddings"])
    train_candidate_labels_all = labels_from_references(train_bank_all, train_refs)
    hard_indices = np.flatnonzero(hard_group_mask(train_blocks_all["answers"]))
    train_bank = [train_bank_all[int(index)] for index in hard_indices]
    train_embeddings = train_arrays["embeddings"][hard_indices]
    train_blocks = slice_blocks(train_blocks_all, hard_indices)
    train_candidate_labels = train_candidate_labels_all[hard_indices]
    folds = local_folds_from_global(hard_indices)

    with np.load(STAGE1_OOF, allow_pickle=False) as stage1_data, np.load(STAGE2_OOF, allow_pickle=False) as stage2_data:
        if not np.array_equal(stage1_data["indices"].astype(int), hard_indices):
            raise ValueError("Stage-1 OOF index mismatch")
        if not np.array_equal(stage2_data["indices"].astype(int), hard_indices):
            raise ValueError("Stage-2 OOF index mismatch")
        train_candidate_scores = np.stack([
            stage1_data["scores"].astype(np.float64),
            stage2_data["scores"].astype(np.float64),
        ], axis=2)
    train_group_data = extract_group_features(
        train_bank, train_embeddings, train_blocks, train_candidate_scores
    )
    train_group_labels = group_labels(train_candidate_labels, train_group_data["memberships"])

    val_bank, val_arrays = load_inputs("validation", DATA)
    val_refs = read_jsonl(DATA / "gsm8k_validation.jsonl")
    val_blocks = extract_feature_blocks(val_bank, val_arrays["embeddings"])
    val_candidate_labels = labels_from_references(val_bank, val_refs)
    val_hard = hard_group_mask(val_blocks["answers"])
    stage1_model = load_model(STAGE1_CHECKPOINT)
    stage2_model = load_model(STAGE2_CHECKPOINT)
    val_candidate_scores = np.stack([
        score_model(stage1_model, val_blocks),
        score_model(stage2_model, val_blocks),
    ], axis=2)
    val_group_data = extract_group_features(
        val_bank, val_arrays["embeddings"], val_blocks, val_candidate_scores
    )
    val_group_labels = group_labels(val_candidate_labels, val_group_data["memberships"])

    train_plurality = plurality_group_indices(train_bank, train_group_data)
    val_plurality = plurality_group_indices(val_bank, val_group_data)
    baseline_results = []
    baseline_results.append({
        "key": "plurality",
        "train_oof_metrics": decision_metrics(train_group_labels, train_plurality, train_plurality),
        "validation_metrics": decision_metrics(val_group_labels, val_plurality, val_plurality, val_hard),
    })
    for model_index, name in ((0, "stage1_candidate"), (1, "stage2_candidate")):
        train_candidate_choices = masked_candidate_argmax(
            train_candidate_scores[:, :, model_index], train_blocks["valid"]
        )
        val_candidate_choices = masked_candidate_argmax(
            val_candidate_scores[:, :, model_index], val_blocks["valid"]
        )
        train_groups = candidate_choices_to_groups(train_candidate_choices, train_group_data)
        val_groups = candidate_choices_to_groups(val_candidate_choices, val_group_data)
        baseline_results.append({
            "key": name,
            "train_oof_metrics": decision_metrics(train_group_labels, train_groups, train_plurality),
            "validation_metrics": decision_metrics(val_group_labels, val_groups, val_plurality, val_hard),
        })

    pooling_results = []
    for alpha in (0.0, 0.2, 0.4, 0.6, 0.8, 1.0):
        train_choices = shrinkage_pool_choices(
            train_bank, train_candidate_scores[:, :, 1], train_group_data, alpha
        )
        val_choices = shrinkage_pool_choices(
            val_bank, val_candidate_scores[:, :, 1], val_group_data, alpha
        )
        train_groups = candidate_choices_to_groups(train_choices, train_group_data)
        val_groups = candidate_choices_to_groups(val_choices, val_group_data)
        pooling_results.append({
            "key": f"shrinkage_pool_a{alpha:g}", "alpha": alpha,
            "train_oof_metrics": decision_metrics(train_group_labels, train_groups, train_plurality),
            "validation_metrics": decision_metrics(val_group_labels, val_groups, val_plurality, val_hard),
        })

    group_results = []
    group_artifacts = {}
    for scalar_l2 in (0.01, 0.1):
        for embedding_l2 in (0.1, 1.0):
            for pairwise_weight in (0.0, 0.1):
                key = f"group_s{scalar_l2:g}_e{embedding_l2:g}_p{pairwise_weight:g}"
                config = {
                    "scalar_l2": scalar_l2,
                    "embedding_l2": embedding_l2,
                    "pairwise_weight": pairwise_weight,
                }
                result, full_model, oof_scores, val_scores = run_group_config(
                    key, config,
                    train_group_data["features"], train_group_labels, train_group_data["valid"],
                    train_group_data["scalar"].shape[2], folds, train_bank, train_group_data,
                    train_candidate_scores, val_group_data["features"], val_group_data["valid"],
                    val_bank, val_group_data, val_candidate_scores, val_group_labels, val_hard,
                )
                group_results.append(result)
                group_artifacts[key] = (full_model, oof_scores, val_scores)

    selected_group_result = max(
        group_results,
        key=lambda row: (
            row["validation_metrics"]["hard_correct"],
            row["validation_metrics"]["trap_correct"],
            row["cv_mean"],
            -row["scalar_l2"] - row["embedding_l2"] - row["pairwise_weight"],
        ),
    )
    group_model, group_oof_scores, group_val_scores = group_artifacts[selected_group_result["key"]]

    train_gate_base = gate_feature_matrices(
        train_bank, train_group_data, group_oof_scores, train_group_labels
    )
    val_gate_base = gate_feature_matrices(
        val_bank, val_group_data, group_val_scores, val_group_labels
    )
    reliability_results = []
    reliability_artifacts = {}
    for reliability_l2 in (0.01, 0.1, 1.0):
        q_oof, _ = cross_fitted_logistic(
            train_gate_base["features"],
            train_gate_base["plurality_correct"].astype(float),
            folds, reliability_l2, class_balance=False,
        )
        reliability_model = fit_logistic(
            train_gate_base["features"],
            train_gate_base["plurality_correct"].astype(float),
            np.ones(len(train_bank), dtype=bool),
            reliability_l2,
            class_balance=False,
        )
        q_val = logistic_probabilities(reliability_model, val_gate_base["features"])
        result = {
            "l2": reliability_l2,
            "train_oof_nll": binary_nll(train_gate_base["plurality_correct"], q_oof),
            "validation_nll": binary_nll(
                val_gate_base["plurality_correct"][val_hard], q_val[val_hard]
            ),
            "validation_nll_scope": "validation-hard",
        }
        reliability_results.append(result)
        reliability_artifacts[reliability_l2] = (q_oof, q_val, reliability_model)
    selected_reliability = min(reliability_results, key=lambda row: row["validation_nll"])
    q_oof, q_val, reliability_model = reliability_artifacts[selected_reliability["l2"]]
    q_oof_logit = np.log(np.clip(q_oof, 1e-8, 1 - 1e-8) / np.clip(1 - q_oof, 1e-8, 1))
    q_val_logit = np.log(np.clip(q_val, 1e-8, 1 - 1e-8) / np.clip(1 - q_val, 1e-8, 1))
    train_gate_features = np.column_stack([train_gate_base["features"], q_oof, q_oof_logit])
    val_gate_features = np.column_stack([val_gate_base["features"], q_val, q_val_logit])
    gate_feature_names = train_gate_base["names"] + ["plurality_reliability_probability", "plurality_reliability_logit"]

    gate_results = []
    gate_artifacts = {}
    for gate_l2 in (0.01, 0.1, 1.0):
        for class_balance in (False, True):
            gate_oof, _ = cross_fitted_logistic(
                train_gate_features,
                train_gate_base["overturn_label"],
                folds, gate_l2,
                eligible=train_gate_base["informative"],
                class_balance=class_balance,
            )
            gate_model = fit_logistic(
                train_gate_features,
                train_gate_base["overturn_label"],
                train_gate_base["informative"],
                gate_l2,
                class_balance=class_balance,
            )
            gate_val = logistic_probabilities(gate_model, val_gate_features)
            for threshold in (0.3, 0.4, 0.5, 0.6, 0.7, 0.8):
                train_selected = gate_group_choices(
                    train_gate_base["plurality"], train_gate_base["challenger"],
                    gate_oof, threshold,
                )
                val_selected = gate_group_choices(
                    val_gate_base["plurality"], val_gate_base["challenger"],
                    gate_val, threshold,
                )
                key = f"gate_l2{gate_l2:g}_b{int(class_balance)}_t{threshold:g}"
                result = {
                    "key": key,
                    "gate_l2": gate_l2,
                    "class_balance": class_balance,
                    "threshold": threshold,
                    "reliability_l2": selected_reliability["l2"],
                    "train_oof_metrics": decision_metrics(
                        train_group_labels, train_selected, train_gate_base["plurality"]
                    ),
                    "validation_metrics": decision_metrics(
                        val_group_labels, val_selected, val_gate_base["plurality"], val_hard
                    ),
                }
                gate_results.append(result)
                gate_artifacts[key] = (gate_model, gate_oof, gate_val, train_selected, val_selected)

    # Fixed margin gates are an interpretable non-learned CREO baseline.
    margin_results = []
    train_margin = np.asarray([
        group_oof_scores[i, train_gate_base["challenger"][i]]
        - group_oof_scores[i, train_gate_base["plurality"][i]]
        for i in range(len(train_bank))
    ])
    val_margin = np.asarray([
        group_val_scores[i, val_gate_base["challenger"][i]]
        - group_val_scores[i, val_gate_base["plurality"][i]]
        for i in range(len(val_bank))
    ])
    for threshold in (-0.5, -0.25, 0.0, 0.25, 0.5, 1.0):
        train_selected = np.where(
            train_margin >= threshold, train_gate_base["challenger"], train_gate_base["plurality"]
        )
        val_selected = np.where(
            val_margin >= threshold, val_gate_base["challenger"], val_gate_base["plurality"]
        )
        margin_results.append({
            "key": f"fixed_margin_t{threshold:g}", "threshold": threshold,
            "train_oof_metrics": decision_metrics(
                train_group_labels, train_selected, train_gate_base["plurality"]
            ),
            "validation_metrics": decision_metrics(
                val_group_labels, val_selected, val_gate_base["plurality"], val_hard
            ),
        })

    stage2_baseline = next(row for row in baseline_results if row["key"] == "stage2_candidate")
    stage2_hard_correct = stage2_baseline["validation_metrics"]["hard_correct"]
    eligible_gates = []
    for row in gate_results:
        metrics = row["validation_metrics"]
        nontrap_drop = (
            stage2_baseline["validation_metrics"]["nontrap_accuracy"] - metrics["nontrap_accuracy"]
        )
        if (
            metrics["hard_correct"] >= stage2_hard_correct + 2
            and metrics["good_overturns"] >= metrics["bad_overturns"]
            and nontrap_drop <= 1.0
        ):
            eligible_gates.append(row)
    if eligible_gates:
        selected_gate_result = max(
            eligible_gates,
            key=lambda row: (
                row["validation_metrics"]["hard_correct"],
                row["validation_metrics"]["trap_correct"],
                row["train_oof_metrics"]["hard_accuracy"],
                -row["gate_l2"], -row["threshold"],
            ),
        )
        selection_status = "creo_gate_selected"
    else:
        selected_gate_result = max(
            gate_results,
            key=lambda row: (
                row["validation_metrics"]["hard_correct"],
                row["validation_metrics"]["trap_correct"],
                row["train_oof_metrics"]["hard_accuracy"],
            ),
        )
        selection_status = "no_gate_met_predeclared_replacement_rule"
    gate_model, gate_oof, gate_val, train_selected, val_selected = gate_artifacts[selected_gate_result["key"]]

    bundle_dir = RUNS / "provisional_creo_bundle"
    bundle_config = {
        "seed": SEED,
        "threshold": selected_gate_result["threshold"],
        "gate_feature_names": gate_feature_names,
        "group_scalar_names": train_group_data["scalar_names"],
        "selected_group_config": {
            key: selected_group_result[key]
            for key in ("key", "scalar_l2", "embedding_l2", "pairwise_weight")
        },
        "selection_status": selection_status,
        "base_checkpoints": ["base_stage1", "base_stage2"],
    }
    save_creo_bundle(bundle_dir, group_model, reliability_model, gate_model, bundle_config)
    for source, name in ((STAGE1_CHECKPOINT, "base_stage1"), (STAGE2_CHECKPOINT, "base_stage2")):
        destination = bundle_dir / name
        if not destination.exists():
            shutil.copytree(source, destination)

    summary = {
        "seed": SEED,
        "test_touched": False,
        "historical_test_score_not_used_for_selection": True,
        "baseline_results": baseline_results,
        "pooling_results": pooling_results,
        "group_results": group_results,
        "selected_group_result": selected_group_result,
        "reliability_results": reliability_results,
        "selected_reliability": selected_reliability,
        "margin_results": margin_results,
        "gate_results": gate_results,
        "selected_gate_result": selected_gate_result,
        "selection_status": selection_status,
    }
    dump_json(RUNS / "experiment_summary.json", summary)
    np.savez(
        RUNS / "selected_predictions.npz",
        train_group_scores=group_oof_scores,
        validation_group_scores=group_val_scores,
        train_reliability=q_oof,
        validation_reliability=q_val,
        train_gate_probability=gate_oof,
        validation_gate_probability=gate_val,
        train_selected_groups=train_selected,
        validation_selected_groups=val_selected,
    )
    asset_bytes = sum(path.stat().st_size for path in bundle_dir.rglob("*") if path.is_file())
    ended = time.time()
    dump_json(RUNS / "search_state.json", {
        "status": "paused_before_stage3_test",
        "seed": SEED,
        "started_unix": started_unix,
        "ended_unix": ended,
        "elapsed_seconds": time.perf_counter() - started_perf,
        "test_touched": False,
        "selection_status": selection_status,
        "provisional_bundle": str(bundle_dir),
        "bundle_bytes": asset_bytes,
    })
    print(json.dumps({
        "selection_status": selection_status,
        "stage2_validation": stage2_baseline["validation_metrics"],
        "selected_group": selected_group_result,
        "selected_gate": selected_gate_result,
        "bundle": str(bundle_dir),
        "bundle_bytes": asset_bytes,
        "elapsed_seconds": time.perf_counter() - started_perf,
    }, indent=2))


if __name__ == "__main__":
    main()
