"""Finite, leakage-aware Stage 4 CREO-3W search using train and validation only."""
from __future__ import annotations

import json
import math
from pathlib import Path
import time

import numpy as np

from common import DATA, load_inputs, read_jsonl
from selector_features import extract_feature_blocks, hard_group_mask, labels_from_references
from stage4_features import (
    build_answer_groups,
    candidate_choices_to_groups,
    centered_embeddings,
    consensus_features,
    extract_group_features,
    group_argmax,
    group_choices_to_candidates,
    group_labels,
    masked_candidate_choices,
    merge_branch_probabilities,
    plurality_groups,
)
from stage4_model import (
    binary_probabilities,
    calibration_metrics,
    fit_binary_logistic,
    fit_multinomial_gate,
    gate_probabilities,
    save_bundle,
    score_candidate_model,
    score_group_model,
    train_candidate_model,
    train_group_model,
)


SEED = 20261022
RUNS = Path("runs/stage4")
STARTED = time.perf_counter()
SEARCH_LIMIT_SECONDS = 90 * 60
STAGE2_OOF = Path("runs/stage2/cv/hand_centered__uniform__pair0.1/oof_predictions.npz")
STAGE2_FOLD_CORRECT = [350, 335, 354, 353, 334]
STABLE_FOLD_CORRECT = [351, 340, 359, 354, 332]


def dump_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def elapsed() -> float:
    return time.perf_counter() - STARTED


def print_progress(event: str, payload: dict) -> None:
    print(json.dumps({"event": event, "elapsed_seconds": elapsed(), **payload}), flush=True)


def extended_folds(hard_indices: np.ndarray, easy_indices: np.ndarray) -> list[np.ndarray]:
    payload = json.loads(Path("runs/folds.json").read_text(encoding="utf-8"))
    if payload.get("seed") != SEED:
        raise ValueError("Frozen fold seed mismatch")
    rng = np.random.default_rng(SEED)
    shuffled_easy = rng.permutation(easy_indices)
    easy_parts = np.array_split(shuffled_easy, 5)
    folds = []
    for hard_fold, easy_fold in zip(payload["fold_indices"], easy_parts):
        folds.append(np.asarray(list(map(int, hard_fold)) + easy_fold.astype(int).tolist(), dtype=int))
    concatenated_hard = np.concatenate([
        np.asarray([index for index in fold if index in set(hard_indices.tolist())], dtype=int)
        for fold in folds
    ])
    if set(concatenated_hard.tolist()) != set(hard_indices.tolist()):
        raise ValueError("Extended folds do not preserve frozen hard folds")
    if set(np.concatenate(folds).tolist()) != set(range(len(hard_indices) + len(easy_indices))):
        raise ValueError("Extended folds do not cover train")
    return folds


def fold_stats(values: list[float]) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    return float(values.mean()), float(values.std(ddof=1) / math.sqrt(len(values)))


def method_metrics(
    name: str,
    selected_groups: np.ndarray,
    group_label_array: np.ndarray,
    plurality: np.ndarray,
    hard_mask: np.ndarray,
    folds: list[np.ndarray],
    candidate_labels: np.ndarray,
) -> dict:
    correct = group_label_array[np.arange(len(selected_groups)), selected_groups]
    plurality_correct = group_label_array[np.arange(len(selected_groups)), plurality]
    fold_rows = []
    hard_fold_correct = []
    for fold_no, fold in enumerate(folds):
        fold_hard = fold[hard_mask[fold]]
        count = int(correct[fold_hard].sum())
        hard_fold_correct.append(count)
        fold_rows.append({
            "fold": fold_no, "n_hard": len(fold_hard), "correct": count,
            "accuracy": 100.0 * count / len(fold_hard),
            "delta_correct_vs_stage2": count - STAGE2_FOLD_CORRECT[fold_no],
            "delta_correct_vs_stable_direct": count - STABLE_FOLD_CORRECT[fold_no],
        })
    mean, se = fold_stats([row["accuracy"] for row in fold_rows])
    hard = hard_mask
    trap = hard & ~plurality_correct
    nontrap = hard & plurality_correct
    different = selected_groups != plurality
    good = different & ~plurality_correct & correct
    bad = different & plurality_correct & ~correct
    single = hard & (candidate_labels.sum(axis=1) == 1)
    return {
        "name": name,
        "cv_mean": mean,
        "cv_se": se,
        "hard_correct": int(correct[hard].sum()), "hard_n": int(hard.sum()),
        "all_correct": int(correct.sum()), "all_n": len(correct),
        "folds": fold_rows,
        "fold_correct": hard_fold_correct,
        "trap_correct": int(correct[trap].sum()), "trap_n": int(trap.sum()),
        "trap_accuracy": 100.0 * float(correct[trap].mean()),
        "single_correct": int(correct[single].sum()), "single_n": int(single.sum()),
        "single_accuracy": 100.0 * float(correct[single].mean()),
        "correct_plurality_correct": int(correct[nontrap].sum()), "correct_plurality_n": int(nontrap.sum()),
        "correct_plurality_accuracy": 100.0 * float(correct[nontrap].mean()),
        "good_overturns": int(good[hard].sum()), "bad_overturns": int(bad[hard].sum()),
        "net_corrected": int(good[hard].sum() - bad[hard].sum()),
        "overturns": int((different & hard).sum()),
    }


def candidate_cv(
    key: str,
    config: dict,
    bank: list[dict],
    centered: np.ndarray,
    blocks: dict,
    labels: np.ndarray,
    hard: np.ndarray,
    folds: list[np.ndarray],
    groups: dict,
    group_label_array: np.ndarray,
    plurality: np.ndarray,
) -> tuple[dict, dict]:
    started = time.perf_counter()
    scores = np.zeros((len(bank), 4), dtype=np.float64)
    choices = np.zeros(len(bank), dtype=np.int64)
    models = []
    histories = []
    all_indices = np.arange(len(bank))
    for fold_no, eval_indices in enumerate(folds):
        train_mask = np.ones(len(bank), dtype=bool); train_mask[eval_indices] = False
        model = train_candidate_model(
            centered, blocks["answers"], blocks["stage2_hand"], labels, blocks["valid"], hard,
            all_indices[train_mask], config, SEED + fold_no * 101,
        )
        fold_scores, _ = score_candidate_model(
            model, centered[eval_indices], [blocks["answers"][int(i)] for i in eval_indices],
            blocks["stage2_hand"][eval_indices],
        )
        scores[eval_indices] = fold_scores
        fold_bank = [bank[int(i)] for i in eval_indices]
        choices[eval_indices] = masked_candidate_choices(
            fold_bank, fold_scores, blocks["valid"][eval_indices]
        )
        models.append(model)
        histories.append({"fold": fold_no, "history": model["loss_history"]})
    selected_groups = candidate_choices_to_groups(choices, groups)
    result = method_metrics(
        key, selected_groups, group_label_array, plurality, hard, folds, labels
    )
    result.update({"config": config, "elapsed_seconds": time.perf_counter() - started})
    output = RUNS / "candidate" / key
    dump_json(output / "result.json", result)
    dump_json(output / "loss_histories.json", histories)
    np.savez(output / "oof_predictions.npz", scores=scores, choices=choices, groups=selected_groups)
    print_progress("candidate_complete", {"key": key, "cv_mean": result["cv_mean"],
                                            "fold_correct": result["fold_correct"]})
    return result, {"scores": scores, "choices": choices, "groups": selected_groups, "models": models}


def branch_choices(method: str, branch_groups: np.ndarray, reliabilities: np.ndarray | None,
                   gate_probs: np.ndarray | None) -> np.ndarray:
    if method == "plurality":
        return branch_groups[:, 0]
    if method == "candidate":
        return branch_groups[:, 1]
    if method == "group":
        return branch_groups[:, 2]
    if method == "fixed_reliability":
        return merge_branch_probabilities(branch_groups, reliabilities)
    if method.startswith("gate_"):
        return merge_branch_probabilities(branch_groups, gate_probs)
    raise ValueError(method)


def balanced_weights(hard: np.ndarray) -> np.ndarray:
    weights = np.zeros(len(hard), dtype=float)
    weights[hard] = 0.5 / max(int(hard.sum()), 1)
    weights[~hard] = 0.5 / max(int((~hard).sum()), 1)
    return weights


def crossfit_binary(features, labels, folds, hard, l2=0.1):
    probabilities = np.zeros(len(labels)); models = []
    weights = balanced_weights(hard); all_indices = np.arange(len(labels))
    for fold in folds:
        train_mask = ~np.isin(all_indices, fold)
        model = fit_binary_logistic(features, labels, train_mask, weights, l2=l2)
        probabilities[fold] = binary_probabilities(model, features[fold])
        models.append(model)
    full = fit_binary_logistic(features, labels, np.ones(len(labels), dtype=bool), weights, l2=l2)
    return probabilities, models, full


def crossfit_gate(features, positive, folds, l2):
    probabilities = np.zeros((len(features), 3)); all_indices = np.arange(len(features))
    for fold in folds:
        train_mask = ~np.isin(all_indices, fold)
        model = fit_multinomial_gate(features, positive, train_mask, l2=l2)
        probabilities[fold] = gate_probabilities(model, features[fold])
    full = fit_multinomial_gate(features, positive, np.ones(len(features), dtype=bool), l2=l2)
    return probabilities, full


def main() -> None:
    RUNS.mkdir(parents=True, exist_ok=True)
    dump_json(RUNS / "search_state.json", {
        "status": "running", "seed": SEED, "started_unix": time.time(),
        "test_touched": False, "search_budget_seconds": SEARCH_LIMIT_SECONDS,
    })
    bank, arrays = load_inputs("train", DATA)
    references = read_jsonl(DATA / "gsm8k_train.jsonl")
    blocks = extract_feature_blocks(bank, arrays["embeddings"])
    labels = labels_from_references(bank, references)
    hard = hard_group_mask(blocks["answers"])
    hard_indices = np.flatnonzero(hard); easy_indices = np.flatnonzero(~hard)
    folds = extended_folds(hard_indices, easy_indices)
    centered = centered_embeddings(arrays["embeddings"])
    groups = build_answer_groups(bank, blocks["answers"])
    group_label_array = group_labels(labels, groups)
    plurality = plurality_groups(bank, groups)
    dump_json(RUNS / "fold_assignment.json", {
        "seed": SEED, "fold_indices": [fold.tolist() for fold in folds],
        "hard_fold_indices": [fold[hard[fold]].tolist() for fold in folds],
    })

    with np.load(STAGE2_OOF, allow_pickle=False) as stage2:
        if not np.array_equal(stage2["indices"].astype(int), hard_indices):
            raise ValueError("Stage 2 OOF index mismatch")
        stage2_hard_choices = stage2["choices"].astype(int)
    stage2_choices = np.zeros(len(bank), dtype=int)
    stage2_choices[hard_indices] = stage2_hard_choices
    stage2_groups = candidate_choices_to_groups(stage2_choices, groups)
    stage2_metrics = method_metrics(
        "stage2", stage2_groups, group_label_array, plurality, hard, folds, labels
    )

    phase_a = [
        ("linear_d32", {"mode": "linear_lowrank", "projection_dim": 32, "hidden_dim": 16,
                         "pairwise_weight": 0.0, "bce_weight": 0.0, "epochs": 5, "batch_size": 256}),
        ("interaction_d32_h16", {"mode": "interaction", "projection_dim": 32, "hidden_dim": 16,
                                  "pairwise_weight": 0.0, "bce_weight": 0.0, "epochs": 5, "batch_size": 256}),
        ("interaction_pair", {"mode": "interaction", "projection_dim": 32, "hidden_dim": 16,
                               "pairwise_weight": 0.1, "bce_weight": 0.0, "epochs": 5, "batch_size": 256}),
        ("interaction_pair_bce005", {"mode": "interaction", "projection_dim": 32, "hidden_dim": 16,
                                      "pairwise_weight": 0.1, "bce_weight": 0.05, "epochs": 5, "batch_size": 256}),
        ("interaction_pair_bce01", {"mode": "interaction", "projection_dim": 32, "hidden_dim": 16,
                                     "pairwise_weight": 0.1, "bce_weight": 0.1, "epochs": 5, "batch_size": 256}),
    ]
    candidate_results = [stage2_metrics]
    candidate_artifacts = {}
    for key, config in phase_a:
        result, artifact = candidate_cv(
            key, config, bank, centered, blocks, labels, hard, folds,
            groups, group_label_array, plurality,
        )
        candidate_results.append(result); candidate_artifacts[key] = artifact

    trainable_ranked = sorted(
        [row for row in candidate_results if row["name"] != "stage2"],
        key=lambda row: row["cv_mean"], reverse=True,
    )
    top_phase_a = trainable_ranked[:2]
    capacity_configs = []
    for parent in top_phase_a:
        base = dict(parent["config"])
        for projection_dim, hidden_dim in ((32, 32), (48, 16), (48, 32)):
            config = {**base, "projection_dim": projection_dim, "hidden_dim": hidden_dim}
            key = f"{parent['name']}__d{projection_dim}_h{hidden_dim}"
            capacity_configs.append((key, config))
    seen = set()
    for key, config in capacity_configs:
        signature = tuple(sorted(config.items()))
        if signature in seen or elapsed() > SEARCH_LIMIT_SECONDS * 0.72:
            continue
        seen.add(signature)
        result, artifact = candidate_cv(
            key, config, bank, centered, blocks, labels, hard, folds,
            groups, group_label_array, plurality,
        )
        candidate_results.append(result); candidate_artifacts[key] = artifact

    best_candidate_result = max(
        [row for row in candidate_results if row["name"] != "stage2"],
        key=lambda row: row["cv_mean"],
    )
    candidate_key = best_candidate_result["name"]
    candidate_config = best_candidate_result["config"]
    candidate_artifact = candidate_artifacts[candidate_key]
    stage2_retention = stage2_metrics["correct_plurality_accuracy"]
    candidate_replacement_pass = (
        best_candidate_result["cv_mean"] >= stage2_metrics["cv_mean"] + 0.75
        and sum(row["delta_correct_vs_stage2"] >= 0 for row in best_candidate_result["folds"]) >= 4
        and best_candidate_result["good_overturns"] > best_candidate_result["bad_overturns"]
        and stage2_retention - best_candidate_result["correct_plurality_accuracy"] <= 1.0
    )

    # Strict nested candidate cross-fitting for group-model training features.
    group_fold_cache = []
    all_indices = np.arange(len(bank))
    for outer, eval_indices in enumerate(folds):
        train_hard = hard_indices[~np.isin(hard_indices, eval_indices)]
        inner_scores = np.zeros((len(bank), 4), dtype=float)
        for inner, inner_eval in enumerate(folds):
            if inner == outer:
                continue
            inner_eval_hard = inner_eval[hard[inner_eval]]
            inner_train_mask = ~np.isin(all_indices, np.concatenate([eval_indices, inner_eval]))
            inner_config = {**candidate_config, "epochs": min(3, int(candidate_config["epochs"]))}
            inner_model = train_candidate_model(
                centered, blocks["answers"], blocks["stage2_hand"], labels, blocks["valid"], hard,
                all_indices[inner_train_mask], inner_config, SEED + 5000 + outer * 101 + inner,
            )
            inner_scores[inner_eval_hard] = score_candidate_model(
                inner_model, centered[inner_eval_hard],
                [blocks["answers"][int(i)] for i in inner_eval_hard],
                blocks["stage2_hand"][inner_eval_hard],
            )[0]
        outer_model = candidate_artifact["models"][outer]
        _, outer_projected_all = score_candidate_model(
            outer_model, centered, blocks["answers"], blocks["stage2_hand"]
        )
        train_features = extract_group_features(
            [bank[int(i)] for i in train_hard],
            {
                "answers": [groups["answers"][int(i)] for i in train_hard],
                "candidate_answers": [groups["candidate_answers"][int(i)] for i in train_hard],
                "memberships": [groups["memberships"][int(i)] for i in train_hard],
                "candidate_to_group": groups["candidate_to_group"][train_hard],
                "valid": groups["valid"][train_hard], "support": groups["support"][train_hard],
            },
            outer_projected_all[train_hard], inner_scores[train_hard],
        )
        eval_features = extract_group_features(
            [bank[int(i)] for i in eval_indices],
            {
                "answers": [groups["answers"][int(i)] for i in eval_indices],
                "candidate_answers": [groups["candidate_answers"][int(i)] for i in eval_indices],
                "memberships": [groups["memberships"][int(i)] for i in eval_indices],
                "candidate_to_group": groups["candidate_to_group"][eval_indices],
                "valid": groups["valid"][eval_indices], "support": groups["support"][eval_indices],
            },
            outer_projected_all[eval_indices], candidate_artifact["scores"][eval_indices],
        )
        group_fold_cache.append({"train_indices": train_hard, "eval_indices": eval_indices,
                                 "train": train_features, "eval": eval_features})
        print_progress("nested_candidate_complete", {"outer_fold": outer})

    group_configs = [
        ("group_scalar", {"mode": "scalar_only", "hidden_dim": 16, "pairwise_weight": 0.0}),
        ("group_projected", {"mode": "projected_only", "hidden_dim": 16, "pairwise_weight": 0.0}),
        ("group_combined", {"mode": "combined", "hidden_dim": 16, "pairwise_weight": 0.0}),
        ("group_combined_pair", {"mode": "combined", "hidden_dim": 16, "pairwise_weight": 0.1}),
    ]
    group_results = [method_metrics(
        "group_support_plurality", plurality, group_label_array, plurality, hard, folds, labels
    )]
    group_artifacts = {}
    for key, config in group_configs:
        oof_scores = np.full((len(bank), 4), -np.inf, dtype=float)
        histories = []
        for fold_no, cache in enumerate(group_fold_cache):
            train_indices = cache["train_indices"]; train_data = cache["train"]
            model = train_group_model(
                train_data["features"], group_label_array[train_indices], train_data["valid"],
                np.arange(len(train_indices)), train_data["scalar_dim"], config, SEED + 7000 + fold_no,
            )
            eval_indices = cache["eval_indices"]; eval_data = cache["eval"]
            oof_scores[eval_indices] = score_group_model(model, eval_data["features"], eval_data["valid"])
            histories.append({"fold": fold_no, "history": model["loss_history"]})
        choices = group_argmax(oof_scores, groups["valid"])
        result = method_metrics(key, choices, group_label_array, plurality, hard, folds, labels)
        result["config"] = config
        group_results.append(result); group_artifacts[key] = {"scores": oof_scores, "groups": choices}
        output = RUNS / "group" / key
        dump_json(output / "result.json", result); dump_json(output / "loss_histories.json", histories)
        np.savez(output / "oof_predictions.npz", scores=oof_scores, groups=choices)
        print_progress("group_complete", {"key": key, "cv_mean": result["cv_mean"],
                                          "fold_correct": result["fold_correct"]})

    best_group_result = max(group_results, key=lambda row: row["cv_mean"])
    if best_group_result["name"] == "group_support_plurality":
        group_key = max(group_artifacts, key=lambda key: next(r for r in group_results if r["name"] == key)["cv_mean"])
    else:
        group_key = best_group_result["name"]
    group_config = next(row["config"] for row in group_results if row["name"] == group_key)
    group_artifact = group_artifacts[group_key]

    # Assemble fold-specific group features for calibration/gating.
    oof_group_features = None
    for cache in group_fold_cache:
        eval_indices = cache["eval_indices"]; data = cache["eval"]
        if oof_group_features is None:
            oof_group_features = {
                "scalar": np.zeros((len(bank), 4, data["scalar"].shape[2])),
                "projected": np.zeros((len(bank), 4, data["projected"].shape[2])),
                "features": np.zeros((len(bank), 4, data["features"].shape[2])),
                "scalar_dim": data["scalar_dim"], "scalar_names": data["scalar_names"],
                "valid": groups["valid"], "support": groups["support"],
                "memberships": groups["memberships"],
            }
        for name in ("scalar", "projected", "features"):
            oof_group_features[name][eval_indices] = data[name]

    candidate_groups = candidate_artifact["groups"]
    learned_groups = group_artifact["groups"]
    branch_groups = np.column_stack([plurality, candidate_groups, learned_groups])
    base_features, consensus_names = consensus_features(
        bank, groups, oof_group_features, plurality, candidate_groups, learned_groups,
        candidate_artifact["scores"], group_artifact["scores"], hard,
    )
    branch_correct = np.column_stack([
        group_label_array[np.arange(len(bank)), branch_groups[:, branch]] for branch in range(3)
    ])
    reliability_oof = np.zeros((len(bank), 3)); reliability_models = []
    for branch in range(3):
        reliability_oof[:, branch], _, full = crossfit_binary(
            base_features, branch_correct[:, branch], folds, hard, l2=0.1
        )
        reliability_models.append(full)
    gate_features = np.column_stack([base_features, reliability_oof])

    method_results = []
    method_groups = {}
    for name, selected in (("plurality", plurality), ("candidate", candidate_groups), ("group", learned_groups)):
        row = method_metrics(name, selected, group_label_array, plurality, hard, folds, labels)
        method_results.append(row); method_groups[name] = selected
    fixed_groups = merge_branch_probabilities(branch_groups, reliability_oof)
    fixed_result = method_metrics(
        "fixed_reliability", fixed_groups, group_label_array, plurality, hard, folds, labels
    )
    method_results.append(fixed_result); method_groups["fixed_reliability"] = fixed_groups
    gate_models = {}
    gate_oof_probabilities = {}
    for l2 in (0.01, 0.1, 1.0):
        probabilities, full_model = crossfit_gate(gate_features, branch_correct, folds, l2)
        name = f"gate_l2_{l2:g}"
        selected = merge_branch_probabilities(branch_groups, probabilities)
        row = method_metrics(name, selected, group_label_array, plurality, hard, folds, labels)
        row["l2"] = l2
        method_results.append(row); method_groups[name] = selected
        gate_models[name] = full_model; gate_oof_probabilities[name] = probabilities
        print_progress("gate_complete", {"key": name, "cv_mean": row["cv_mean"],
                                         "fold_correct": row["fold_correct"]})

    simple_best = max([row for row in method_results if row["name"] in {"plurality", "candidate", "group"}],
                      key=lambda row: row["cv_mean"])
    # A research leader is saved for reproducibility, but formal replacement
    # eligibility is judged against the stronger frozen Stage-2 baseline.
    research_leader = max(method_results, key=lambda row: row["cv_mean"])
    eligible_replacements = []
    for row in method_results:
        if (
            row["cv_mean"] >= stage2_metrics["cv_mean"] + 0.75
            and sum(fold["correct"] >= STAGE2_FOLD_CORRECT[i] for i, fold in enumerate(row["folds"])) >= 4
            and row["good_overturns"] > row["bad_overturns"]
            and stage2_metrics["correct_plurality_accuracy"] - row["correct_plurality_accuracy"] <= 1.0
        ):
            eligible_replacements.append(row)
    selected_result = research_leader
    selected_method = selected_result["name"]
    selected_groups = method_groups[selected_method]
    selected_correct = group_label_array[np.arange(len(bank)), selected_groups]
    selected_prob_oof, _, selected_calibrator = crossfit_binary(
        gate_features, selected_correct, folds, hard, l2=0.1
    )
    calibration = {
        "all": calibration_metrics(selected_correct, selected_prob_oof),
        "hard": calibration_metrics(selected_correct[hard], selected_prob_oof[hard]),
        "branch": {
            name: calibration_metrics(branch_correct[:, i], reliability_oof[:, i])
            for i, name in enumerate(("plurality", "candidate", "group"))
        },
        "warning": "Calibrated model scores are not guaranteed correctness probabilities.",
    }
    dump_json(RUNS / "calibration.json", calibration)

    # Fit final candidate and group models on complete train; no validation labels enter fitting.
    full_candidate_config = {**candidate_config, "epochs": max(8, int(candidate_config["epochs"]))}
    full_candidate = train_candidate_model(
        centered, blocks["answers"], blocks["stage2_hand"], labels, blocks["valid"], hard,
        np.arange(len(bank)), full_candidate_config, SEED + 9000,
    )
    full_candidate_scores, full_projected = score_candidate_model(
        full_candidate, centered, blocks["answers"], blocks["stage2_hand"]
    )
    full_group_train_features = extract_group_features(
        bank, groups, full_projected, candidate_artifact["scores"]
    )
    full_group = train_group_model(
        full_group_train_features["features"], group_label_array, full_group_train_features["valid"],
        hard_indices, full_group_train_features["scalar_dim"], group_config, SEED + 9100,
    )
    gate_model = gate_models.get(selected_method, gate_models[max(gate_models, key=lambda k: next(
        row for row in method_results if row["name"] == k
    )["cv_mean"])])
    bundle = {
        "candidate_model": full_candidate,
        "group_model": full_group,
        "plurality_calibrator": reliability_models[0],
        "candidate_calibrator": reliability_models[1],
        "group_calibrator": reliability_models[2],
        "selected_calibrator": selected_calibrator,
        "gate_model": gate_model,
    }
    bundle_config = {
        "seed": SEED,
        "selected_method": selected_method,
        "candidate_config": candidate_config,
        "group_config": group_config,
        "consensus_feature_names": consensus_names,
        "gate_feature_names": consensus_names + ["plurality_reliability", "candidate_reliability", "group_reliability"],
        "reliability_l2": 0.1,
        "test_touched": False,
    }
    save_bundle(RUNS / "provisional_bundle", bundle, bundle_config)

    # Validation consistency check: OOF top three only, each evaluated once in this single pass.
    top_three = sorted(method_results, key=lambda row: row["cv_mean"], reverse=True)[:3]
    val_bank, val_arrays = load_inputs("validation", DATA)
    val_refs = read_jsonl(DATA / "gsm8k_validation.jsonl")
    val_blocks = extract_feature_blocks(val_bank, val_arrays["embeddings"])
    val_labels = labels_from_references(val_bank, val_refs)
    val_hard = hard_group_mask(val_blocks["answers"])
    val_groups = build_answer_groups(val_bank, val_blocks["answers"])
    val_group_labels = group_labels(val_labels, val_groups)
    val_plurality = plurality_groups(val_bank, val_groups)
    val_candidate_scores, val_projected = score_candidate_model(
        full_candidate, centered_embeddings(val_arrays["embeddings"]),
        val_blocks["answers"], val_blocks["stage2_hand"],
    )
    val_candidate_choices = masked_candidate_choices(val_bank, val_candidate_scores, val_blocks["valid"])
    val_candidate_groups = candidate_choices_to_groups(val_candidate_choices, val_groups)
    val_group_features = extract_group_features(val_bank, val_groups, val_projected, val_candidate_scores)
    val_group_scores = score_group_model(full_group, val_group_features["features"], val_group_features["valid"])
    val_learned_groups = group_argmax(val_group_scores, val_group_features["valid"])
    val_branch_groups = np.column_stack([val_plurality, val_candidate_groups, val_learned_groups])
    val_base, _ = consensus_features(
        val_bank, val_groups, val_group_features, val_plurality, val_candidate_groups,
        val_learned_groups, val_candidate_scores, val_group_scores, val_hard,
    )
    val_reliability = np.column_stack([
        binary_probabilities(reliability_models[branch], val_base) for branch in range(3)
    ])
    val_gate_features = np.column_stack([val_base, val_reliability])
    validation_results = []
    validation_selected = {}
    for row in top_three:
        name = row["name"]
        if name.startswith("gate_"):
            probabilities = gate_probabilities(gate_models[name], val_gate_features)
            choices = merge_branch_probabilities(val_branch_groups, probabilities)
        elif name == "fixed_reliability":
            choices = merge_branch_probabilities(val_branch_groups, val_reliability)
        elif name == "plurality":
            choices = val_plurality
        elif name == "candidate":
            choices = val_candidate_groups
        else:
            choices = val_learned_groups
        correct = val_group_labels[np.arange(len(val_bank)), choices]
        p_correct = val_group_labels[np.arange(len(val_bank)), val_plurality]
        trap = val_hard & ~p_correct; nontrap = val_hard & p_correct
        single = val_hard & (val_labels.sum(axis=1) == 1)
        validation_results.append({
            "name": name, "all_correct": int(correct.sum()), "all_n": len(correct),
            "all_accuracy": 100.0 * float(correct.mean()),
            "hard_correct": int(correct[val_hard].sum()), "hard_n": int(val_hard.sum()),
            "hard_accuracy": 100.0 * float(correct[val_hard].mean()),
            "trap_correct": int(correct[trap].sum()), "trap_n": int(trap.sum()),
            "single_correct": int(correct[single].sum()), "single_n": int(single.sum()),
            "correct_consensus_correct": int(correct[nontrap].sum()), "correct_consensus_n": int(nontrap.sum()),
        })
        validation_selected[name] = choices

    fold_results = {
        "stage2": stage2_metrics["folds"],
        "stable_direct_fold_correct": STABLE_FOLD_CORRECT,
        "selected": selected_result["folds"],
    }
    paired_audit = {
        "selected_vs_stage2_fold_deltas": [
            selected_result["folds"][i]["correct"] - STAGE2_FOLD_CORRECT[i] for i in range(5)
        ],
        "selected_vs_stable_direct_fold_deltas": [
            selected_result["folds"][i]["correct"] - STABLE_FOLD_CORRECT[i] for i in range(5)
        ],
        "selected_metrics": selected_result,
        "test_touched": False,
    }
    summary = {
        "seed": SEED, "elapsed_seconds": elapsed(), "test_touched": False,
        "validation_reuse_warning": "Validation was observed in prior stages; selection is OOF-primary.",
        "stage2_baseline": stage2_metrics,
        "candidate_results": candidate_results,
        "best_candidate": best_candidate_result,
        "candidate_replacement_pass": candidate_replacement_pass,
        "group_results": group_results,
        "best_group": best_group_result,
        "method_results": method_results,
        "simple_stage4_leader": simple_best,
        "eligible_replacement_methods": [row["name"] for row in eligible_replacements],
        "replacement_pass": bool(eligible_replacements),
        "formal_replacement_decision": (
            max(eligible_replacements, key=lambda row: row["cv_mean"])["name"]
            if eligible_replacements else "retain_frozen_pre_stage4_model"
        ),
        "selected_method": selected_result,
        "validation_top_three": validation_results,
        "top_three_names": [row["name"] for row in top_three],
    }
    dump_json(RUNS / "experiment_summary.json", summary)
    dump_json(RUNS / "fold_results.json", fold_results)
    dump_json(RUNS / "paired_audit.json", paired_audit)
    np.savez(
        RUNS / "oof_predictions.npz",
        candidate_scores=candidate_artifact["scores"],
        candidate_groups=candidate_groups,
        group_scores=group_artifact["scores"],
        group_groups=learned_groups,
        plurality_groups=plurality,
        reliability=reliability_oof,
        selected_groups=selected_groups,
        selected_correct=selected_correct,
        selected_probability=selected_prob_oof,
    )
    np.savez(
        RUNS / "validation_predictions.npz",
        **{name: choices for name, choices in validation_selected.items()},
    )
    asset_bytes = sum(path.stat().st_size for path in (RUNS / "provisional_bundle").rglob("*") if path.is_file())
    dump_json(RUNS / "search_state.json", {
        "status": "paused_before_test", "seed": SEED, "elapsed_seconds": elapsed(),
        "test_touched": False, "asset_bytes": asset_bytes,
        "selected_method": selected_method,
    })
    print_progress("stage4_complete", {
        "selected_method": selected_method, "cv_mean": selected_result["cv_mean"],
        "cv_se": selected_result["cv_se"], "asset_bytes": asset_bytes,
        "validation": validation_results,
    })


if __name__ == "__main__":
    main()
