"""Bounded stage-2 train-hard CV search.

The script reuses the exact stage-1 folds, fixes one winner, and stops before
opening validation. Official validation prediction/scoring is a separate step.
"""
from __future__ import annotations

from collections import Counter
import csv
import json
import math
from pathlib import Path
import time

import numpy as np

from common import DATA, load_inputs, read_jsonl
from selector_features import (
    extract_feature_blocks,
    hard_group_mask,
    labels_from_references,
    plurality_choice,
)
from selector_model import (
    cluster_choices_from_scores,
    fit_transform,
    save_model,
    train_linear_model,
)


SEED = 20261022
STAGE2 = Path("runs/stage2")
STAGE1_OOF = Path("runs/cv/hand_centered__h0.01__e1__listwise/oof_predictions.npz")
MIN_COMPLEX_GAIN = 0.75
MIN_STABLE_FOLDS = 4
HAND_L2 = 1e-2
EMBEDDING_L2 = 1.0


def dump_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def accuracy(labels: np.ndarray, choices: np.ndarray) -> float:
    return 100.0 * float(np.mean(labels[np.arange(len(labels)), choices]))


def fold_stats(values: list[float]) -> tuple[float, float]:
    array = np.asarray(values, dtype=np.float64)
    return float(array.mean()), float(array.std(ddof=1) / math.sqrt(len(array)))


def fold_accuracies(
    labels: np.ndarray,
    choices: np.ndarray,
    hard_indices: np.ndarray,
    folds: list[np.ndarray],
) -> list[float]:
    positions = {int(index): pos for pos, index in enumerate(hard_indices)}
    values = []
    for fold in folds:
        local = np.asarray([positions[int(index)] for index in fold], dtype=int)
        values.append(accuracy(labels[local], choices[local]))
    return values


def result_from_choices(
    key: str,
    method: str,
    labels: np.ndarray,
    choices: np.ndarray,
    hard_indices: np.ndarray,
    folds: list[np.ndarray],
    config: dict,
) -> dict:
    fold_acc = fold_accuracies(labels, choices, hard_indices, folds)
    mean, se = fold_stats(fold_acc)
    return {
        "key": key,
        "method": method,
        "cv_fold_accuracy": fold_acc,
        "cv_mean": mean,
        "cv_se": se,
        **config,
    }


def stable_advance(reference: dict, candidates: list[dict]) -> tuple[dict, dict]:
    best = max(candidates, key=lambda item: item["cv_mean"])
    deltas = [
        candidate - base
        for candidate, base in zip(best["cv_fold_accuracy"], reference["cv_fold_accuracy"])
    ]
    decision = {
        "reference": reference["key"],
        "best_candidate": best["key"],
        "gain_points": best["cv_mean"] - reference["cv_mean"],
        "nondecreasing_folds": int(sum(delta >= -1e-12 for delta in deltas)),
        "fold_deltas": deltas,
        "required_gain_points": MIN_COMPLEX_GAIN,
        "required_nondecreasing_folds": MIN_STABLE_FOLDS,
    }
    decision["advanced"] = (
        decision["gain_points"] >= MIN_COMPLEX_GAIN
        and decision["nondecreasing_folds"] >= MIN_STABLE_FOLDS
    )
    return (best if decision["advanced"] else reference), decision


def answer_pattern(answers: list[str | None]) -> str:
    counts = sorted(Counter(a for a in answers if a is not None).values(), reverse=True)
    return "-".join(str(value) for value in counts) if counts else "none-valid"


def make_weight_schemes(
    n_all: int,
    hard_indices: np.ndarray,
    labels_hard: np.ndarray,
    answers_hard: list[list[str | None]],
    stage1_choices: np.ndarray,
    plurality_choices: np.ndarray,
) -> dict[str, np.ndarray]:
    correct_count = labels_hard.sum(axis=1)
    patterns = np.asarray([answer_pattern(row) for row in answers_hard], dtype=object)
    plurality_correct = labels_hard[np.arange(len(labels_hard)), plurality_choices]
    wrong_3_to_1 = (patterns == "3-1") & ~plurality_correct
    stage1_answers = np.asarray(
        [answers_hard[i][stage1_choices[i]] for i in range(len(answers_hard))], dtype=object
    )
    plurality_answers = np.asarray(
        [answers_hard[i][plurality_choices[i]] for i in range(len(answers_hard))], dtype=object
    )
    conflict = stage1_answers != plurality_answers
    diverse = np.asarray([
        len({a for a in row if a is not None}) >= 3 for row in answers_hard
    ], dtype=bool)
    single = correct_count == 1

    local = {
        "uniform": np.ones(len(hard_indices)),
        "single_correct_1p5": np.where(single, 1.5, 1.0),
        "single_correct_2": np.where(single, 2.0, 1.0),
        "wrong_3to1_2": np.where(wrong_3_to_1, 2.0, 1.0),
        "single1p5_wrong3to1_2": np.maximum(
            np.where(single, 1.5, 1.0), np.where(wrong_3_to_1, 2.0, 1.0)
        ),
        "single2_wrong3to1_3": np.maximum(
            np.where(single, 2.0, 1.0), np.where(wrong_3_to_1, 3.0, 1.0)
        ),
        "stage1_conflict_1p5": np.where(conflict, 1.5, 1.0),
        "answer_diverse_1p5": np.where(diverse, 1.5, 1.0),
    }
    result = {}
    for name, local_values in local.items():
        values = np.ones(n_all, dtype=np.float64)
        values[hard_indices] = local_values
        result[name] = values
    return result


def budget_allows_new_experiment() -> bool:
    state = json.loads((STAGE2 / "search_state.json").read_text(encoding="utf-8"))
    return time.time() - float(state["started_unix"]) < 90 * 60


def run_cv(
    key: str,
    config: dict,
    blocks: dict,
    bank: list[dict],
    labels_all: np.ndarray,
    hard_indices: np.ndarray,
    folds: list[np.ndarray],
    group_weights: np.ndarray,
) -> dict:
    out = STAGE2 / "cv" / key
    result_path = out / "result.json"
    if result_path.is_file():
        return json.loads(result_path.read_text(encoding="utf-8"))
    if not budget_allows_new_experiment():
        raise TimeoutError("Stage-2 90-minute budget reached before starting a new experiment")
    out.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    hard_set = set(int(index) for index in hard_indices)
    oof_scores = np.full((len(bank), 4), np.nan, dtype=np.float64)
    oof_choices = np.full(len(bank), -1, dtype=np.int64)
    fold_results = []
    histories = []
    for fold_no, eval_idx in enumerate(folds):
        eval_set = set(int(index) for index in eval_idx)
        train_idx = np.asarray(sorted(hard_set - eval_set), dtype=int)
        transformed, meta = fit_transform(blocks, config["variant"], train_idx)
        fit_started = time.perf_counter()
        model = train_linear_model(
            transformed[train_idx],
            labels_all[train_idx],
            blocks["valid"][train_idx],
            meta,
            hand_l2=HAND_L2,
            embedding_l2=EMBEDDING_L2,
            seed=SEED + fold_no,
            loss_kind="listwise",
            max_steps=100,
            group_weights=group_weights[train_idx],
            pairwise_weight=config["pairwise_weight"],
            weight_scheme=config["weight_scheme"],
        )
        scores = np.einsum("nkd,d->nk", transformed[eval_idx], model["weights"])
        eval_bank = [bank[int(index)] for index in eval_idx]
        choices = cluster_choices_from_scores(
            eval_bank,
            scores,
            blocks["valid"][eval_idx],
            rule=config["decision_rule"],
            alpha=config["cluster_alpha"],
        )
        choices = np.asarray(choices, dtype=int)
        oof_scores[eval_idx] = scores
        oof_choices[eval_idx] = choices
        fold_results.append({
            "fold": fold_no,
            "n_train": len(train_idx),
            "n_eval": len(eval_idx),
            "accuracy": accuracy(labels_all[eval_idx], choices),
            "fit_seconds": time.perf_counter() - fit_started,
            "steps": len(model["loss_history"]),
            "final_loss": model["loss_history"][-1]["loss"],
        })
        histories.append({"fold": fold_no, "history": model["loss_history"]})
    fold_acc = [row["accuracy"] for row in fold_results]
    mean, se = fold_stats(fold_acc)
    result = {
        "key": key,
        "method": "learned",
        **config,
        "hand_l2": HAND_L2,
        "embedding_l2": EMBEDDING_L2,
        "seed": SEED,
        "cv_fold_accuracy": fold_acc,
        "cv_mean": mean,
        "cv_se": se,
        "elapsed_seconds": time.perf_counter() - started,
    }
    dump_json(out / "config.json", result)
    dump_json(out / "fold_results.json", fold_results)
    dump_json(out / "loss_histories.json", histories)
    np.savez(
        out / "oof_predictions.npz",
        indices=hard_indices,
        scores=oof_scores[hard_indices],
        choices=oof_choices[hard_indices],
    )
    dump_json(result_path, result)
    return result


def main() -> None:
    total_started = time.perf_counter()
    state = json.loads((STAGE2 / "search_state.json").read_text(encoding="utf-8"))
    if state.get("validation_runs", 0) != 0:
        raise RuntimeError("Stage-2 validation has already run; CV search is frozen")

    bank, arrays = load_inputs("train", DATA)
    references = read_jsonl(DATA / "gsm8k_train.jsonl")
    blocks = extract_feature_blocks(bank, arrays["embeddings"])
    labels_all = labels_from_references(bank, references)
    hard_mask = hard_group_mask(blocks["answers"])
    hard_indices = np.flatnonzero(hard_mask)
    labels_hard = labels_all[hard_indices]
    answers_hard = [blocks["answers"][int(index)] for index in hard_indices]
    bank_hard = [bank[int(index)] for index in hard_indices]
    valid_hard = blocks["valid"][hard_indices]

    folds_payload = json.loads(Path("runs/folds.json").read_text(encoding="utf-8"))
    folds = [np.asarray(fold, dtype=int) for fold in folds_payload["fold_indices"]]
    if folds_payload.get("seed") != SEED or set(np.concatenate(folds)) != set(hard_indices):
        raise ValueError("Frozen stage-1 folds are incompatible with current train-hard groups")

    with np.load(STAGE1_OOF, allow_pickle=False) as data:
        if not np.array_equal(data["indices"].astype(int), hard_indices):
            raise ValueError("Stage-1 OOF index mismatch")
        stage1_scores = data["scores"].astype(np.float64)
        stage1_choices = data["choices"].astype(int)
    plurality = np.asarray([plurality_choice(row) for row in bank_hard], dtype=int)

    experiments = []
    decisions = []

    # A. Answer-cluster pooling on the frozen stage-1 OOF scores.
    cluster_specs = [
        ("candidate", 0.0),
        ("cluster_mean", 0.0),
        ("cluster_logsumexp", 0.0),
        ("cluster_max_plus_log_count", 0.25),
        ("cluster_max_plus_log_count", 0.5),
        ("cluster_max_plus_log_count", 1.0),
    ]
    cluster_results = []
    for rule, alpha in cluster_specs:
        choices = np.asarray(cluster_choices_from_scores(
            bank_hard, stage1_scores, valid_hard, rule=rule, alpha=alpha
        ), dtype=int)
        key = f"stage1_oof__{rule}__a{alpha:g}"
        result = result_from_choices(
            key, "cluster_pooling", labels_hard, choices, hard_indices, folds,
            {
                "variant": "hand_centered",
                "weight_scheme": "uniform",
                "pairwise_weight": 0.0,
                "decision_rule": rule,
                "cluster_alpha": alpha,
                "hand_l2": HAND_L2,
                "embedding_l2": EMBEDDING_L2,
            },
        )
        cluster_results.append(result)
        experiments.append(result)
    cluster_reference = cluster_results[0]
    selected, decision = stable_advance(cluster_reference, cluster_results[1:])
    decision["stage"] = "cluster_pooling"
    decisions.append(decision)
    selected_rule = selected["decision_rule"]
    selected_alpha = selected["cluster_alpha"]
    current = selected

    # D. Fixed 32-dimensional lexical interaction replacement block.
    lexical_config = {
        "variant": "stage2_hand_centered",
        "weight_scheme": "uniform",
        "pairwise_weight": 0.0,
        "decision_rule": selected_rule,
        "cluster_alpha": selected_alpha,
    }
    lexical = run_cv(
        "stage2_hand_centered__uniform__pair0",
        lexical_config,
        blocks,
        bank,
        labels_all,
        hard_indices,
        folds,
        np.ones(len(bank)),
    )
    experiments.append(lexical)
    current, decision = stable_advance(current, [lexical])
    decision["stage"] = "lexical_interaction"
    decisions.append(decision)
    selected_variant = current["variant"]

    # B. Pre-declared targeted weighting schemes.
    weight_schemes = make_weight_schemes(
        len(bank), hard_indices, labels_hard, answers_hard, stage1_choices, plurality
    )
    weighted_results = []
    for scheme in (
        "single_correct_1p5",
        "single_correct_2",
        "wrong_3to1_2",
        "single1p5_wrong3to1_2",
        "single2_wrong3to1_3",
        "stage1_conflict_1p5",
        "answer_diverse_1p5",
    ):
        config = {
            "variant": selected_variant,
            "weight_scheme": scheme,
            "pairwise_weight": 0.0,
            "decision_rule": selected_rule,
            "cluster_alpha": selected_alpha,
        }
        result = run_cv(
            f"{selected_variant}__{scheme}__pair0",
            config,
            blocks,
            bank,
            labels_all,
            hard_indices,
            folds,
            weight_schemes[scheme],
        )
        weighted_results.append(result)
        experiments.append(result)
    current, decision = stable_advance(current, weighted_results)
    decision["stage"] = "targeted_weighting"
    decisions.append(decision)
    selected_weight_scheme = current["weight_scheme"]

    # C. Pairwise/listwise hybrids on the selected features and weights.
    hybrid_results = []
    for pairwise_weight in (0.1, 0.25, 0.5, 1.0):
        config = {
            "variant": selected_variant,
            "weight_scheme": selected_weight_scheme,
            "pairwise_weight": pairwise_weight,
            "decision_rule": selected_rule,
            "cluster_alpha": selected_alpha,
        }
        result = run_cv(
            f"{selected_variant}__{selected_weight_scheme}__pair{pairwise_weight:g}",
            config,
            blocks,
            bank,
            labels_all,
            hard_indices,
            folds,
            weight_schemes[selected_weight_scheme],
        )
        hybrid_results.append(result)
        experiments.append(result)
    current, decision = stable_advance(current, hybrid_results)
    decision["stage"] = "pairwise_listwise_hybrid"
    decisions.append(decision)

    # The selection is now frozen before any validation access.
    winner = current
    full_features, meta = fit_transform(blocks, winner["variant"], hard_indices)
    model = train_linear_model(
        full_features[hard_indices],
        labels_all[hard_indices],
        blocks["valid"][hard_indices],
        meta,
        hand_l2=HAND_L2,
        embedding_l2=EMBEDDING_L2,
        seed=SEED,
        loss_kind="listwise",
        max_steps=160,
        group_weights=weight_schemes[winner["weight_scheme"]][hard_indices],
        pairwise_weight=winner["pairwise_weight"],
        weight_scheme=winner["weight_scheme"],
    )
    checkpoint = STAGE2 / "provisional_model_bundle"
    save_model(
        model,
        checkpoint,
        decision_rule=winner["decision_rule"],
        cluster_alpha=winner["cluster_alpha"],
    )
    checkpoint_bytes = sum(path.stat().st_size for path in checkpoint.rglob("*") if path.is_file())

    compact = []
    for item in experiments:
        compact.append({
            key: item[key] for key in (
                "key", "method", "variant", "weight_scheme", "pairwise_weight",
                "decision_rule", "cluster_alpha", "cv_mean", "cv_se", "cv_fold_accuracy"
            )
        })
    dump_json(STAGE2 / "experiment_summary.json", {
        "seed": SEED,
        "folds_exactly_reused": True,
        "selection_rule": {
            "minimum_gain_points": MIN_COMPLEX_GAIN,
            "minimum_nondecreasing_folds": MIN_STABLE_FOLDS,
        },
        "decisions": decisions,
        "winner_pre_validation": winner,
        "experiments": compact,
        "checkpoint_bytes": checkpoint_bytes,
        "validation_not_yet_run": True,
        "test_touched": False,
    })
    with (STAGE2 / "experiment_summary.csv").open("w", newline="", encoding="utf-8") as stream:
        fields = [
            "key", "method", "variant", "weight_scheme", "pairwise_weight",
            "decision_rule", "cluster_alpha", "cv_mean", "cv_se",
        ]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in compact:
            writer.writerow({name: row[name] for name in fields})
    dump_json(STAGE2 / "winner_pre_validation.json", winner)
    state.update({
        "status": "winner_fixed_pre_validation",
        "winner_fixed_unix": time.time(),
        "cv_runner_elapsed_seconds": time.perf_counter() - total_started,
        "validation_runs": 0,
        "test_touched": False,
        "provisional_checkpoint": str(checkpoint),
        "checkpoint_bytes": checkpoint_bytes,
    })
    dump_json(STAGE2 / "search_state.json", state)
    print(json.dumps({
        "status": state["status"],
        "winner": winner,
        "decisions": decisions,
        "checkpoint": str(checkpoint),
        "checkpoint_bytes": checkpoint_bytes,
        "elapsed_seconds": state["cv_runner_elapsed_seconds"],
    }, indent=2))


if __name__ == "__main__":
    main()
