"""Bounded train-hard CV search and validation confirmation for MP2.

This script never opens test inputs or references. It implements the pre-test
research protocol and writes all evidence under runs/.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import platform
from pathlib import Path
import sys
import time

import numpy as np

from common import DATA, load_inputs, read_jsonl
from selector_features import (
    deterministic_random_choice,
    extract_feature_blocks,
    hard_group_mask,
    labels_from_references,
    longest_choice,
    plurality_choice,
)
from selector_model import (
    VARIANT_BLOCKS,
    choices_from_scores,
    fit_transform,
    save_model,
    score_model,
    train_linear_model,
)


SEED = 20261022
DEFAULT_HAND_L2 = 1e-2
DEFAULT_EMBEDDING_L2 = 1e-1
HAND_GRID = (1e-3, 1e-2, 1e-1)
EMBEDDING_GRID = (1e-2, 1e-1, 1.0)
VARIANTS = (
    "structure_only",
    "raw_embedding_only",
    "centered_embedding_only",
    "hand_centered",
    "hand_raw_centered",
)
COMPLEXITY = {
    "structure_only": 1,
    "raw_embedding_only": 2,
    "centered_embedding_only": 2,
    "hand_centered": 3,
    "hand_raw_centered": 4,
}


def dump_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def accuracy(labels: np.ndarray, choices: list[int] | np.ndarray) -> float:
    choices = np.asarray(choices, dtype=int)
    return 100.0 * float(np.mean(labels[np.arange(len(labels)), choices]))


def make_folds(indices: np.ndarray, seed: int, n_folds: int = 5) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    shuffled = np.asarray(indices, dtype=int).copy()
    rng.shuffle(shuffled)
    return [np.asarray(part, dtype=int) for part in np.array_split(shuffled, n_folds)]


def fold_stats(values: list[float]) -> tuple[float, float]:
    values = np.asarray(values, dtype=np.float64)
    mean = float(values.mean())
    se = float(values.std(ddof=1) / math.sqrt(len(values))) if len(values) > 1 else 0.0
    return mean, se


def baseline_choices(name: str, bank: list[dict]) -> list[int]:
    if name == "fixed_candidate_0":
        return [0 for _ in bank]
    if name == "deterministic_random":
        return [deterministic_random_choice(row) for row in bank]
    if name == "longest_trace":
        return [longest_choice(row) for row in bank]
    if name == "plurality":
        return [plurality_choice(row) for row in bank]
    raise ValueError(name)


def evaluate_baseline_cv(name, bank, labels, hard_indices, folds) -> dict:
    all_choices = np.asarray(baseline_choices(name, bank), dtype=int)
    fold_acc = [accuracy(labels[fold], all_choices[fold]) for fold in folds]
    mean, se = fold_stats(fold_acc)
    return {
        "kind": "baseline", "name": name, "cv_fold_accuracy": fold_acc,
        "cv_mean": mean, "cv_se": se,
        "train_hard_accuracy": accuracy(labels[hard_indices], all_choices[hard_indices]),
    }


def run_cv(
    run_dir: Path,
    key: str,
    variant: str,
    hand_l2: float,
    embedding_l2: float,
    blocks: dict,
    bank: list[dict],
    labels: np.ndarray,
    hard_indices: np.ndarray,
    folds: list[np.ndarray],
    loss_kind: str = "listwise",
) -> dict:
    out = run_dir / "cv" / key
    result_path = out / "result.json"
    if result_path.is_file():
        return json.loads(result_path.read_text(encoding="utf-8"))
    out.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    oof_scores = np.full((len(bank), 4), np.nan, dtype=np.float64)
    oof_choices = np.full(len(bank), -1, dtype=np.int64)
    fold_results = []
    loss_histories = []
    hard_set = set(int(i) for i in hard_indices)
    for fold_no, eval_idx in enumerate(folds):
        eval_set = set(int(i) for i in eval_idx)
        train_idx = np.asarray(sorted(hard_set - eval_set), dtype=int)
        transformed, meta = fit_transform(blocks, variant, train_idx)
        fit_started = time.perf_counter()
        model = train_linear_model(
            transformed[train_idx], labels[train_idx], blocks["valid"][train_idx], meta,
            hand_l2=hand_l2, embedding_l2=embedding_l2,
            seed=SEED + fold_no, loss_kind=loss_kind, max_steps=100,
        )
        scores = np.einsum("nkd,d->nk", transformed[eval_idx], model["weights"])
        fold_bank = [bank[int(i)] for i in eval_idx]
        fold_choices = choices_from_scores(fold_bank, scores, blocks["valid"][eval_idx])
        oof_scores[eval_idx] = scores
        oof_choices[eval_idx] = fold_choices
        fold_acc = accuracy(labels[eval_idx], fold_choices)
        fold_results.append({
            "fold": fold_no, "n_train": len(train_idx), "n_eval": len(eval_idx),
            "accuracy": fold_acc, "fit_seconds": time.perf_counter() - fit_started,
            "steps": len(model["loss_history"]),
            "final_loss": model["loss_history"][-1]["loss"],
        })
        loss_histories.append({"fold": fold_no, "history": model["loss_history"]})

    fold_acc = [fold["accuracy"] for fold in fold_results]
    mean, se = fold_stats(fold_acc)
    result = {
        "kind": "learned", "key": key, "variant": variant,
        "blocks": list(VARIANT_BLOCKS[variant]), "hand_l2": hand_l2,
        "embedding_l2": embedding_l2, "loss_kind": loss_kind,
        "seed": SEED, "cv_fold_accuracy": fold_acc, "cv_mean": mean,
        "cv_se": se, "elapsed_seconds": time.perf_counter() - started,
    }
    dump_json(out / "config.json", {
        "variant": variant, "hand_l2": hand_l2, "embedding_l2": embedding_l2,
        "loss_kind": loss_kind, "seed": SEED,
    })
    dump_json(out / "fold_results.json", fold_results)
    dump_json(out / "loss_histories.json", loss_histories)
    np.savez(out / "oof_predictions.npz", indices=hard_indices,
             scores=oof_scores[hard_indices], choices=oof_choices[hard_indices])
    dump_json(result_path, result)
    return result


def select_with_simplicity(results: list[dict], tolerance_points: float = 0.75) -> dict:
    best = max(results, key=lambda item: item["cv_mean"])
    eligible = [item for item in results if best["cv_mean"] - item["cv_mean"] < tolerance_points]
    return min(eligible, key=lambda item: (COMPLEXITY[item["variant"]], -item["cv_mean"]))


def train_full_model(config: dict, blocks: dict, labels: np.ndarray, hard_indices: np.ndarray) -> dict:
    transformed, meta = fit_transform(blocks, config["variant"], hard_indices)
    return train_linear_model(
        transformed[hard_indices], labels[hard_indices], blocks["valid"][hard_indices], meta,
        hand_l2=config["hand_l2"], embedding_l2=config["embedding_l2"],
        seed=SEED, loss_kind=config["loss_kind"], max_steps=160,
    )


def masked_probabilities(scores: np.ndarray, valid: np.ndarray, temperature: float) -> np.ndarray:
    active = valid.copy()
    active[~active.any(axis=1)] = True
    scaled = scores / temperature
    scaled = np.where(active, scaled, -np.inf)
    scaled -= np.max(scaled, axis=1, keepdims=True)
    probs = np.where(active, np.exp(scaled), 0.0)
    return probs / np.maximum(probs.sum(axis=1, keepdims=True), 1e-300)


def binary_nll(outcomes: np.ndarray, confidence: np.ndarray) -> float:
    confidence = np.clip(confidence, 1e-8, 1 - 1e-8)
    return float(-np.mean(outcomes * np.log(confidence) + (1 - outcomes) * np.log(1 - confidence)))


def fit_temperature(scores, valid, labels, choices) -> tuple[float, float]:
    outcomes = labels[np.arange(len(labels)), choices].astype(np.float64)
    best = (float("inf"), 1.0)
    for temperature in np.logspace(-1.0, 1.0, 121):
        probs = masked_probabilities(scores, valid, float(temperature))
        confidence = probs[np.arange(len(probs)), choices]
        value = binary_nll(outcomes, confidence)
        if value < best[0]:
            best = (value, float(temperature))
    return best[1], best[0]


def calibration_metrics(scores, valid, labels, choices, answers, plurality, temperature) -> dict:
    probs = masked_probabilities(scores, valid, temperature)
    choices = np.asarray(choices, dtype=int)
    confidence = probs[np.arange(len(probs)), choices]
    outcomes = labels[np.arange(len(labels)), choices].astype(np.float64)
    entropy = -np.sum(np.where(probs > 0, probs * np.log(np.maximum(probs, 1e-300)), 0.0), axis=1)
    cluster_probability = np.zeros(len(probs), dtype=np.float64)
    for i, choice in enumerate(choices):
        chosen_answer = answers[i][choice]
        cluster_probability[i] = sum(
            probs[i, j] for j in range(4) if answers[i][j] == chosen_answer
        )
    margin = np.sort(probs, axis=1)[:, -1] - np.sort(probs, axis=1)[:, -2]
    bins = []
    ece = 0.0
    edges = np.linspace(0.0, 1.0, 6)
    for b in range(5):
        mask = (confidence >= edges[b]) & (confidence < edges[b + 1] if b < 4 else confidence <= 1.0)
        if mask.any():
            acc = float(outcomes[mask].mean())
            conf = float(confidence[mask].mean())
            ece += mask.mean() * abs(acc - conf)
            bins.append({"lower": edges[b], "upper": edges[b + 1], "n": int(mask.sum()),
                         "accuracy": acc, "mean_confidence": conf})
    order = np.argsort(-confidence)
    cumulative_risk = np.cumsum(1.0 - outcomes[order]) / np.arange(1, len(outcomes) + 1)
    aurc = float(cumulative_risk.mean())
    risk_coverage = []
    for coverage in np.linspace(0.1, 1.0, 10):
        kept = max(1, int(math.ceil(coverage * len(outcomes))))
        risk_coverage.append({
            "coverage": float(coverage),
            "n": kept,
            "risk": float(cumulative_risk[kept - 1]),
        })
    answer_agreement = np.mean([
        answers[i][int(choices[i])] == answers[i][int(plurality[i])]
        for i in range(len(choices))
    ])
    return {
        "temperature": temperature, "nll": binary_nll(outcomes, confidence),
        "ece_5bin": float(ece), "aurc": aurc,
        "mean_margin": float(margin.mean()),
        "mean_cluster_probability": float(cluster_probability.mean()),
        "mean_entropy": float(entropy.mean()),
        "plurality_comparator_answer_agreement": float(answer_agreement),
        "bins": bins, "risk_coverage": risk_coverage,
    }


def stratified_metrics(labels, choices, answers) -> dict:
    choices = np.asarray(choices, dtype=int)
    distinct = np.asarray([len({a for a in row if a is not None}) for row in answers])
    positives = labels.sum(axis=1)
    result = {}
    for name, values in (("distinct_answers", distinct), ("correct_candidate_count", positives)):
        result[name] = {}
        for value in sorted(set(int(x) for x in values)):
            mask = values == value
            result[name][str(value)] = {"n": int(mask.sum()), "accuracy": accuracy(labels[mask], choices[mask])}
    return result


def validation_result(name, labels, choices, answers, hard_mask) -> dict:
    return {
        "name": name,
        "validation_all": accuracy(labels, choices),
        "validation_hard": accuracy(labels[hard_mask], np.asarray(choices)[hard_mask]),
        "n_all": len(labels), "n_hard": int(hard_mask.sum()),
        "strata": stratified_metrics(labels, choices, answers),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-dir", type=Path, default=Path("runs"))
    args = parser.parse_args()
    run_dir = args.runs_dir
    run_dir.mkdir(parents=True, exist_ok=True)
    wall_started = time.time()
    perf_started = time.perf_counter()
    dump_json(run_dir / "search_state.json", {
        "status": "running", "seed": SEED, "started_unix": wall_started,
        "budget_minutes": 90, "test_touched": False,
    })
    dump_json(run_dir / "environment.json", {
        "python": sys.version, "numpy": np.__version__, "platform": platform.platform(),
        "processor": platform.processor(), "logical_cpu_count": os.cpu_count(),
        "thread_environment": {name: os.environ.get(name) for name in
                               ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")},
    })

    train_bank, train_arrays = load_inputs("train", DATA)
    train_refs = read_jsonl(DATA / "gsm8k_train.jsonl")
    train_blocks = extract_feature_blocks(train_bank, train_arrays["embeddings"])
    train_labels = labels_from_references(train_bank, train_refs)
    train_hard = hard_group_mask(train_blocks["answers"])
    hard_indices = np.flatnonzero(train_hard)
    folds = make_folds(hard_indices, SEED)
    dump_json(run_dir / "folds.json", {"seed": SEED, "fold_indices": [fold.tolist() for fold in folds]})
    dump_json(run_dir / "data_summary.json", {
        "train_questions": len(train_bank), "train_embedding_shape": list(train_arrays["embeddings"].shape),
        "train_hard_groups": int(train_hard.sum()), "train_nonhard_groups": int((~train_hard).sum()),
        "train_correct_candidates_mean": float(train_labels.sum(axis=1).mean()),
    })

    summary_rows = []
    baseline_results = []
    for name in ("fixed_candidate_0", "deterministic_random", "longest_trace", "plurality"):
        result = evaluate_baseline_cv(name, train_bank, train_labels, hard_indices, folds)
        baseline_results.append(result)
        summary_rows.append(result)

    cache = {}
    initial = []
    for variant in VARIANTS:
        key = f"{variant}__h{DEFAULT_HAND_L2:g}__e{DEFAULT_EMBEDDING_L2:g}__listwise"
        result = run_cv(run_dir, key, variant, DEFAULT_HAND_L2, DEFAULT_EMBEDDING_L2,
                        train_blocks, train_bank, train_labels, hard_indices, folds)
        cache[(variant, DEFAULT_HAND_L2, DEFAULT_EMBEDDING_L2, "listwise")] = result
        initial.append(result)
        summary_rows.append(result)

    top_two_variants = [item["variant"] for item in sorted(initial, key=lambda x: -x["cv_mean"])[:2]]
    searched = list(initial)
    for variant in top_two_variants:
        hand_values = HAND_GRID if any(b in VARIANT_BLOCKS[variant] for b in ("hand", "relative")) else (DEFAULT_HAND_L2,)
        emb_values = EMBEDDING_GRID if any(b in VARIANT_BLOCKS[variant] for b in ("raw", "centered")) else (DEFAULT_EMBEDDING_L2,)
        for hand_l2 in hand_values:
            for embedding_l2 in emb_values:
                cache_key = (variant, hand_l2, embedding_l2, "listwise")
                if cache_key not in cache:
                    key = f"{variant}__h{hand_l2:g}__e{embedding_l2:g}__listwise"
                    cache[cache_key] = run_cv(
                        run_dir, key, variant, hand_l2, embedding_l2,
                        train_blocks, train_bank, train_labels, hard_indices, folds,
                    )
                    summary_rows.append(cache[cache_key])
                searched.append(cache[cache_key])

    # Best regularization per variant; then apply the pre-declared simplicity rule.
    best_by_variant = []
    for variant in VARIANTS:
        candidates = [item for item in summary_rows if item.get("variant") == variant]
        best_by_variant.append(max(candidates, key=lambda x: x["cv_mean"]))
    provisional_config = select_with_simplicity(best_by_variant)

    # Required loss ablation on the chosen information blocks.
    uniform_key = (
        f"{provisional_config['variant']}__h{provisional_config['hand_l2']:g}__"
        f"e{provisional_config['embedding_l2']:g}__uniform_positive"
    )
    uniform_result = run_cv(
        run_dir, uniform_key, provisional_config["variant"], provisional_config["hand_l2"],
        provisional_config["embedding_l2"], train_blocks, train_bank, train_labels,
        hard_indices, folds, loss_kind="uniform_positive",
    )
    summary_rows.append(uniform_result)

    top_three = sorted(best_by_variant, key=lambda x: -x["cv_mean"])[:3]

    # One validation pass for the CV top three, plus rule baselines.
    val_bank, val_arrays = load_inputs("validation", DATA)
    val_refs = read_jsonl(DATA / "gsm8k_validation.jsonl")
    val_blocks = extract_feature_blocks(val_bank, val_arrays["embeddings"])
    val_labels = labels_from_references(val_bank, val_refs)
    val_hard = hard_group_mask(val_blocks["answers"])
    validation_results = []
    for name in ("fixed_candidate_0", "deterministic_random", "longest_trace", "plurality"):
        validation_results.append(validation_result(
            name, val_labels, baseline_choices(name, val_bank), val_blocks["answers"], val_hard
        ))

    trained_models = {}
    for config in top_three:
        model = train_full_model(config, train_blocks, train_labels, hard_indices)
        scores = score_model(model, val_blocks)
        choices = choices_from_scores(val_bank, scores, val_blocks["valid"])
        name = config["key"]
        validation_results.append(validation_result(name, val_labels, choices, val_blocks["answers"], val_hard))
        trained_models[name] = (model, scores, choices)

    # Keep the CV-selected configuration. Validation is confirmation, not a new feature search.
    provisional_model = train_full_model(provisional_config, train_blocks, train_labels, hard_indices)
    provisional_dir = run_dir / "provisional_model_bundle"

    oof_path = run_dir / "cv" / provisional_config["key"] / "oof_predictions.npz"
    with np.load(oof_path, allow_pickle=False) as data:
        oof_indices = data["indices"].astype(int)
        oof_scores = data["scores"].astype(np.float64)
        oof_choices = data["choices"].astype(int)
    temperature, oof_nll = fit_temperature(
        oof_scores, train_blocks["valid"][oof_indices], train_labels[oof_indices], oof_choices
    )
    save_model(provisional_model, provisional_dir, temperature=temperature)

    val_scores = score_model(provisional_model, val_blocks)
    val_choices = choices_from_scores(val_bank, val_scores, val_blocks["valid"])
    val_plurality = baseline_choices("plurality", val_bank)
    uncertainty = calibration_metrics(
        val_scores, val_blocks["valid"], val_labels, val_choices,
        val_blocks["answers"], val_plurality, temperature,
    )
    uncertainty["oof_temperature_fit_nll"] = oof_nll
    dump_json(run_dir / "validation_uncertainty.json", uncertainty)
    dump_json(run_dir / "validation_internal.json", validation_results + [
        validation_result("provisional_best", val_labels, val_choices, val_blocks["answers"], val_hard)
    ])

    # Persist compact experiment tables.
    serializable_rows = []
    for row in summary_rows:
        serializable_rows.append({
            "kind": row["kind"], "name": row.get("name", row.get("key")),
            "variant": row.get("variant", ""), "hand_l2": row.get("hand_l2", ""),
            "embedding_l2": row.get("embedding_l2", ""), "loss_kind": row.get("loss_kind", ""),
            "cv_mean": row["cv_mean"], "cv_se": row["cv_se"],
            "cv_fold_accuracy": row["cv_fold_accuracy"],
        })
    dump_json(run_dir / "experiment_summary.json", {
        "seed": SEED, "top_two_regularization_search": top_two_variants,
        "cv_top_three": top_three, "provisional_best": provisional_config,
        "uniform_positive_ablation": uniform_result,
        "experiments": serializable_rows, "validation": validation_results,
    })
    with (run_dir / "experiment_summary.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=[
            "kind", "name", "variant", "hand_l2", "embedding_l2", "loss_kind", "cv_mean", "cv_se"
        ])
        writer.writeheader()
        for row in serializable_rows:
            writer.writerow({key: row[key] for key in writer.fieldnames})

    elapsed = time.perf_counter() - perf_started
    dump_json(run_dir / "search_state.json", {
        "status": "paused_pre_test", "seed": SEED, "started_unix": wall_started,
        "ended_unix": time.time(), "elapsed_seconds": elapsed,
        "budget_minutes": 90, "budget_exhausted": elapsed >= 90 * 60,
        "test_touched": False, "provisional_checkpoint": str(provisional_dir),
    })
    print(json.dumps({
        "status": "paused_pre_test", "elapsed_seconds": elapsed,
        "train_hard": int(train_hard.sum()), "validation_hard": int(val_hard.sum()),
        "provisional_best": provisional_config,
        "provisional_validation_all": accuracy(val_labels, val_choices),
        "provisional_validation_hard": accuracy(val_labels[val_hard], np.asarray(val_choices)[val_hard]),
        "temperature": temperature,
    }, indent=2))


if __name__ == "__main__":
    main()

