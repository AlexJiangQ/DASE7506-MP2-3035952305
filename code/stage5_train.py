"""Train and evaluate Stage 5 CREO-RR without accessing test data."""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import time

import numpy as np

from common import DATA, PROTOCOL, load_inputs, read_jsonl
from creo_features import extract_group_features, group_labels
from run_stage3 import local_folds_from_global, slice_blocks
from selector_features import extract_feature_blocks, hard_group_mask, labels_from_references
from stage5_features import (
    FEATURE_SETS,
    SOURCE_NAMES,
    assemble_state,
    build_full_state,
    build_residual_examples,
    combine_feature_blocks,
    groups_to_candidates,
    select_revisions,
    source_example_mask,
)
from stage5_model import (
    calibration_metrics,
    cross_fitted_temperature,
    fit_temperature,
    multiclass_nll,
    predict_logits,
    save_stage5_bundle,
    softmax,
    train_multinomial,
)


SEED = 20261022
RUNS = Path(os.environ.get("STAGE5_RUN_DIR", "runs/stage5"))
ANCHOR_CHECKPOINT = Path("runs/stage3/provisional_creo_bundle")
STAGE1_OOF = Path("runs/cv/hand_centered__h0.01__e1__listwise/oof_predictions.npz")
STAGE2_OOF = Path("runs/stage2/cv/hand_centered__uniform__pair0.1/oof_predictions.npz")
STAGE3_PREDICTIONS = Path("runs/stage3/selected_predictions.npz")
PROPOSAL_SETS = {
    "plurality": ("plurality",),
    "stage2": ("stage2",),
    "stage3_group": ("stage3_group",),
    "all_unique": SOURCE_NAMES,
}
HARM_COSTS = (1.5, 2.0, 3.0)
THRESHOLDS = (0.0, 0.05, 0.10)


def dump_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def protected_hashes() -> dict[str, str]:
    roots = [
        Path("runs/provisional_model_bundle"),
        Path("runs/final/frozen_checkpoint"),
        Path("runs/stage3/provisional_creo_bundle"),
    ]
    files = [
        Path("runs/folds.json"), STAGE1_OOF, STAGE2_OOF,
        STAGE3_PREDICTIONS, Path("runs/stage3/experiment_summary.json"),
    ]
    for root in roots:
        files.extend(path for path in root.rglob("*") if path.is_file())
    return {path.as_posix(): sha256(path) for path in sorted(set(files))}


def slice_examples(examples: dict, row_mask: np.ndarray) -> dict:
    row_mask = np.asarray(row_mask, dtype=bool)
    result = {
        "blocks": {name: values[row_mask] for name, values in examples["blocks"].items()},
        "block_names": examples["block_names"],
        "question_indices": examples["question_indices"][row_mask],
        "proposal_groups": examples["proposal_groups"][row_mask],
        "source_masks": examples["source_masks"][row_mask],
        "anchor_groups": examples["anchor_groups"],
    }
    for name in ("outcomes", "anchor_correct", "proposal_correct"):
        if name in examples:
            result[name] = examples[name][row_mask]
    if "group_labels" in examples:
        result["group_labels"] = examples["group_labels"]
    return result


def fold_stats(values: list[float]) -> tuple[float, float]:
    array = np.asarray(values, dtype=np.float64)
    return float(array.mean()), float(array.std(ddof=1) / math.sqrt(len(array)))


def regime_name(state: dict, i: int) -> str:
    supports = sorted([
        int(state["group_data"]["support"][i, g])
        for g in np.flatnonzero(state["group_data"]["valid"][i])
    ], reverse=True)
    pattern = "-".join(map(str, supports))
    return pattern if pattern in {"3-1", "2-2", "2-1-1", "1-1-1-1"} else "other"


def revision_metrics(
    state: dict,
    group_labels_array: np.ndarray,
    selected: np.ndarray,
    hard_mask: np.ndarray | None = None,
    folds: list[np.ndarray] | None = None,
) -> dict:
    n = len(selected)
    if hard_mask is None:
        hard_mask = np.ones(n, dtype=bool)
    anchor = np.asarray(state["anchor_groups"], dtype=np.int64)
    plurality = np.asarray(state["plurality_groups"], dtype=np.int64)
    row = np.arange(n)
    anchor_correct = group_labels_array[row, anchor]
    selected_correct = group_labels_array[row, selected]
    plurality_correct = group_labels_array[row, plurality]
    revised = selected != anchor
    beneficial = revised & ~anchor_correct & selected_correct
    harmful = revised & anchor_correct & ~selected_correct
    neutral = revised & ~(beneficial | harmful)
    hard_indices = np.flatnonzero(hard_mask)
    correct_plurality = hard_mask & plurality_correct
    wrong_plurality = hard_mask & ~plurality_correct
    candidate_correct_counts = state.get(
        "candidate_correct_counts", group_labels_array.sum(axis=1)
    )
    single_correct = np.asarray(candidate_correct_counts) == 1
    metrics = {
        "all_correct": int(selected_correct.sum()),
        "all_n": n,
        "all_accuracy": 100.0 * float(selected_correct.mean()),
        "hard_correct": int(selected_correct[hard_mask].sum()),
        "hard_n": int(hard_mask.sum()),
        "hard_accuracy": 100.0 * float(selected_correct[hard_mask].mean()),
        "anchor_correct": int(anchor_correct.sum()),
        "revisions": int(revised.sum()),
        "beneficial_revisions": int(beneficial.sum()),
        "harmful_revisions": int(harmful.sum()),
        "neutral_revisions": int(neutral.sum()),
        "net_corrected": int(beneficial.sum() - harmful.sum()),
        "revision_precision": float(beneficial.sum() / revised.sum()) if revised.any() else None,
        "revision_coverage": float(revised.mean()),
        "wrong_plurality_correct": int(selected_correct[wrong_plurality].sum()),
        "wrong_plurality_n": int(wrong_plurality.sum()),
        "correct_plurality_correct": int(selected_correct[correct_plurality].sum()),
        "correct_plurality_n": int(correct_plurality.sum()),
        "correct_plurality_accuracy": (
            100.0 * float(selected_correct[correct_plurality].mean())
            if correct_plurality.any() else None
        ),
        "single_correct_correct": int(selected_correct[single_correct].sum()),
        "single_correct_n": int(single_correct.sum()),
    }
    regimes = {}
    for name in ("3-1", "2-2", "2-1-1", "1-1-1-1", "other"):
        mask = np.asarray([regime_name(state, i) == name for i in range(n)])
        regimes[name] = {
            "n": int(mask.sum()),
            "correct": int(selected_correct[mask].sum()),
            "accuracy": 100.0 * float(selected_correct[mask].mean()) if mask.any() else None,
            "revisions": int(revised[mask].sum()),
            "beneficial": int(beneficial[mask].sum()),
            "harmful": int(harmful[mask].sum()),
        }
    metrics["regimes"] = regimes
    if folds is not None:
        stage2 = np.asarray(state["source_groups"]["stage2"], dtype=np.int64)
        stage2_correct = group_labels_array[row, stage2]
        fold_rows = []
        for number, indices in enumerate(folds):
            fold_rows.append({
                "fold": number,
                "n": len(indices),
                "correct": int(selected_correct[indices].sum()),
                "accuracy": 100.0 * float(selected_correct[indices].mean()),
                "delta_vs_anchor": int(selected_correct[indices].sum() - anchor_correct[indices].sum()),
                "delta_vs_stage2": int(selected_correct[indices].sum() - stage2_correct[indices].sum()),
                "beneficial": int(beneficial[indices].sum()),
                "harmful": int(harmful[indices].sum()),
            })
        mean, standard_error = fold_stats([item["accuracy"] for item in fold_rows])
        metrics.update({
            "folds": fold_rows,
            "cv_mean": mean,
            "cv_standard_error": standard_error,
            "folds_non_decreasing_vs_anchor": int(sum(item["delta_vs_anchor"] >= 0 for item in fold_rows)),
        })
    return metrics


def cross_fit(
    features: np.ndarray,
    examples: dict,
    folds: list[np.ndarray],
    class_weight_mode: str,
    harm_aux_weight: float,
) -> tuple[np.ndarray, list[dict], list[dict]]:
    logits = np.zeros((len(features), 3), dtype=np.float64)
    models = []
    histories = []
    all_questions = np.arange(len(examples["anchor_groups"]), dtype=np.int64)
    for fold_no, eval_questions in enumerate(folds):
        train_questions = np.setdiff1d(all_questions, eval_questions)
        model = train_multinomial(
            features, examples["outcomes"], examples["question_indices"], train_questions,
            class_weight_mode=class_weight_mode, harm_aux_weight=harm_aux_weight,
        )
        eval_rows = np.isin(examples["question_indices"], eval_questions)
        logits[eval_rows] = predict_logits(model, features[eval_rows])
        histories.append({"fold": fold_no, "history": model["loss_history"]})
        models.append(model)
    return logits, models, histories


def evaluate_oof_config(
    key: str,
    state: dict,
    examples_all: dict,
    group_label_array: np.ndarray,
    folds: list[np.ndarray],
    proposal_sources: tuple[str, ...],
    feature_set: str,
    class_weight_mode: str,
    harm_aux_weight: float,
    harm_cost: float,
    threshold: float,
    calibrate: bool,
) -> tuple[dict, dict]:
    subset = slice_examples(examples_all, source_example_mask(examples_all, proposal_sources))
    features, feature_names = combine_feature_blocks(subset, feature_set)
    started = time.perf_counter()
    logits, models, histories = cross_fit(
        features, subset, folds, class_weight_mode, harm_aux_weight
    )
    if calibrate:
        probabilities, fold_temperatures = cross_fitted_temperature(
            logits, subset["outcomes"], subset["question_indices"], folds
        )
    else:
        probabilities = softmax(logits)
        fold_temperatures = [1.0] * len(folds)
    selected, selection = select_revisions(
        state, subset, probabilities, proposal_sources, harm_cost, threshold
    )
    metrics = revision_metrics(state, group_label_array, selected, folds=folds)
    result = {
        "key": key,
        "proposal_sources": list(proposal_sources),
        "feature_set": feature_set,
        "feature_count": len(feature_names),
        "class_weight_mode": class_weight_mode,
        "harm_aux_weight": harm_aux_weight,
        "harm_cost": harm_cost,
        "threshold": threshold,
        "calibrated": calibrate,
        "fold_temperatures": fold_temperatures,
        "metrics": metrics,
        "elapsed_seconds": time.perf_counter() - started,
    }
    artifact = {
        "examples": subset,
        "features": features,
        "feature_names": feature_names,
        "logits": logits,
        "probabilities": probabilities,
        "selected": selected,
        "selection": selection,
        "models": models,
        "histories": histories,
    }
    return result, artifact


def evaluate_fixed_probabilities(
    key: str,
    state: dict,
    artifact: dict,
    group_label_array: np.ndarray,
    folds: list[np.ndarray],
    proposal_sources: tuple[str, ...],
    feature_set: str,
    class_weight_mode: str,
    harm_aux_weight: float,
    harm_cost: float,
    threshold: float,
) -> tuple[dict, dict]:
    selected, selection = select_revisions(
        state, artifact["examples"], artifact["probabilities"], proposal_sources,
        harm_cost, threshold,
    )
    result = {
        "key": key, "proposal_sources": list(proposal_sources),
        "feature_set": feature_set, "feature_count": len(artifact["feature_names"]),
        "class_weight_mode": class_weight_mode, "harm_aux_weight": harm_aux_weight,
        "harm_cost": harm_cost, "threshold": threshold, "calibrated": True,
        "metrics": revision_metrics(state, group_label_array, selected, folds=folds),
    }
    return result, {**artifact, "selected": selected, "selection": selection}


def proposal_coverage(state: dict, labels: np.ndarray) -> dict:
    row = np.arange(len(labels))
    anchor = state["anchor_groups"]
    anchor_correct = labels[row, anchor]
    source_correct = {
        name: labels[row, state["source_groups"][name]] for name in SOURCE_NAMES
    }
    payload = {"anchor_wrong": int((~anchor_correct).sum()), "sources": {}}
    beneficial_sets = {}
    for name in SOURCE_NAMES:
        proposal = state["source_groups"][name]
        beneficial = ~anchor_correct & source_correct[name] & (proposal != anchor)
        harmful = anchor_correct & ~source_correct[name] & (proposal != anchor)
        beneficial_sets[name] = set(np.flatnonzero(beneficial).tolist())
        payload["sources"][name] = {
            "different_from_anchor": int((proposal != anchor).sum()),
            "beneficial_upper_bound": int(beneficial.sum()),
            "harmful_upper_bound": int(harmful.sum()),
            "unique_beneficial": 0,
        }
    for name in SOURCE_NAMES:
        others = set().union(*(beneficial_sets[other] for other in SOURCE_NAMES if other != name))
        payload["sources"][name]["unique_beneficial"] = len(beneficial_sets[name] - others)
    union = set().union(*beneficial_sets.values())
    payload["oracle_correctable"] = len(union)
    payload["oracle_anchor_correct_plus_correctable"] = int(anchor_correct.sum()) + len(union)
    payload["beneficial_overlap"] = {
        left: {right: len(beneficial_sets[left] & beneficial_sets[right]) for right in SOURCE_NAMES}
        for left in SOURCE_NAMES
    }
    return payload


def safety_passes(result: dict, anchor_metrics: dict) -> bool:
    metrics = result["metrics"]
    correct_plurality_drop = (
        anchor_metrics["correct_plurality_accuracy"] - metrics["correct_plurality_accuracy"]
    )
    return bool(
        metrics["all_correct"] >= anchor_metrics["all_correct"]
        and metrics["folds_non_decreasing_vs_anchor"] >= 3
        and metrics["beneficial_revisions"] > metrics["harmful_revisions"]
        and (metrics["revision_precision"] or 0.0) >= 0.65
        and correct_plurality_drop <= 0.75
    )


def paired_bootstrap(anchor_correct: np.ndarray, selected_correct: np.ndarray) -> dict:
    difference = selected_correct.astype(float) - anchor_correct.astype(float)
    rng = np.random.default_rng(SEED)
    estimates = np.empty(10000, dtype=np.float64)
    for start in range(0, 10000, 500):
        indices = rng.integers(0, len(difference), size=(min(500, 10000 - start), len(difference)))
        estimates[start:start + len(indices)] = difference[indices].mean(axis=1) * 100.0
    return {
        "accuracy_difference_points": float(difference.mean() * 100.0),
        "ci95": [float(np.quantile(estimates, 0.025)), float(np.quantile(estimates, 0.975))],
        "replicates": 10000,
    }


def probability_diagnostics(artifact: dict, result: dict) -> dict:
    probabilities = artifact["probabilities"]
    labels = artifact["examples"]["outcomes"]
    p_beneficial = np.clip(probabilities[:, 0], 1e-8, 1 - 1e-8)
    p_harmful = np.clip(probabilities[:, 1], 1e-8, 1 - 1e-8)
    y_beneficial = labels == 0
    y_harmful = labels == 1
    binary_nll = lambda y, p: -float(np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))
    calibration = calibration_metrics(probabilities, labels)
    utility = artifact["selection"]["utility"]
    quantiles = np.quantile(utility, np.linspace(0, 1, 6))
    bins = []
    for index in range(5):
        mask = (utility >= quantiles[index]) & (
            utility <= quantiles[index + 1] if index == 4 else utility < quantiles[index + 1]
        )
        bins.append({
            "n": int(mask.sum()),
            "utility_mean": float(utility[mask].mean()) if mask.any() else None,
            "beneficial_fraction": float(y_beneficial[mask].mean()) if mask.any() else None,
            "harmful_fraction": float(y_harmful[mask].mean()) if mask.any() else None,
        })
    order = np.argsort(-utility)
    error = (~y_beneficial[order]).astype(float)
    risk = np.cumsum(error) / np.arange(1, len(error) + 1)
    return {
        "beneficial_binary_nll": binary_nll(y_beneficial, p_beneficial),
        "harmful_binary_nll": binary_nll(y_harmful, p_harmful),
        **calibration,
        "utility_quintiles": bins,
        "aurc": float(risk.mean()),
        "revision_precision": result["metrics"]["revision_precision"],
        "revision_coverage": result["metrics"]["revision_coverage"],
    }


def main() -> None:
    if RUNS.exists():
        raise FileExistsError("Stage-5 run directory already exists; refusing to overwrite")
    RUNS.mkdir(parents=True)
    started = time.perf_counter()
    initial_hashes = protected_hashes()

    # Train-only construction. Validation is deliberately not loaded here.
    bank_all, arrays = load_inputs("train", DATA)
    references = read_jsonl(DATA / "gsm8k_train.jsonl")
    blocks_all = extract_feature_blocks(bank_all, arrays["embeddings"])
    candidate_labels_all = labels_from_references(bank_all, references)
    hard_indices = np.flatnonzero(hard_group_mask(blocks_all["answers"]))
    bank = [bank_all[int(i)] for i in hard_indices]
    embeddings = arrays["embeddings"][hard_indices]
    blocks = slice_blocks(blocks_all, hard_indices)
    candidate_labels = candidate_labels_all[hard_indices]
    folds = local_folds_from_global(hard_indices)
    with np.load(STAGE1_OOF, allow_pickle=False) as s1, np.load(STAGE2_OOF, allow_pickle=False) as s2:
        if not np.array_equal(s1["indices"], hard_indices) or not np.array_equal(s2["indices"], hard_indices):
            raise ValueError("Frozen OOF indices do not align with train-hard")
        candidate_scores = np.stack([s1["scores"], s2["scores"]], axis=2)
    with np.load(STAGE3_PREDICTIONS, allow_pickle=False) as s3:
        group_scores = s3["train_group_scores"].copy()
        reliability = s3["train_reliability"].copy()
        gate_probability = s3["train_gate_probability"].copy()
        anchor_groups = s3["train_selected_groups"].copy()
    group_data = extract_group_features(bank, embeddings, blocks, candidate_scores)
    state = assemble_state(
        bank, embeddings, blocks, candidate_scores, group_data, group_scores,
        reliability, gate_probability, anchor_groups,
    )
    state["candidate_correct_counts"] = candidate_labels.sum(axis=1)
    group_label_array = group_labels(candidate_labels, group_data["memberships"])
    examples_all = build_residual_examples(state, candidate_labels)
    anchor_metrics = revision_metrics(state, group_label_array, anchor_groups, folds=folds)
    if anchor_metrics["all_correct"] != 1720:
        raise ValueError("Stage-3 OOF anchor does not reproduce 1720/2114")
    coverage = proposal_coverage(state, group_label_array)
    retained_sources = tuple(
        name for name in SOURCE_NAMES
        if coverage["sources"][name]["unique_beneficial"] > 0
    )
    if not retained_sources:
        raise ValueError("Proposal coverage pruning removed every source")
    effective_proposal_sets = {
        "plurality": ("plurality",),
        "stage2": ("stage2",),
        "stage3_group": ("stage3_group",),
        "all_unique": retained_sources,
    }

    protocol = {
        "format": "dase7506-mp2-stage5-prevalidation-v1",
        "seed": SEED,
        "created_before_stage5_validation": True,
        "stage4_available": False,
        "stage4_removed_with_user_approval": True,
        "anchor": "Stage-3 high-validation CREO, strictly frozen",
        "proposal_sets": {key: list(value) for key, value in effective_proposal_sets.items()},
        "proposal_coverage_retained_sources": list(retained_sources),
        "proposal_coverage_rejected_sources": [
            name for name in SOURCE_NAMES if name not in retained_sources
        ],
        "feature_sets": {
            key: {
                "blocks": list(value),
                "names": combine_feature_blocks(examples_all, key)[1],
            }
            for key, value in FEATURE_SETS.items()
        },
        "model_family": "multinomial logistic; optional harm auxiliary; no MLP unless logistic underfits",
        "class_weight_modes": ["uniform", "balanced-capped"],
        "harm_aux_weights": [0.0, 0.1],
        "harm_costs": list(HARM_COSTS),
        "revision_thresholds": list(THRESHOLDS),
        "l2": 0.1,
        "selection_rule": {
            "oof": ["accuracy >= anchor", ">=3/5 folds non-decreasing", "good > harmful",
                    "revision precision >=65%", "correct-plurality drop <=0.75 point"],
            "validation": ["all or hard +2 correct", "beneficial >= harmful",
                           "hard improvement >0", "trap drop <=1", "nontrap drop <=1"],
        },
        "validation_shortlist_limit": 3,
        "protected_artifact_hashes": initial_hashes,
        "test_touched": False,
        "bugfix_record": (
            {
                "original_run": "runs/stage5",
                "reason": [
                    "single-correct reporting used correct answer-group count instead of correct candidate count",
                    "Phase-A implementation failed to prune a source with zero unique beneficial coverage"
                ],
                "model_features_training_and_selection_unchanged": True,
            }
            if RUNS.as_posix() != "runs/stage5" else None
        ),
    }
    dump_json(RUNS / "prevalidation_protocol.json", protocol)
    dump_json(RUNS / "proposal_coverage.json", coverage)

    cache = {}
    phase_a = [{"key": "anchor_only", "metrics": anchor_metrics}]
    for proposal_key, sources in effective_proposal_sets.items():
        key = f"phaseA_{proposal_key}"
        result, artifact = evaluate_oof_config(
            key, state, examples_all, group_label_array, folds, sources,
            "consensus_score", "balanced-capped", 0.0, 2.0, 0.05, False,
        )
        phase_a.append(result)
        cache[(proposal_key, "consensus_score", "balanced-capped", 0.0, False)] = artifact
    phase_a_candidates = [
        row for row in phase_a[1:]
        if all(source in retained_sources for source in row["proposal_sources"])
    ]
    best_phase_a = max(phase_a_candidates, key=lambda row: (
        row["metrics"]["all_correct"], row["metrics"]["revision_precision"] or 0.0,
        len(row["proposal_sources"]),
    ))
    proposal_key = next(
        key for key, value in effective_proposal_sets.items()
        if list(value) == best_phase_a["proposal_sources"]
    )
    proposal_sources = effective_proposal_sets[proposal_key]

    phase_b = []
    phase_b_artifacts = {}
    for feature_set in FEATURE_SETS:
        cache_key = (proposal_key, feature_set, "balanced-capped", 0.0, False)
        if cache_key in cache:
            artifact = cache[cache_key]
            result, artifact = evaluate_fixed_probabilities(
                f"phaseB_{feature_set}", state, artifact, group_label_array, folds,
                proposal_sources, feature_set, "balanced-capped", 0.0, 2.0, 0.05,
            )
        else:
            result, artifact = evaluate_oof_config(
                f"phaseB_{feature_set}", state, examples_all, group_label_array, folds,
                proposal_sources, feature_set, "balanced-capped", 0.0, 2.0, 0.05, False,
            )
        phase_b.append(result)
        phase_b_artifacts[feature_set] = artifact
    best_b_score = max(row["metrics"]["all_accuracy"] for row in phase_b)
    eligible_simple = [
        row for row in phase_b if best_b_score - row["metrics"]["all_accuracy"] < 0.25
    ]
    simplicity = {name: index for index, name in enumerate(FEATURE_SETS)}
    primary_b = min(eligible_simple, key=lambda row: simplicity[row["feature_set"]])
    best_b = max(phase_b, key=lambda row: row["metrics"]["all_correct"])
    top_feature_sets = []
    for name in (primary_b["feature_set"], best_b["feature_set"]):
        if name not in top_feature_sets:
            top_feature_sets.append(name)
    for row in sorted(phase_b, key=lambda item: item["metrics"]["all_correct"], reverse=True):
        if len(top_feature_sets) >= 2:
            break
        if row["feature_set"] not in top_feature_sets:
            top_feature_sets.append(row["feature_set"])

    phase_c = []
    calibrated_artifacts = {}
    for feature_set in top_feature_sets:
        base = phase_b_artifacts[feature_set]
        probabilities, fold_temperatures = cross_fitted_temperature(
            base["logits"], base["examples"]["outcomes"],
            base["examples"]["question_indices"], folds,
        )
        calibrated = {**base, "probabilities": probabilities,
                      "fold_temperatures": fold_temperatures}
        calibrated_artifacts[feature_set] = calibrated
        for harm_cost in HARM_COSTS:
            for threshold in THRESHOLDS:
                key = f"phaseC_{feature_set}_c{harm_cost:g}_t{threshold:g}"
                result, _ = evaluate_fixed_probabilities(
                    key, state, calibrated, group_label_array, folds, proposal_sources,
                    feature_set, "balanced-capped", 0.0, harm_cost, threshold,
                )
                phase_c.append(result)
    best_c = max(phase_c, key=lambda row: (
        row["metrics"]["all_correct"], row["metrics"]["folds_non_decreasing_vs_anchor"],
        row["metrics"]["revision_precision"] or 0.0,
    ))

    phase_d = []
    final_artifacts = {}
    for weight_mode in ("uniform", "balanced-capped"):
        for harm_aux in (0.0, 0.1):
            key = f"phaseD_{best_c['feature_set']}_{weight_mode}_h{harm_aux:g}"
            result, artifact = evaluate_oof_config(
                key, state, examples_all, group_label_array, folds, proposal_sources,
                best_c["feature_set"], weight_mode, harm_aux,
                best_c["harm_cost"], best_c["threshold"], True,
            )
            phase_d.append(result)
            final_artifacts[key] = artifact

    all_final = phase_c + phase_d
    # Phase-C candidates reuse balanced/no-aux calibrated artifacts.
    for result in phase_c:
        artifact = calibrated_artifacts[result["feature_set"]]
        _, selected_artifact = evaluate_fixed_probabilities(
            result["key"], state, artifact, group_label_array, folds, proposal_sources,
            result["feature_set"], result["class_weight_mode"], result["harm_aux_weight"],
            result["harm_cost"], result["threshold"],
        )
        final_artifacts[result["key"]] = selected_artifact
    for result in all_final:
        result["oof_safety_pass"] = safety_passes(result, anchor_metrics)
    ranked = sorted(all_final, key=lambda row: (
        row["oof_safety_pass"], row["metrics"]["all_correct"],
        row["metrics"]["folds_non_decreasing_vs_anchor"],
        row["metrics"]["revision_precision"] or 0.0,
    ), reverse=True)
    shortlist = ranked[:3]

    # The OOF shortlist and all search choices are now frozen. First Stage-5 validation starts here.
    val_bank, val_arrays = load_inputs("validation", DATA)
    val_references = read_jsonl(DATA / "gsm8k_validation.jsonl")
    val_candidate_labels = labels_from_references(val_bank, val_references)
    val_blocks = extract_feature_blocks(val_bank, val_arrays["embeddings"])
    val_hard = hard_group_mask(val_blocks["answers"])
    val_state = build_full_state(val_bank, val_arrays["embeddings"], ANCHOR_CHECKPOINT)
    val_state["candidate_correct_counts"] = val_candidate_labels.sum(axis=1)
    val_group_labels = group_labels(val_candidate_labels, val_state["group_data"]["memberships"])
    with np.load(STAGE3_PREDICTIONS, allow_pickle=False) as stored:
        if not np.array_equal(val_state["anchor_groups"], stored["validation_selected_groups"]):
            raise ValueError("Frozen Stage-3 validation anchor mismatch")
    val_examples_all = build_residual_examples(val_state, val_candidate_labels)
    val_anchor_metrics = revision_metrics(
        val_state, val_group_labels, val_state["anchor_groups"], val_hard
    )
    if val_anchor_metrics["all_correct"] != 755 or val_anchor_metrics["hard_correct"] != 312:
        raise ValueError("Stage-3 validation anchor does not reproduce expected metrics")

    validation_results = []
    validation_artifacts = {}
    all_questions = np.arange(len(bank), dtype=np.int64)
    for rank, oof_result in enumerate(shortlist, 1):
        train_subset = slice_examples(
            examples_all, source_example_mask(examples_all, proposal_sources)
        )
        val_subset = slice_examples(
            val_examples_all, source_example_mask(val_examples_all, proposal_sources)
        )
        train_features, feature_names = combine_feature_blocks(
            train_subset, oof_result["feature_set"]
        )
        val_features, _ = combine_feature_blocks(val_subset, oof_result["feature_set"])
        model = train_multinomial(
            train_features, train_subset["outcomes"], train_subset["question_indices"],
            all_questions, oof_result["class_weight_mode"], oof_result["harm_aux_weight"],
        )
        oof_artifact = final_artifacts[oof_result["key"]]
        temperature = fit_temperature(oof_artifact["logits"], train_subset["outcomes"])
        val_logits = predict_logits(model, val_features)
        val_probabilities = softmax(val_logits / temperature)
        val_selected, val_selection = select_revisions(
            val_state, val_subset, val_probabilities, proposal_sources,
            oof_result["harm_cost"], oof_result["threshold"],
        )
        val_metrics = revision_metrics(
            val_state, val_group_labels, val_selected, val_hard
        )
        row_indices = np.arange(len(val_bank))
        bootstrap = paired_bootstrap(
            val_group_labels[row_indices, val_state["anchor_groups"]],
            val_group_labels[row_indices, val_selected],
        )
        val_result = {
            "rank": rank, "key": oof_result["key"],
            "oof_result": oof_result, "validation_metrics": val_metrics,
            "paired_bootstrap": bootstrap, "temperature": temperature,
        }
        validation_results.append(val_result)
        validation_artifacts[oof_result["key"]] = {
            "model": model, "temperature": temperature, "selected": val_selected,
            "probabilities": val_probabilities, "selection": val_selection,
            "train_subset": train_subset, "val_subset": val_subset,
            "feature_names": feature_names,
        }
        np.savez(
            RUNS / f"validation_shortlist_{rank}.npz",
            selected_groups=val_selected, probabilities=val_probabilities,
            proposal_groups=val_subset["proposal_groups"],
            question_indices=val_subset["question_indices"],
        )

    def validation_passes(item: dict) -> bool:
        metrics = item["validation_metrics"]
        all_gain = metrics["all_correct"] - val_anchor_metrics["all_correct"]
        hard_gain = metrics["hard_correct"] - val_anchor_metrics["hard_correct"]
        return bool(
            (all_gain >= 2 or hard_gain >= 2)
            and metrics["beneficial_revisions"] >= metrics["harmful_revisions"]
            and hard_gain > 0
            and metrics["wrong_plurality_correct"] >= val_anchor_metrics["wrong_plurality_correct"] - 1
            and metrics["correct_plurality_correct"] >= val_anchor_metrics["correct_plurality_correct"] - 1
        )

    for item in validation_results:
        item["validation_improvement_pass"] = validation_passes(item)
        item["successor_eligible"] = bool(
            item["oof_result"]["oof_safety_pass"] and item["validation_improvement_pass"]
        )
    eligible = [item for item in validation_results if item["successor_eligible"]]
    if eligible:
        selected_validation = max(eligible, key=lambda item: (
            item["validation_metrics"]["all_correct"],
            item["validation_metrics"]["hard_correct"],
            item["oof_result"]["metrics"]["all_correct"],
        ))
        selection_status = "stage5_successor_selected"
        apply_residual = True
    else:
        selected_validation = max(validation_results, key=lambda item: (
            item["validation_metrics"]["all_correct"],
            item["validation_metrics"]["hard_correct"],
            item["oof_result"]["metrics"]["all_correct"],
        ))
        selection_status = "no_stage5_model_met_replacement_rule_anchor_retained"
        apply_residual = False

    selected_key = selected_validation["key"]
    selected_oof = selected_validation["oof_result"]
    selected_oof_artifact = final_artifacts[selected_key]
    selected_val_artifact = validation_artifacts[selected_key]
    bundle = RUNS / "provisional_bundle"
    save_stage5_bundle(
        bundle, selected_val_artifact["model"], selected_val_artifact["temperature"],
        {
            "seed": SEED,
            "feature_set": selected_oof["feature_set"],
            "feature_names": selected_val_artifact["feature_names"],
            "proposal_sources": list(proposal_sources),
            "harm_cost": selected_oof["harm_cost"],
            "threshold": selected_oof["threshold"],
            "apply_residual": apply_residual,
            "selection_status": selection_status,
            "anchor_checkpoint": "anchor_stage3",
        },
    )
    shutil.copytree(ANCHOR_CHECKPOINT, bundle / "anchor_stage3")

    final_groups = (
        selected_val_artifact["selected"] if apply_residual else val_state["anchor_groups"]
    )
    validation_choices = groups_to_candidates(val_state, final_groups)
    validation_prediction_path = RUNS / "validation_predictions.json"
    if validation_prediction_path.exists():
        raise FileExistsError("First validation prediction must not be overwritten")
    dump_json(validation_prediction_path, {
        "protocol": PROTOCOL, "split": "validation",
        "ids": [row["id"] for row in val_bank],
        "selected": validation_choices,
        "prediction_seconds": None,
        "source": "stage5_train first frozen validation prediction",
    })

    selected_groups_oof = (
        selected_oof_artifact["selected"] if apply_residual else state["anchor_groups"]
    )
    np.savez(
        RUNS / "oof_predictions.npz",
        hard_indices=hard_indices,
        selected_groups=selected_groups_oof,
        anchor_groups=state["anchor_groups"],
        residual_logits=selected_oof_artifact["logits"],
        residual_probabilities=selected_oof_artifact["probabilities"],
        proposal_groups=selected_oof_artifact["examples"]["proposal_groups"],
        question_indices=selected_oof_artifact["examples"]["question_indices"],
    )
    dump_json(RUNS / "fold_results.json", selected_oof["metrics"]["folds"])
    paired_audit = {
        "selection_status": selection_status,
        "anchor_oof": anchor_metrics,
        "selected_oof": selected_oof["metrics"],
        "anchor_validation": val_anchor_metrics,
        "selected_validation": selected_validation["validation_metrics"],
        "paired_bootstrap": selected_validation["paired_bootstrap"],
    }
    dump_json(RUNS / "paired_revision_audit.json", paired_audit)
    calibration = probability_diagnostics(selected_oof_artifact, selected_oof)
    calibration["uncalibrated_multiclass_nll"] = multiclass_nll(
        selected_oof_artifact["logits"], selected_oof_artifact["examples"]["outcomes"]
    )
    dump_json(RUNS / "calibration.json", calibration)

    final_hashes = protected_hashes()
    integrity = {
        "unchanged": initial_hashes == final_hashes,
        "initial": initial_hashes,
        "final": final_hashes,
        "test_artifacts_accessed": False,
    }
    dump_json(RUNS / "protected_artifact_integrity.json", integrity)
    if not integrity["unchanged"]:
        raise RuntimeError("A protected Stage 1-3 artifact changed")

    summary = {
        "seed": SEED,
        "stage4_removed_with_user_approval": True,
        "test_touched": False,
        "bugfix_record": protocol["bugfix_record"],
        "elapsed_seconds": time.perf_counter() - started,
        "anchor_oof": anchor_metrics,
        "proposal_coverage": coverage,
        "phase_a": phase_a,
        "phase_a_selected_proposal_set": proposal_key,
        "phase_b": phase_b,
        "phase_b_top_feature_sets": top_feature_sets,
        "phase_c": phase_c,
        "phase_d": phase_d,
        "oof_shortlist": shortlist,
        "validation_results": validation_results,
        "selection_status": selection_status,
        "apply_residual": apply_residual,
        "selected_key": selected_key,
        "bundle": str(bundle),
    }
    dump_json(RUNS / "experiment_summary.json", summary)
    dump_json(RUNS / "search_state.json", {
        "status": "paused_before_test", "test_touched": False,
        "elapsed_seconds": time.perf_counter() - started,
        "selection_status": selection_status,
        "bundle": str(bundle),
    })
    print(json.dumps({
        "selection_status": selection_status,
        "apply_residual": apply_residual,
        "selected_key": selected_key,
        "oof": selected_oof["metrics"],
        "validation": selected_validation["validation_metrics"],
        "bundle": str(bundle),
        "elapsed_seconds": time.perf_counter() - started,
    }, indent=2))


if __name__ == "__main__":
    main()
