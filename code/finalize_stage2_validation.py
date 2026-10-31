"""Post-hoc diagnostics for the single frozen stage-2 validation prediction.

No choices are generated here. The script reads the official frozen prediction,
computes strata, and fits calibration using train OOF scores only.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import time

import numpy as np

from common import DATA, load_inputs, read_jsonl
from run_experiments import calibration_metrics, fit_temperature, masked_probabilities, binary_nll
from selector_features import extract_feature_blocks, hard_group_mask, labels_from_references, plurality_choice
from selector_model import cluster_choices_from_scores, load_model, score_model


STAGE2 = Path("runs/stage2")


def dump_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def accuracy(labels: np.ndarray, choices: np.ndarray) -> float:
    return 100.0 * float(np.mean(labels[np.arange(len(labels)), choices]))


def strata(labels: np.ndarray, choices: np.ndarray, answers: list[list[str | None]]) -> dict:
    distinct = np.asarray([len({a for a in row if a is not None}) for row in answers])
    correct_count = labels.sum(axis=1)
    output = {}
    for dimension, values in (
        ("distinct_valid_answers", distinct),
        ("correct_candidate_count", correct_count),
    ):
        output[dimension] = {}
        for value in sorted(set(int(item) for item in values)):
            mask = values == value
            output[dimension][str(value)] = {
                "n": int(mask.sum()),
                "accuracy": accuracy(labels[mask], choices[mask]),
            }
    return output


def main() -> None:
    state_path = STAGE2 / "search_state.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    if state.get("status") != "winner_fixed_pre_validation" or state.get("validation_runs") != 0:
        raise RuntimeError("Expected exactly one frozen validation run after winner selection")

    prediction_path = STAGE2 / "validation_predictions.jsonl"
    score_path = STAGE2 / "validation_score.json"
    prediction = json.loads(prediction_path.read_text(encoding="utf-8"))
    official_score = json.loads(score_path.read_text(encoding="utf-8"))
    choices = np.asarray(prediction["selected"], dtype=int)

    val_bank, val_arrays = load_inputs("validation", DATA)
    val_refs = read_jsonl(DATA / "gsm8k_validation.jsonl")
    if prediction["ids"] != [row["id"] for row in val_bank]:
        raise ValueError("Frozen prediction IDs do not align with validation inputs")
    val_blocks = extract_feature_blocks(val_bank, val_arrays["embeddings"])
    val_labels = labels_from_references(val_bank, val_refs)
    val_hard = hard_group_mask(val_blocks["answers"])

    model = load_model(STAGE2 / "provisional_model_bundle")
    val_scores = score_model(model, val_blocks)
    reconstructed = np.asarray(cluster_choices_from_scores(
        val_bank,
        val_scores,
        val_blocks["valid"],
        rule=model.get("decision_rule", "candidate"),
        alpha=float(model.get("cluster_alpha", 0.0)),
    ), dtype=int)
    if not np.array_equal(reconstructed, choices):
        raise ValueError("Frozen official choices differ from checkpoint reconstruction")

    winner = json.loads((STAGE2 / "winner_pre_validation.json").read_text(encoding="utf-8"))
    oof_path = STAGE2 / "cv" / winner["key"] / "oof_predictions.npz"
    with np.load(oof_path, allow_pickle=False) as data:
        train_indices = data["indices"].astype(int)
        oof_scores = data["scores"].astype(np.float64)
        oof_choices = data["choices"].astype(int)
    train_bank, train_arrays = load_inputs("train", DATA)
    train_refs = read_jsonl(DATA / "gsm8k_train.jsonl")
    train_blocks = extract_feature_blocks(train_bank, train_arrays["embeddings"])
    train_labels = labels_from_references(train_bank, train_refs)
    temperature, fitted_oof_nll = fit_temperature(
        oof_scores,
        train_blocks["valid"][train_indices],
        train_labels[train_indices],
        oof_choices,
    )
    plurality = np.asarray([plurality_choice(row) for row in val_bank], dtype=int)
    calibrated = calibration_metrics(
        val_scores,
        val_blocks["valid"],
        val_labels,
        choices,
        val_blocks["answers"],
        plurality,
        temperature,
    )
    uncalibrated = calibration_metrics(
        val_scores,
        val_blocks["valid"],
        val_labels,
        choices,
        val_blocks["answers"],
        plurality,
        1.0,
    )
    calibration = {
        "method": "temperature fitted on train OOF only",
        "temperature": temperature,
        "oof_fitted_binary_nll": fitted_oof_nll,
        "validation_uncalibrated": uncalibrated,
        "validation_oof_temperature": calibrated,
        "validation_nll_improved": calibrated["nll"] < uncalibrated["nll"],
        "warning": "Softmax confidence is a score-derived confidence, not a probability of correctness.",
    }
    dump_json(STAGE2 / "validation_uncertainty.json", calibration)

    internal = {
        "official": official_score,
        "validation_all": accuracy(val_labels, choices),
        "validation_hard": accuracy(val_labels[val_hard], choices[val_hard]),
        "n_all": len(val_labels),
        "n_hard": int(val_hard.sum()),
        "strata": strata(val_labels, choices, val_blocks["answers"]),
        "prediction_sha256": hashlib.sha256(prediction_path.read_bytes()).hexdigest(),
        "checkpoint_reconstruction_identical": True,
        "post_validation_tuning": False,
    }
    if abs(internal["validation_all"] - float(official_score["accuracy"])) > 1e-12:
        raise ValueError("Internal and official validation accuracy differ")
    dump_json(STAGE2 / "validation_internal.json", internal)

    summary_path = STAGE2 / "experiment_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["validation_not_yet_run"] = False
    summary["validation"] = internal
    summary["calibration"] = {
        "temperature": temperature,
        "validation_uncalibrated_nll": uncalibrated["nll"],
        "validation_calibrated_nll": calibrated["nll"],
        "validation_uncalibrated_ece": uncalibrated["ece_5bin"],
        "validation_calibrated_ece": calibrated["ece_5bin"],
        "validation_nll_improved": calibration["validation_nll_improved"],
    }
    dump_json(summary_path, summary)

    state.update({
        "status": "validated_engineering_checks_pending",
        "validation_runs": 1,
        "validation_completed_unix": time.time(),
        "validation_prediction_seconds": official_score.get("prediction_seconds"),
        "validation_scoring_seconds": official_score.get("scoring_seconds"),
        "test_touched": False,
    })
    dump_json(state_path, state)
    print(json.dumps({"validation": internal, "calibration": calibration}, indent=2))


if __name__ == "__main__":
    main()
