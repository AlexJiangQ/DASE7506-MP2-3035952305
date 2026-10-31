"""Recompute fixed-model uncertainty metrics without model selection."""
import json
from pathlib import Path

import numpy as np

from common import DATA, load_inputs, read_jsonl
from run_experiments import calibration_metrics, fit_temperature
from selector_features import extract_feature_blocks, hard_group_mask, labels_from_references, plurality_choice
from selector_model import choices_from_scores, load_model, score_model


RUNS = Path("runs")


def main():
    summary = json.loads((RUNS / "experiment_summary.json").read_text(encoding="utf-8"))
    config = summary["provisional_best"]
    train_bank, train_arrays = load_inputs("train", DATA)
    train_refs = read_jsonl(DATA / "gsm8k_train.jsonl")
    train_blocks = extract_feature_blocks(train_bank, train_arrays["embeddings"])
    train_labels = labels_from_references(train_bank, train_refs)
    hard_indices = np.flatnonzero(hard_group_mask(train_blocks["answers"]))
    with np.load(RUNS / "cv" / config["key"] / "oof_predictions.npz", allow_pickle=False) as data:
        oof_indices = data["indices"].astype(int)
        oof_scores = data["scores"].astype(np.float64)
        oof_choices = data["choices"].astype(int)
    if not np.array_equal(oof_indices, hard_indices):
        raise ValueError("OOF alignment mismatch")
    temperature, oof_nll = fit_temperature(
        oof_scores, train_blocks["valid"][oof_indices], train_labels[oof_indices], oof_choices
    )

    val_bank, val_arrays = load_inputs("validation", DATA)
    val_refs = read_jsonl(DATA / "gsm8k_validation.jsonl")
    val_blocks = extract_feature_blocks(val_bank, val_arrays["embeddings"])
    val_labels = labels_from_references(val_bank, val_refs)
    model = load_model(RUNS / "provisional_model_bundle")
    scores = score_model(model, val_blocks)
    choices = choices_from_scores(val_bank, scores, val_blocks["valid"])
    plurality = [plurality_choice(row) for row in val_bank]
    metrics = calibration_metrics(
        scores, val_blocks["valid"], val_labels, choices,
        val_blocks["answers"], plurality, temperature,
    )
    metrics["oof_temperature_fit_nll"] = oof_nll
    metrics["note"] = "Calibration is descriptive; it does not alter predictions or model selection."
    (RUNS / "validation_uncertainty.json").write_text(
        json.dumps(metrics, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()

