"""Post-selection train-OOF reporting audit for canonical CREO-RR."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from common import DATA, load_inputs, read_jsonl
from creo_features import extract_group_features, group_labels
from run_stage3 import slice_blocks
from selector_features import extract_feature_blocks, hard_group_mask, labels_from_references
from stage5_features import (
    assemble_state, build_residual_examples, select_revisions, source_example_mask
)
from stage5_train import revision_metrics, slice_examples


RUN = Path("runs/stage5/bugfix2")


def main() -> None:
    summary = json.loads((RUN / "experiment_summary.json").read_text(encoding="utf-8"))
    protocol = json.loads((RUN / "prevalidation_protocol.json").read_text(encoding="utf-8"))
    bank_all, arrays = load_inputs("train", DATA)
    references = read_jsonl(DATA / "gsm8k_train.jsonl")
    blocks_all = extract_feature_blocks(bank_all, arrays["embeddings"])
    labels_all = labels_from_references(bank_all, references)
    hard_indices = np.flatnonzero(hard_group_mask(blocks_all["answers"]))
    bank = [bank_all[int(i)] for i in hard_indices]
    embeddings = arrays["embeddings"][hard_indices]
    blocks = slice_blocks(blocks_all, hard_indices)
    candidate_labels = labels_all[hard_indices]
    with np.load("runs/cv/hand_centered__h0.01__e1__listwise/oof_predictions.npz", allow_pickle=False) as s1, np.load(
        "runs/stage2/cv/hand_centered__uniform__pair0.1/oof_predictions.npz", allow_pickle=False
    ) as s2, np.load("runs/stage3/selected_predictions.npz", allow_pickle=False) as s3:
        candidate_scores = np.stack([s1["scores"], s2["scores"]], axis=2)
        group_scores = s3["train_group_scores"].copy()
        reliability = s3["train_reliability"].copy()
        gate_probability = s3["train_gate_probability"].copy()
        anchor_groups = s3["train_selected_groups"].copy()
    groups = extract_group_features(bank, embeddings, blocks, candidate_scores)
    state = assemble_state(
        bank, embeddings, blocks, candidate_scores, groups, group_scores,
        reliability, gate_probability, anchor_groups,
    )
    state["candidate_correct_counts"] = candidate_labels.sum(axis=1)
    labels = group_labels(candidate_labels, groups["memberships"])
    examples_all = build_residual_examples(state, candidate_labels)
    sources = tuple(protocol["proposal_coverage_retained_sources"])
    examples = slice_examples(examples_all, source_example_mask(examples_all, sources))
    with np.load(RUN / "oof_predictions.npz", allow_pickle=False) as data:
        probabilities = data["residual_probabilities"].copy()
        if not np.array_equal(data["proposal_groups"], examples["proposal_groups"]):
            raise ValueError("Reporting proposal rows do not match canonical OOF artifact")
        if not np.array_equal(data["question_indices"], examples["question_indices"]):
            raise ValueError("Reporting question rows do not match canonical OOF artifact")
    selected_validation = next(
        item for item in summary["validation_results"] if item["key"] == summary["selected_key"]
    )
    config = selected_validation["oof_result"]
    selected, selection = select_revisions(
        state, examples, probabilities, sources,
        config["harm_cost"], config["threshold"],
    )
    selected_metrics = revision_metrics(state, labels, selected)

    utility = selection["utility"]
    reporting_thresholds = (-0.10, -0.05, 0.0, 0.025, 0.05, 0.075, 0.10, 0.15, 0.20, 0.30)
    curve = []
    for threshold in reporting_thresholds:
        groups_selected, _ = select_revisions(
            state, examples, probabilities, sources, config["harm_cost"], threshold
        )
        metrics = revision_metrics(state, labels, groups_selected)
        curve.append({
            "threshold": threshold,
            "correct": metrics["all_correct"],
            "accuracy": metrics["all_accuracy"],
            "revisions": metrics["revisions"],
            "beneficial": metrics["beneficial_revisions"],
            "harmful": metrics["harmful_revisions"],
            "neutral": metrics["neutral_revisions"],
            "precision": metrics["revision_precision"],
            "coverage": metrics["revision_coverage"],
        })
    class_counts = {
        "beneficial": int(np.sum(examples["outcomes"] == 0)),
        "harmful": int(np.sum(examples["outcomes"] == 1)),
        "neutral": int(np.sum(examples["outcomes"] == 2)),
    }
    payload = {
        "scope": "train-hard strict OOF reporting only",
        "proposal_sources": list(sources),
        "residual_example_count": len(examples["outcomes"]),
        "residual_class_counts": class_counts,
        "selected_oof_configuration": {
            "key": config["key"], "harm_cost": config["harm_cost"],
            "threshold": config["threshold"], "feature_set": config["feature_set"],
        },
        "selected_oof_metrics_reconstructed": selected_metrics,
        "utility_min": float(utility.min()),
        "utility_max": float(utility.max()),
        "precision_coverage_curve": curve,
        "curve_is_diagnostic_not_model_selection": True,
        "test_touched": False
    }
    (RUN / "reporting_audit.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
