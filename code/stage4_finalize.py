"""Finalize the already-evaluated OOF research leader without rerunning validation."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from common import DATA, load_inputs, read_jsonl
from selector_features import extract_feature_blocks, hard_group_mask, labels_from_references
from stage4_features import (
    build_answer_groups,
    candidate_choices_to_groups,
    consensus_features,
    extract_group_features,
    group_labels,
    merge_branch_probabilities,
)
from stage4_model import calibration_metrics, load_bundle, save_bundle
from stage4_train import RUNS, crossfit_binary, extended_folds


def main() -> None:
    summary_path = RUNS / "experiment_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    research_leader = max(summary["method_results"], key=lambda row: row["cv_mean"])
    if research_leader["name"] != "fixed_reliability":
        raise ValueError("Unexpected Stage 4 research leader")

    bank, arrays = load_inputs("train", DATA)
    blocks = extract_feature_blocks(bank, arrays["embeddings"])
    labels = labels_from_references(bank, read_jsonl(DATA / "gsm8k_train.jsonl"))
    hard = hard_group_mask(blocks["answers"])
    folds = extended_folds(np.flatnonzero(hard), np.flatnonzero(~hard))
    groups = build_answer_groups(bank, blocks["answers"])
    group_label_array = group_labels(labels, groups)
    with np.load(RUNS / "oof_predictions.npz", allow_pickle=False) as data:
        stored = {key: data[key].copy() for key in data.files}
    candidate_scores = stored["candidate_scores"]
    group_scores = stored["group_scores"]
    plurality = stored["plurality_groups"].astype(int)
    candidate_groups = stored["candidate_groups"].astype(int)
    learned_groups = stored["group_groups"].astype(int)
    reliabilities = stored["reliability"]
    projection_dim = int(summary["best_candidate"]["config"]["projection_dim"])
    neutral_projected = np.zeros((len(bank), 4, projection_dim), dtype=float)
    group_features = extract_group_features(
        bank, groups, neutral_projected, candidate_scores
    )
    base_features, _ = consensus_features(
        bank, groups, group_features, plurality, candidate_groups, learned_groups,
        candidate_scores, group_scores, hard,
    )
    gate_features = np.column_stack([base_features, reliabilities])
    branch_groups = np.column_stack([plurality, candidate_groups, learned_groups])
    selected_groups = merge_branch_probabilities(branch_groups, reliabilities)
    selected_correct = group_label_array[np.arange(len(bank)), selected_groups]
    selected_probability, _, selected_calibrator = crossfit_binary(
        gate_features, selected_correct, folds, hard, l2=0.1
    )

    calibration = json.loads((RUNS / "calibration.json").read_text(encoding="utf-8"))
    calibration["selected_method"] = "fixed_reliability"
    calibration["all"] = calibration_metrics(selected_correct, selected_probability)
    calibration["hard"] = calibration_metrics(selected_correct[hard], selected_probability[hard])
    (RUNS / "calibration.json").write_text(json.dumps(calibration, indent=2) + "\n", encoding="utf-8")

    bundle_path = RUNS / "provisional_bundle"
    bundle = load_bundle(bundle_path)
    bundle["selected_calibrator"] = selected_calibrator
    config = {
        key: value for key, value in bundle["config"].items()
        if key not in {"format", "model_metadata"}
    }
    config.update({
        "selected_method": "fixed_reliability",
        "replacement_pass": False,
        "formal_replacement_decision": "retain_frozen_pre_stage4_model",
        "provisional_bundle_role": "Stage 4 research leader only",
        "test_touched": False,
    })
    save_bundle(bundle_path, bundle, config)
    bundle_bytes = sum(
        path.stat().st_size for path in bundle_path.rglob("*") if path.is_file()
    )

    np.savez(
        RUNS / "oof_predictions.npz",
        **{key: value for key, value in stored.items() if key not in {
            "selected_groups", "selected_correct", "selected_probability"
        }},
        selected_groups=selected_groups,
        selected_correct=selected_correct,
        selected_probability=selected_probability,
    )
    summary["selected_method"] = research_leader
    summary["simple_stage4_leader"] = next(
        row for row in summary["method_results"] if row["name"] == "candidate"
    )
    summary["eligible_replacement_methods"] = []
    summary["replacement_pass"] = False
    summary["formal_replacement_decision"] = "retain_frozen_pre_stage4_model"
    summary["finalization_note"] = (
        "Saved the raw OOF research leader for reproducibility; it is not eligible "
        "to replace the stronger frozen pre-Stage-4 model. Validation was not rerun."
    )
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    selected_folds = research_leader["folds"]
    fold_results = {
        "stage2": summary["stage2_baseline"]["folds"],
        "stable_direct_fold_correct": [351, 340, 359, 354, 332],
        "selected": selected_folds,
    }
    (RUNS / "fold_results.json").write_text(
        json.dumps(fold_results, indent=2) + "\n", encoding="utf-8"
    )
    hard_indices = np.flatnonzero(hard)
    with np.load("runs/stage2/cv/hand_centered__uniform__pair0.1/oof_predictions.npz", allow_pickle=False) as stage2:
        stage2_choices = stage2["choices"].astype(int)
    stage2_groups = candidate_choices_to_groups(stage2_choices, {
        "candidate_to_group": groups["candidate_to_group"][hard_indices]
    })
    stage2_correct = group_label_array[hard_indices, stage2_groups]
    fixed_correct = selected_correct[hard_indices]
    plurality_correct = group_label_array[hard_indices, plurality[hard_indices]]
    single = labels[hard_indices].sum(axis=1) == 1

    def counts(mask):
        first = stage2_correct[mask]; second = fixed_correct[mask]
        return {
            "n": int(mask.sum()),
            "both_correct": int(np.sum(first & second)),
            "stage2_only": int(np.sum(first & ~second)),
            "stage4_only": int(np.sum(~first & second)),
            "both_wrong": int(np.sum(~first & ~second)),
            "delta_correct": int(second.sum() - first.sum()),
        }

    paired = {
        "scope": "train-hard OOF only; no test data used",
        "selected_method": "fixed_reliability",
        "selected_vs_stage2_fold_deltas": [
            selected_folds[i]["correct"] - [350, 335, 354, 353, 334][i] for i in range(5)
        ],
        "selected_vs_stable_direct_fold_deltas": [
            selected_folds[i]["correct"] - [351, 340, 359, 354, 332][i] for i in range(5)
        ],
        "strata": {
            "all_hard": counts(np.ones(len(hard_indices), dtype=bool)),
            "wrong_plurality_traps": counts(~plurality_correct),
            "correct_plurality": counts(plurality_correct),
            "single_correct_candidate": counts(single),
        },
        "selected_metrics": research_leader,
        "test_touched": False,
    }
    (RUNS / "paired_audit.json").write_text(
        json.dumps(paired, indent=2) + "\n", encoding="utf-8"
    )
    state_path = RUNS / "search_state.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    state.update({
        "selected_method": "fixed_reliability",
        "replacement_pass": False,
        "formal_replacement_decision": "retain_frozen_pre_stage4_model",
        "asset_bytes": bundle_bytes,
        "test_touched": False,
    })
    state_path.write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "selected_method": "fixed_reliability",
        "oof_correct": int(selected_correct[hard].sum()),
        "oof_n": int(hard.sum()),
        "replacement_pass": False,
        "validation_reused": False,
        "test_touched": False,
    }, indent=2))


if __name__ == "__main__":
    main()
