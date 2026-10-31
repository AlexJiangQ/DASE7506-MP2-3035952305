"""Targeted CREO experiment: gate a frozen comparator against plurality."""
from __future__ import annotations

import json
from pathlib import Path
import shutil

import numpy as np

from common import DATA, load_inputs, read_jsonl
from creo_features import extract_group_features, gate_feature_matrices, group_labels, plurality_group_indices
from creo_model import (
    fit_logistic,
    gate_group_choices,
    load_creo_bundle,
    logistic_logits,
    logistic_probabilities,
    save_creo_bundle,
    score_group_model,
)
from run_stage3 import (
    binary_nll,
    candidate_choices_to_groups,
    cross_fitted_logistic,
    decision_metrics,
    local_folds_from_global,
    masked_candidate_argmax,
    slice_blocks,
)
from selector_features import extract_feature_blocks, hard_group_mask, labels_from_references
from selector_model import load_model, score_model


ROOT = Path("runs/stage3")
SOURCE_BUNDLE = ROOT / "provisional_creo_bundle"
OUTPUT = ROOT / "direct_challenger"


def dump_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    bank_all, train_arrays = load_inputs("train", DATA)
    train_refs = read_jsonl(DATA / "gsm8k_train.jsonl")
    blocks_all = extract_feature_blocks(bank_all, train_arrays["embeddings"])
    labels_all = labels_from_references(bank_all, train_refs)
    hard_indices = np.flatnonzero(hard_group_mask(blocks_all["answers"]))
    train_bank = [bank_all[int(i)] for i in hard_indices]
    train_blocks = slice_blocks(blocks_all, hard_indices)
    train_labels = labels_all[hard_indices]
    folds = local_folds_from_global(hard_indices)

    with np.load("runs/cv/hand_centered__h0.01__e1__listwise/oof_predictions.npz", allow_pickle=False) as s1, np.load(
        "runs/stage2/cv/hand_centered__uniform__pair0.1/oof_predictions.npz", allow_pickle=False
    ) as s2, np.load(ROOT / "selected_predictions.npz", allow_pickle=False) as selected:
        train_candidate_scores = np.stack([s1["scores"], s2["scores"]], axis=2)
        group_oof_scores = selected["train_group_scores"].astype(np.float64)
    train_groups = extract_group_features(
        train_bank, train_arrays["embeddings"][hard_indices], train_blocks, train_candidate_scores
    )
    train_group_labels = group_labels(train_labels, train_groups["memberships"])

    val_bank, val_arrays = load_inputs("validation", DATA)
    val_refs = read_jsonl(DATA / "gsm8k_validation.jsonl")
    val_blocks = extract_feature_blocks(val_bank, val_arrays["embeddings"])
    val_labels = labels_from_references(val_bank, val_refs)
    val_hard = hard_group_mask(val_blocks["answers"])
    base_models = [
        load_model(SOURCE_BUNDLE / "base_stage1"),
        load_model(SOURCE_BUNDLE / "base_stage2"),
    ]
    val_candidate_scores = np.stack(
        [score_model(model, val_blocks) for model in base_models], axis=2
    )
    val_groups = extract_group_features(
        val_bank, val_arrays["embeddings"], val_blocks, val_candidate_scores
    )
    val_group_labels = group_labels(val_labels, val_groups["memberships"])
    source_bundle = load_creo_bundle(SOURCE_BUNDLE)
    group_model = source_bundle["group_model"]
    group_val_scores = score_group_model(
        group_model, val_groups["features"], val_groups["valid"]
    )
    train_plurality = plurality_group_indices(train_bank, train_groups)
    val_plurality = plurality_group_indices(val_bank, val_groups)

    stage2_train_candidates = masked_candidate_argmax(
        train_candidate_scores[:, :, 1], train_blocks["valid"]
    )
    stage2_train_groups = candidate_choices_to_groups(stage2_train_candidates, train_groups)
    stage2_correct = train_group_labels[np.arange(len(train_bank)), stage2_train_groups]
    stage2_val_candidates = masked_candidate_argmax(
        val_candidate_scores[:, :, 1], val_blocks["valid"]
    )
    stage2_val_groups = candidate_choices_to_groups(stage2_val_candidates, val_groups)
    stage2_val_metrics = decision_metrics(
        val_group_labels, stage2_val_groups, val_plurality, val_hard
    )

    results = []
    artifacts = {}
    reliability_rows = []
    for source_index, source_name in ((0, "stage1"), (1, "stage2")):
        train_candidates = masked_candidate_argmax(
            train_candidate_scores[:, :, source_index], train_blocks["valid"]
        )
        val_candidates = masked_candidate_argmax(
            val_candidate_scores[:, :, source_index], val_blocks["valid"]
        )
        train_challenger = candidate_choices_to_groups(train_candidates, train_groups)
        val_challenger = candidate_choices_to_groups(val_candidates, val_groups)
        train_base = gate_feature_matrices(
            train_bank, train_groups, group_oof_scores, train_group_labels,
            challenger_indices=train_challenger,
        )
        val_base = gate_feature_matrices(
            val_bank, val_groups, group_val_scores, val_group_labels,
            challenger_indices=val_challenger,
        )

        reliability_artifacts = {}
        for reliability_l2 in (0.01, 0.1, 1.0):
            q_oof, _ = cross_fitted_logistic(
                train_base["features"], train_base["plurality_correct"].astype(float),
                folds, reliability_l2,
            )
            reliability_model = fit_logistic(
                train_base["features"], train_base["plurality_correct"].astype(float),
                np.ones(len(train_bank), dtype=bool), reliability_l2,
            )
            q_val = logistic_probabilities(reliability_model, val_base["features"])
            row = {
                "source": source_name,
                "l2": reliability_l2,
                "oof_nll": binary_nll(train_base["plurality_correct"], q_oof),
                "validation_hard_nll": binary_nll(
                    val_base["plurality_correct"][val_hard], q_val[val_hard]
                ),
            }
            reliability_rows.append(row)
            reliability_artifacts[reliability_l2] = (q_oof, q_val, reliability_model)
        source_reliability = min(
            [row for row in reliability_rows if row["source"] == source_name],
            key=lambda row: row["validation_hard_nll"],
        )
        q_oof, q_val, reliability_model = reliability_artifacts[source_reliability["l2"]]
        q_oof_logit = np.log(np.clip(q_oof, 1e-8, 1 - 1e-8) / np.clip(1 - q_oof, 1e-8, 1))
        q_val_logit = np.log(np.clip(q_val, 1e-8, 1 - 1e-8) / np.clip(1 - q_val, 1e-8, 1))
        train_features = np.column_stack([train_base["features"], q_oof, q_oof_logit])
        val_features = np.column_stack([val_base["features"], q_val, q_val_logit])

        for gate_l2 in (0.01, 0.1, 1.0):
            for balanced in (False, True):
                gate_oof, _ = cross_fitted_logistic(
                    train_features, train_base["overturn_label"], folds, gate_l2,
                    eligible=train_base["informative"], class_balance=balanced,
                )
                gate_model = fit_logistic(
                    train_features, train_base["overturn_label"], train_base["informative"],
                    gate_l2, class_balance=balanced,
                )
                gate_val = logistic_probabilities(gate_model, val_features)
                for threshold in (0.3, 0.4, 0.5, 0.6, 0.7, 0.8):
                    train_selected = gate_group_choices(
                        train_base["plurality"], train_base["challenger"], gate_oof, threshold
                    )
                    val_selected = gate_group_choices(
                        val_base["plurality"], val_base["challenger"], gate_val, threshold
                    )
                    train_correct = train_group_labels[np.arange(len(train_bank)), train_selected]
                    fold_deltas = [
                        int(train_correct[idx].sum() - stage2_correct[idx].sum()) for idx in folds
                    ]
                    key = f"{source_name}_l2{gate_l2:g}_b{int(balanced)}_t{threshold:g}"
                    row = {
                        "key": key,
                        "challenger_source": source_name,
                        "gate_l2": gate_l2,
                        "class_balance": balanced,
                        "threshold": threshold,
                        "reliability_l2": source_reliability["l2"],
                        "train_metrics": decision_metrics(
                            train_group_labels, train_selected, train_plurality
                        ),
                        "validation_metrics": decision_metrics(
                            val_group_labels, val_selected, val_plurality, val_hard
                        ),
                        "oof_delta_vs_stage2": int(train_correct.sum() - stage2_correct.sum()),
                        "fold_deltas_vs_stage2": fold_deltas,
                        "folds_non_decreasing": int(sum(delta >= 0 for delta in fold_deltas)),
                    }
                    results.append(row)
                    artifacts[key] = (reliability_model, gate_model)

    robust = [
        row for row in results
        if row["oof_delta_vs_stage2"] >= 0
        and row["folds_non_decreasing"] >= 4
        and row["validation_metrics"]["hard_correct"] >= stage2_val_metrics["hard_correct"] + 2
        and row["validation_metrics"]["good_overturns"] >= row["validation_metrics"]["bad_overturns"]
    ]
    pool = robust if robust else results
    selected = max(pool, key=lambda row: (
        row["validation_metrics"]["hard_correct"],
        row["oof_delta_vs_stage2"],
        row["folds_non_decreasing"],
        row["validation_metrics"]["trap_correct"],
    ))
    reliability_model, gate_model = artifacts[selected["key"]]
    group_model = {**group_model, "loss_history": []}
    config = {
        "seed": 20261022,
        "threshold": selected["threshold"],
        "challenger_source": selected["challenger_source"],
        "selection_status": "robust_direct_gate" if robust else "no_direct_gate_met_robust_rule",
        "base_checkpoints": ["base_stage1", "base_stage2"],
        "selected_direct_gate": selected,
    }
    bundle_dir = ROOT / "provisional_creo_direct_bundle"
    save_creo_bundle(bundle_dir, group_model, reliability_model, gate_model, config)
    for name in ("base_stage1", "base_stage2"):
        destination = bundle_dir / name
        if not destination.exists():
            shutil.copytree(SOURCE_BUNDLE / name, destination)

    stability_pool = [
        row for row in results
        if row["oof_delta_vs_stage2"] >= 0 and row["folds_non_decreasing"] >= 4
    ]
    stability_selected = max(stability_pool, key=lambda row: (
        row["validation_metrics"]["hard_correct"],
        row["oof_delta_vs_stage2"],
        row["validation_metrics"]["trap_correct"],
    ))
    stable_reliability, stable_gate = artifacts[stability_selected["key"]]
    stable_config = {
        "seed": 20261022,
        "threshold": stability_selected["threshold"],
        "challenger_source": stability_selected["challenger_source"],
        "selection_status": "oof_stable_direct_gate_candidate",
        "base_checkpoints": ["base_stage1", "base_stage2"],
        "selected_direct_gate": stability_selected,
    }
    stable_bundle_dir = ROOT / "provisional_creo_stable_bundle"
    save_creo_bundle(
        stable_bundle_dir, group_model, stable_reliability, stable_gate, stable_config
    )
    for name in ("base_stage1", "base_stage2"):
        destination = stable_bundle_dir / name
        if not destination.exists():
            shutil.copytree(SOURCE_BUNDLE / name, destination)
    payload = {
        "scope": "train-hard OOF and validation only; no test data used",
        "stage2_validation": stage2_val_metrics,
        "reliability_results": reliability_rows,
        "direct_gate_results": results,
        "robust_rule": "OOF delta >= 0; >=4/5 folds non-decreasing; validation-hard +2 correct; good >= bad",
        "robust_count": len(robust),
        "selected": selected,
        "bundle": str(bundle_dir),
        "stability_selected": stability_selected,
        "stable_bundle": str(stable_bundle_dir),
    }
    dump_json(OUTPUT / "experiment_summary.json", payload)
    print(json.dumps({
        "robust_count": len(robust),
        "selected": selected,
        "bundle": str(bundle_dir),
        "stability_selected": stability_selected,
        "stable_bundle": str(stable_bundle_dir),
    }, indent=2))


if __name__ == "__main__":
    main()
