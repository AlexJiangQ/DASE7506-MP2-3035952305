"""Reconstruct already-searched CREO gates and audit their fold stability."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from common import DATA, load_inputs, read_jsonl
from creo_features import extract_group_features, gate_feature_matrices, group_labels
from creo_model import fit_logistic, gate_group_choices, logistic_probabilities
from run_stage3 import cross_fitted_logistic, local_folds_from_global, slice_blocks
from selector_features import extract_feature_blocks, hard_group_mask, labels_from_references


def main() -> None:
    bank_all, arrays = load_inputs("train", DATA)
    refs = read_jsonl(DATA / "gsm8k_train.jsonl")
    blocks_all = extract_feature_blocks(bank_all, arrays["embeddings"])
    labels_all = labels_from_references(bank_all, refs)
    hard_indices = np.flatnonzero(hard_group_mask(blocks_all["answers"]))
    bank = [bank_all[int(i)] for i in hard_indices]
    blocks = slice_blocks(blocks_all, hard_indices)
    folds = local_folds_from_global(hard_indices)

    with np.load("runs/cv/hand_centered__h0.01__e1__listwise/oof_predictions.npz", allow_pickle=False) as s1, np.load(
        "runs/stage2/cv/hand_centered__uniform__pair0.1/oof_predictions.npz", allow_pickle=False
    ) as s2, np.load("runs/stage3/selected_predictions.npz", allow_pickle=False) as selected:
        base_scores = np.stack([s1["scores"], s2["scores"]], axis=2)
        stage2_choices = s2["choices"].astype(int)
        group_scores = selected["train_group_scores"].astype(np.float64)
        reliability = selected["train_reliability"].astype(np.float64)

    groups = extract_group_features(
        bank, arrays["embeddings"][hard_indices], blocks, base_scores
    )
    labels = group_labels(labels_all[hard_indices], groups["memberships"])
    gate_base = gate_feature_matrices(bank, groups, group_scores, labels)
    reliability_logit = np.log(
        np.clip(reliability, 1e-8, 1 - 1e-8)
        / np.clip(1 - reliability, 1e-8, 1)
    )
    features = np.column_stack([gate_base["features"], reliability, reliability_logit])
    stage2_correct = labels_all[hard_indices, stage2_choices]
    summary = json.loads(Path("runs/stage3/experiment_summary.json").read_text(encoding="utf-8"))
    validation_by_key = {row["key"]: row["validation_metrics"] for row in summary["gate_results"]}

    rows = []
    for l2 in (0.01, 0.1, 1.0):
        for balanced in (False, True):
            probabilities, _ = cross_fitted_logistic(
                features,
                gate_base["overturn_label"],
                folds,
                l2,
                eligible=gate_base["informative"],
                class_balance=balanced,
            )
            for threshold in (0.3, 0.4, 0.5, 0.6, 0.7, 0.8):
                key = f"gate_l2{l2:g}_b{int(balanced)}_t{threshold:g}"
                choices = gate_group_choices(
                    gate_base["plurality"], gate_base["challenger"], probabilities, threshold
                )
                correct = labels[np.arange(len(labels)), choices]
                deltas = [
                    int(correct[idx].sum() - stage2_correct[idx].sum())
                    for idx in folds
                ]
                rows.append({
                    "key": key,
                    "oof_correct": int(correct.sum()),
                    "stage2_oof_correct": int(stage2_correct.sum()),
                    "oof_delta": int(correct.sum() - stage2_correct.sum()),
                    "fold_deltas": deltas,
                    "folds_non_decreasing": int(sum(delta >= 0 for delta in deltas)),
                    "folds_improving": int(sum(delta > 0 for delta in deltas)),
                    "validation_hard_correct": validation_by_key[key]["hard_correct"],
                    "validation_trap_correct": validation_by_key[key]["trap_correct"],
                    "validation_nontrap_correct": validation_by_key[key]["nontrap_correct"],
                    "validation_good_overturns": validation_by_key[key]["good_overturns"],
                    "validation_bad_overturns": validation_by_key[key]["bad_overturns"],
                })

    robust = [row for row in rows if row["oof_delta"] >= 0 and row["folds_non_decreasing"] >= 4]
    robust.sort(key=lambda row: (
        row["validation_hard_correct"], row["oof_delta"], row["folds_non_decreasing"]
    ), reverse=True)
    payload = {
        "scope": "already-searched gate grid; train-hard OOF and validation only",
        "stage2_oof_correct": int(stage2_correct.sum()),
        "robust_rule": "OOF delta >= 0 and at least 4/5 folds non-decreasing",
        "robust_candidates": robust,
        "all_candidates": rows,
    }
    output = Path("runs/stage3/gate_stability_audit.json")
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "robust_count": len(robust),
        "top_robust": robust[:10],
    }, indent=2))


if __name__ == "__main__":
    main()
