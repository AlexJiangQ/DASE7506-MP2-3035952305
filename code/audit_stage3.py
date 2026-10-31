"""Read-only Stage 3 paired audit on train-hard OOF and validation predictions."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from common import DATA, load_inputs, read_jsonl
from creo_features import extract_group_features, group_labels, plurality_group_indices
from run_stage3 import local_folds_from_global, slice_blocks
from selector_features import extract_feature_blocks, hard_group_mask, labels_from_references


def paired_counts(first: np.ndarray, second: np.ndarray) -> dict:
    return {
        "both_correct": int(np.sum(first & second)),
        "stage2_only": int(np.sum(first & ~second)),
        "creo_only": int(np.sum(~first & second)),
        "both_wrong": int(np.sum(~first & ~second)),
        "stage2_correct": int(first.sum()),
        "creo_correct": int(second.sum()),
        "delta_correct": int(second.sum() - first.sum()),
        "n": int(len(first)),
    }


def main() -> None:
    bank_all, arrays = load_inputs("train", DATA)
    refs = read_jsonl(DATA / "gsm8k_train.jsonl")
    blocks_all = extract_feature_blocks(bank_all, arrays["embeddings"])
    labels_all = labels_from_references(bank_all, refs)
    hard_indices = np.flatnonzero(hard_group_mask(blocks_all["answers"]))
    bank = [bank_all[int(i)] for i in hard_indices]
    blocks = slice_blocks(blocks_all, hard_indices)
    embeddings = arrays["embeddings"][hard_indices]

    stage1_path = Path("runs/cv/hand_centered__h0.01__e1__listwise/oof_predictions.npz")
    stage2_path = Path("runs/stage2/cv/hand_centered__uniform__pair0.1/oof_predictions.npz")
    selected_path = Path("runs/stage3/selected_predictions.npz")
    with np.load(stage1_path, allow_pickle=False) as s1, np.load(stage2_path, allow_pickle=False) as s2:
        base_scores = np.stack([s1["scores"], s2["scores"]], axis=2)
        stage2_choices = s2["choices"].astype(int)
    groups = extract_group_features(bank, embeddings, blocks, base_scores)
    labels = group_labels(labels_all[hard_indices], groups["memberships"])
    with np.load(selected_path, allow_pickle=False) as selected:
        creo_groups = selected["train_selected_groups"].astype(int)
    creo_correct = labels[np.arange(len(labels)), creo_groups]
    stage2_correct = labels_all[hard_indices, stage2_choices]
    plurality = plurality_group_indices(bank, groups)
    plurality_correct = labels[np.arange(len(labels)), plurality]

    folds = local_folds_from_global(hard_indices)
    fold_rows = []
    for fold_no, indices in enumerate(folds):
        row = {"fold": fold_no, **paired_counts(stage2_correct[indices], creo_correct[indices])}
        row["stage2_accuracy"] = 100.0 * row["stage2_correct"] / row["n"]
        row["creo_accuracy"] = 100.0 * row["creo_correct"] / row["n"]
        fold_rows.append(row)

    masks = {
        "all_train_hard": np.ones(len(labels), dtype=bool),
        "wrong_plurality_traps": ~plurality_correct,
        "correct_plurality_nontraps": plurality_correct,
        "single_correct_candidate": labels_all[hard_indices].sum(axis=1) == 1,
    }
    strata = {name: paired_counts(stage2_correct[mask], creo_correct[mask]) for name, mask in masks.items()}
    payload = {
        "scope": "train-hard OOF only; no test data used",
        "folds": fold_rows,
        "strata": strata,
        "fold_delta_mean_correct": float(np.mean([row["delta_correct"] for row in fold_rows])),
        "folds_non_decreasing": int(sum(row["delta_correct"] >= 0 for row in fold_rows)),
        "folds_improving": int(sum(row["delta_correct"] > 0 for row in fold_rows)),
    }
    output = Path("runs/stage3/paired_oof_audit.json")
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
