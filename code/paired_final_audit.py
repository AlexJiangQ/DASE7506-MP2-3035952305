"""Non-training paired audit of frozen Stage-1 and Stage-2 predictions.

Only existing OOF and validation prediction artifacts are compared. No model
is fitted and no new prediction is generated.
"""
from __future__ import annotations

from collections import Counter
import csv
import json
from pathlib import Path

import numpy as np

from common import DATA, load_inputs, read_jsonl
from selector_features import extract_feature_blocks, hard_group_mask, labels_from_references, plurality_choice


FINAL = Path("runs/final")
STAGE1_OOF = Path("runs/cv/hand_centered__h0.01__e1__listwise/oof_predictions.npz")
STAGE2_OOF = Path("runs/stage2/cv/hand_centered__uniform__pair0.1/oof_predictions.npz")
STAGE1_VALIDATION = Path("runs/validation-predictions-1.json")
STAGE2_VALIDATION = Path("runs/stage2/validation_predictions.jsonl")


def answer_pattern(answers: list[str | None]) -> str:
    counts = sorted(Counter(answer for answer in answers if answer is not None).values(), reverse=True)
    return "-".join(str(value) for value in counts) if counts else "none-valid"


def paired_counts(labels: np.ndarray, stage1: np.ndarray, stage2: np.ndarray, mask: np.ndarray) -> dict:
    indices = np.flatnonzero(mask)
    s1 = labels[indices, stage1[indices]]
    s2 = labels[indices, stage2[indices]]
    n = len(indices)
    counts = {
        "both_correct": int((s1 & s2).sum()),
        "stage1_only": int((s1 & ~s2).sum()),
        "stage2_only": int((~s1 & s2).sum()),
        "both_wrong": int((~s1 & ~s2).sum()),
    }
    return {
        "n": n,
        **counts,
        "stage1_accuracy": 100.0 * (counts["both_correct"] + counts["stage1_only"]) / n,
        "stage2_accuracy": 100.0 * (counts["both_correct"] + counts["stage2_only"]) / n,
        "stage2_minus_stage1_correct": counts["stage2_only"] - counts["stage1_only"],
    }


def make_strata(labels: np.ndarray, answers: list[list[str | None]], bank: list[dict]) -> dict[str, np.ndarray]:
    correct_count = labels.sum(axis=1)
    patterns = np.asarray([answer_pattern(row) for row in answers], dtype=object)
    plurality = np.asarray([plurality_choice(row) for row in bank], dtype=int)
    plurality_correct = labels[np.arange(len(labels)), plurality]
    wrong_3to1 = (patterns == "3-1") & ~plurality_correct
    single = correct_count == 1
    return {
        "all_hard": np.ones(len(labels), dtype=bool),
        "single_correct_all": single,
        "wrong_3to1_consensus": wrong_3to1,
        "single_correct_excluding_wrong_3to1": single & ~wrong_3to1,
        "other_hard": ~single & ~wrong_3to1,
    }


def audit_split(
    name: str,
    labels: np.ndarray,
    answers: list[list[str | None]],
    bank: list[dict],
    stage1: np.ndarray,
    stage2: np.ndarray,
) -> dict:
    strata = make_strata(labels, answers, bank)
    return {
        "name": name,
        "strata": {
            stratum: paired_counts(labels, stage1, stage2, mask)
            for stratum, mask in strata.items()
        },
    }


def main() -> None:
    FINAL.mkdir(parents=True, exist_ok=True)

    train_bank, train_arrays = load_inputs("train", DATA)
    train_refs = read_jsonl(DATA / "gsm8k_train.jsonl")
    train_blocks = extract_feature_blocks(train_bank, train_arrays["embeddings"])
    train_labels_all = labels_from_references(train_bank, train_refs)
    hard_indices = np.flatnonzero(hard_group_mask(train_blocks["answers"]))
    with np.load(STAGE1_OOF, allow_pickle=False) as data1, np.load(STAGE2_OOF, allow_pickle=False) as data2:
        indices1 = data1["indices"].astype(int)
        indices2 = data2["indices"].astype(int)
        if not np.array_equal(indices1, hard_indices) or not np.array_equal(indices2, hard_indices):
            raise ValueError("OOF indices do not match the frozen train-hard ordering")
        stage1_oof = data1["choices"].astype(int)
        stage2_oof = data2["choices"].astype(int)
    train_hard_bank = [train_bank[int(index)] for index in hard_indices]
    train_hard_answers = [train_blocks["answers"][int(index)] for index in hard_indices]
    train_audit = audit_split(
        "train_hard_oof",
        train_labels_all[hard_indices],
        train_hard_answers,
        train_hard_bank,
        stage1_oof,
        stage2_oof,
    )

    validation_bank, validation_arrays = load_inputs("validation", DATA)
    validation_refs = read_jsonl(DATA / "gsm8k_validation.jsonl")
    validation_blocks = extract_feature_blocks(validation_bank, validation_arrays["embeddings"])
    validation_labels_all = labels_from_references(validation_bank, validation_refs)
    validation_hard = hard_group_mask(validation_blocks["answers"])
    stage1_payload = json.loads(STAGE1_VALIDATION.read_text(encoding="utf-8"))
    stage2_payload = json.loads(STAGE2_VALIDATION.read_text(encoding="utf-8"))
    expected_ids = [row["id"] for row in validation_bank]
    if stage1_payload["ids"] != expected_ids or stage2_payload["ids"] != expected_ids:
        raise ValueError("Validation prediction IDs are not aligned")
    stage1_validation_all = np.asarray(stage1_payload["selected"], dtype=int)
    stage2_validation_all = np.asarray(stage2_payload["selected"], dtype=int)
    hard_positions = np.flatnonzero(validation_hard)
    validation_hard_bank = [validation_bank[int(index)] for index in hard_positions]
    validation_hard_answers = [validation_blocks["answers"][int(index)] for index in hard_positions]
    validation_audit = audit_split(
        "validation_hard_existing_predictions",
        validation_labels_all[validation_hard],
        validation_hard_answers,
        validation_hard_bank,
        stage1_validation_all[validation_hard],
        stage2_validation_all[validation_hard],
    )
    validation_all = paired_counts(
        validation_labels_all,
        stage1_validation_all,
        stage2_validation_all,
        np.ones(len(validation_labels_all), dtype=bool),
    )

    output = {
        "kind": "non_training_paired_audit",
        "new_predictions_generated": False,
        "models_trained": False,
        "features_or_hyperparameters_changed": False,
        "sources": {
            "stage1_oof": str(STAGE1_OOF),
            "stage2_oof": str(STAGE2_OOF),
            "stage1_validation": str(STAGE1_VALIDATION),
            "stage2_validation": str(STAGE2_VALIDATION),
        },
        "train_hard_oof": train_audit["strata"],
        "validation_all": validation_all,
        "validation_hard": validation_audit["strata"],
    }
    (FINAL / "paired_audit.json").write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")

    rows = []
    for split_name in ("train_hard_oof", "validation_hard"):
        for stratum, metrics in output[split_name].items():
            rows.append({"split": split_name, "stratum": stratum, **metrics})
    rows.append({"split": "validation_all", "stratum": "all", **validation_all})
    with (FINAL / "paired_audit.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        "# Frozen-candidate paired audit",
        "",
        "This audit only compares existing Stage 1 and Stage 2 OOF/validation choices. It performs no training, tuning, feature change, or new prediction.",
        "",
    ]
    for split_name, title in (("train_hard_oof", "Train-hard OOF"), ("validation_hard", "Validation-hard")):
        lines += [
            f"## {title}",
            "",
            "| Stratum | N | Both correct | Stage 1 only | Stage 2 only | Both wrong | Stage 1 acc. | Stage 2 acc. |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for stratum, metrics in output[split_name].items():
            lines.append(
                f"| {stratum} | {metrics['n']} | {metrics['both_correct']} | {metrics['stage1_only']} | "
                f"{metrics['stage2_only']} | {metrics['both_wrong']} | {metrics['stage1_accuracy']:.2f}% | "
                f"{metrics['stage2_accuracy']:.2f}% |"
            )
        lines.append("")
    va = validation_all
    lines += [
        "## Validation-all",
        "",
        f"Both correct {va['both_correct']}; Stage-1-only {va['stage1_only']}; Stage-2-only {va['stage2_only']}; both wrong {va['both_wrong']}. Stage 2 has {va['stage2_minus_stage1_correct']} net correct answers relative to Stage 1.",
        "",
        "The final candidate remains Stage 2 by the pre-declared CV rule; this audit is descriptive and does not change the model.",
    ]
    (FINAL / "PAIRED_AUDIT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
