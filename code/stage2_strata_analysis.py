"""Post-selection train-OOF difficult-strata analysis for all stage-2 runs."""
from __future__ import annotations

from collections import Counter
import csv
import json
from pathlib import Path
import time

import numpy as np

from common import DATA, load_inputs, read_jsonl
from selector_features import extract_feature_blocks, hard_group_mask, labels_from_references, plurality_choice
from selector_model import cluster_choices_from_scores


STAGE2 = Path("runs/stage2")


def answer_pattern(answers: list[str | None]) -> str:
    counts = sorted(Counter(a for a in answers if a is not None).values(), reverse=True)
    return "-".join(str(value) for value in counts) if counts else "none-valid"


def accuracy(labels: np.ndarray, choices: np.ndarray, mask: np.ndarray) -> float:
    return 100.0 * float(np.mean(labels[mask][np.arange(int(mask.sum())), choices[mask]]))


def main() -> None:
    bank, arrays = load_inputs("train", DATA)
    references = read_jsonl(DATA / "gsm8k_train.jsonl")
    blocks = extract_feature_blocks(bank, arrays["embeddings"])
    labels_all = labels_from_references(bank, references)
    hard_indices = np.flatnonzero(hard_group_mask(blocks["answers"]))
    bank_hard = [bank[int(index)] for index in hard_indices]
    labels = labels_all[hard_indices]
    answers = [blocks["answers"][int(index)] for index in hard_indices]
    valid = blocks["valid"][hard_indices]
    plurality = np.asarray([plurality_choice(row) for row in bank_hard], dtype=int)
    correct_count = labels.sum(axis=1)
    patterns = np.asarray([answer_pattern(row) for row in answers], dtype=object)
    plurality_correct = labels[np.arange(len(labels)), plurality]
    masks = {
        "all_hard": np.ones(len(labels), dtype=bool),
        "single_correct": correct_count == 1,
        "wrong_3to1_consensus": (patterns == "3-1") & ~plurality_correct,
    }

    summary_path = STAGE2 / "experiment_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    with np.load(
        Path("runs/cv/hand_centered__h0.01__e1__listwise/oof_predictions.npz"),
        allow_pickle=False,
    ) as data:
        stage1_scores = data["scores"].astype(np.float64)

    rows = []
    detailed = {}
    for experiment in summary["experiments"]:
        key = experiment["key"]
        if experiment["method"] == "cluster_pooling":
            derived_started = time.perf_counter()
            choices = np.asarray(cluster_choices_from_scores(
                bank_hard,
                stage1_scores,
                valid,
                rule=experiment["decision_rule"],
                alpha=experiment["cluster_alpha"],
            ), dtype=int)
            derived_dir = STAGE2 / "cluster_pooling" / key
            derived_dir.mkdir(parents=True, exist_ok=True)
            np.savez(
                derived_dir / "oof_predictions.npz",
                indices=hard_indices,
                scores=stage1_scores,
                choices=choices,
            )
            (derived_dir / "config.json").write_text(json.dumps({
                "seed": 20261022,
                "source": "frozen stage-1 OOF candidate scores",
                "decision_rule": experiment["decision_rule"],
                "cluster_alpha": experiment["cluster_alpha"],
                "training_required": False,
            }, indent=2) + "\n", encoding="utf-8")
            (derived_dir / "result.json").write_text(json.dumps({
                **experiment,
                "derived_evaluation_seconds": time.perf_counter() - derived_started,
                "failure_reason_if_rejected": (
                    "gain below 0.75-point promotion threshold"
                    if key != "stage1_oof__candidate__a0" else None
                ),
            }, indent=2) + "\n", encoding="utf-8")
        else:
            with np.load(STAGE2 / "cv" / key / "oof_predictions.npz", allow_pickle=False) as data:
                if not np.array_equal(data["indices"].astype(int), hard_indices):
                    raise ValueError(f"OOF index mismatch for {key}")
                choices = data["choices"].astype(int)
        metrics = {
            name: {"n": int(mask.sum()), "accuracy": accuracy(labels, choices, mask)}
            for name, mask in masks.items()
        }
        detailed[key] = metrics
        for stratum, values in metrics.items():
            rows.append({
                "key": key,
                "variant": experiment["variant"],
                "weight_scheme": experiment["weight_scheme"],
                "pairwise_weight": experiment["pairwise_weight"],
                "decision_rule": experiment["decision_rule"],
                "stratum": stratum,
                **values,
            })

    output = {
        "seed": 20261022,
        "source": "train OOF only",
        "strata": detailed,
    }
    (STAGE2 / "oof_stratified_results.json").write_text(
        json.dumps(output, indent=2) + "\n", encoding="utf-8"
    )
    with (STAGE2 / "oof_stratified_results.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary["oof_difficult_strata"] = detailed
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "stage1": detailed["stage1_oof__candidate__a0"],
        "winner": detailed[summary["winner_pre_validation"]["key"]],
    }, indent=2))


if __name__ == "__main__":
    main()
