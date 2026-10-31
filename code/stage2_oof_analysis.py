"""Stage-2 OOF error analysis using the frozen stage-1 folds and predictions.

This script reads train data only. It never opens validation or test data.
"""
from __future__ import annotations

from collections import Counter
import csv
import json
import math
import os
import platform
from pathlib import Path
import sys
import time

import numpy as np

from common import DATA, load_inputs, read_jsonl
from selector_features import extract_feature_blocks, hard_group_mask, labels_from_references, plurality_choice


SEED = 20261022
STAGE1_OOF = Path("runs/cv/hand_centered__h0.01__e1__listwise/oof_predictions.npz")


def dump_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def accuracy(labels: np.ndarray, choices: np.ndarray) -> float:
    if not len(choices):
        return float("nan")
    return 100.0 * float(np.mean(labels[np.arange(len(labels)), choices]))


def answer_pattern(answers: list[str | None]) -> str:
    counts = sorted(Counter(a for a in answers if a is not None).values(), reverse=True)
    return "-".join(str(value) for value in counts) if counts else "none-valid"


def aggregate(mask, labels, comparator, plurality, conflicts, margins) -> dict:
    mask = np.asarray(mask, dtype=bool)
    n = int(mask.sum())
    if n == 0:
        return {"n": 0, "comparator_accuracy": None, "plurality_accuracy": None,
                "delta_points": None, "conflict_rate": None, "mean_margin": None}
    comp = accuracy(labels[mask], comparator[mask])
    plur = accuracy(labels[mask], plurality[mask])
    return {
        "n": n,
        "comparator_accuracy": comp,
        "plurality_accuracy": plur,
        "delta_points": comp - plur,
        "conflict_rate": float(conflicts[mask].mean()),
        "mean_margin": float(margins[mask].mean()),
    }


def main() -> None:
    stage2 = Path("runs/stage2")
    analysis_dir = stage2 / "oof_analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    state_path = stage2 / "search_state.json"
    if state_path.is_file():
        state = json.loads(state_path.read_text(encoding="utf-8"))
    else:
        state = {
            "status": "running",
            "seed": SEED,
            "started_unix": time.time(),
            "budget_minutes": 90,
            "test_touched": False,
            "validation_runs": 0,
        }
    dump_json(state_path, state)
    dump_json(stage2 / "environment.json", {
        "python": sys.version,
        "numpy": np.__version__,
        "platform": platform.platform(),
        "processor": platform.processor(),
        "logical_cpu_count": os.cpu_count(),
        "thread_environment": {name: os.environ.get(name) for name in
                               ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")},
    })

    bank, arrays = load_inputs("train", DATA)
    references = read_jsonl(DATA / "gsm8k_train.jsonl")
    blocks = extract_feature_blocks(bank, arrays["embeddings"])
    labels_all = labels_from_references(bank, references)
    hard = hard_group_mask(blocks["answers"])
    hard_indices = np.flatnonzero(hard)

    folds_payload = json.loads(Path("runs/folds.json").read_text(encoding="utf-8"))
    folds = [np.asarray(fold, dtype=int) for fold in folds_payload["fold_indices"]]
    folded = np.concatenate(folds)
    if folds_payload.get("seed") != SEED or set(folded.tolist()) != set(hard_indices.tolist()):
        raise ValueError("Stage-1 folds do not exactly cover the current train-hard groups")

    with np.load(STAGE1_OOF, allow_pickle=False) as data:
        oof_indices = data["indices"].astype(int)
        scores = data["scores"].astype(np.float64)
        comparator = data["choices"].astype(int)
    if not np.array_equal(oof_indices, hard_indices):
        raise ValueError("Stage-1 OOF indices differ from the frozen train-hard ordering")

    bank_hard = [bank[int(i)] for i in hard_indices]
    labels = labels_all[hard_indices]
    answers = [blocks["answers"][int(i)] for i in hard_indices]
    valid = blocks["valid"][hard_indices]
    plurality = np.asarray([plurality_choice(row) for row in bank_hard], dtype=int)
    active_scores = np.where(valid, scores, -np.inf)
    all_invalid = ~valid.any(axis=1)
    active_scores[all_invalid] = scores[all_invalid]
    sorted_scores = np.sort(active_scores, axis=1)
    margins = sorted_scores[:, -1] - sorted_scores[:, -2]

    comp_answers = np.asarray([answers[i][comparator[i]] for i in range(len(answers))], dtype=object)
    plurality_answers = np.asarray([answers[i][plurality[i]] for i in range(len(answers))], dtype=object)
    conflicts = comp_answers != plurality_answers
    comp_correct = labels[np.arange(len(labels)), comparator]
    plurality_correct = labels[np.arange(len(labels)), plurality]
    correct_count = labels.sum(axis=1)
    distinct = np.asarray([len({a for a in row if a is not None}) for row in answers], dtype=int)
    patterns = np.asarray([answer_pattern(row) for row in answers], dtype=object)
    wrong_consensus_3_to_1 = (patterns == "3-1") & ~plurality_correct
    single_correct = correct_count == 1

    strata = {
        "overall": {"all": np.ones(len(labels), dtype=bool)},
        "correct_candidate_count": {str(v): correct_count == v for v in sorted(set(correct_count.tolist()))},
        "distinct_valid_answers": {str(v): distinct == v for v in sorted(set(distinct.tolist()))},
        "plurality_pattern": {str(v): patterns == v for v in sorted(set(patterns.tolist()))},
        "target_groups": {
            "single_correct": single_correct,
            "wrong_consensus_3_to_1": wrong_consensus_3_to_1,
            "comparator_plurality_conflict": conflicts,
            "comparator_good_overturn": conflicts & comp_correct & ~plurality_correct,
            "comparator_bad_overturn": conflicts & ~comp_correct & plurality_correct,
            "both_correct": comp_correct & plurality_correct,
            "both_wrong": ~comp_correct & ~plurality_correct,
        },
    }
    aggregated = {
        dimension: {
            name: aggregate(mask, labels, comparator, plurality, conflicts, margins)
            for name, mask in values.items()
        }
        for dimension, values in strata.items()
    }

    fold_results = []
    position = {int(index): pos for pos, index in enumerate(hard_indices)}
    for fold_no, fold in enumerate(folds):
        local = np.asarray([position[int(index)] for index in fold], dtype=int)
        fold_results.append({
            "fold": fold_no,
            **aggregate(np.isin(np.arange(len(labels)), local), labels, comparator,
                        plurality, conflicts, margins),
            "good_overturns": int((conflicts[local] & comp_correct[local] & ~plurality_correct[local]).sum()),
            "bad_overturns": int((conflicts[local] & ~comp_correct[local] & plurality_correct[local]).sum()),
        })

    output = {
        "seed": SEED,
        "source_oof": str(STAGE1_OOF),
        "source_folds": "runs/folds.json",
        "n_train_hard": len(hard_indices),
        "folds_exactly_reused": True,
        "strata": aggregated,
        "folds": fold_results,
    }
    dump_json(analysis_dir / "oof_error_analysis.json", output)

    rows = []
    for dimension, values in aggregated.items():
        for stratum, metrics in values.items():
            rows.append({"dimension": dimension, "stratum": stratum, **metrics})
    with (analysis_dir / "oof_error_analysis.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    overall = aggregated["overall"]["all"]
    target = aggregated["target_groups"]
    lines = [
        "# Stage 2 OOF error analysis",
        "",
        f"- Frozen seed/folds: `{SEED}` / exact reuse of `runs/folds.json`.",
        f"- Train-hard groups: {len(hard_indices)}.",
        f"- Stage-1 comparator: {overall['comparator_accuracy']:.4f}%.",
        f"- Plurality: {overall['plurality_accuracy']:.4f}%.",
        f"- Answer-level conflicts: {target['comparator_plurality_conflict']['n']} groups.",
        f"- Good comparator overturns: {target['comparator_good_overturn']['n']}.",
        f"- Bad comparator overturns: {target['comparator_bad_overturn']['n']}.",
        f"- Wrong 3-to-1 consensus groups: {target['wrong_consensus_3_to_1']['n']}.",
        "",
        "These results diagnose where the frozen stage-1 comparator succeeds or fails; no validation data was used.",
    ]
    (analysis_dir / "OOF_ERROR_ANALYSIS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
