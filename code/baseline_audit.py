"""Record reproducible timing/configuration for deterministic baselines."""
import json
from pathlib import Path
import time

import numpy as np

from common import DATA, load_inputs, read_jsonl
from run_experiments import SEED, accuracy, baseline_choices, fold_stats, make_folds
from selector_features import extract_feature_blocks, hard_group_mask, labels_from_references


def main():
    bank, arrays = load_inputs("train", DATA)
    refs = read_jsonl(DATA / "gsm8k_train.jsonl")
    blocks = extract_feature_blocks(bank, arrays["embeddings"])
    labels = labels_from_references(bank, refs)
    hard_indices = np.flatnonzero(hard_group_mask(blocks["answers"]))
    folds = make_folds(hard_indices, SEED)
    for name in ("fixed_candidate_0", "deterministic_random", "longest_trace", "plurality"):
        started = time.perf_counter()
        choices = np.asarray(baseline_choices(name, bank), dtype=int)
        fold_accuracy = [accuracy(labels[fold], choices[fold]) for fold in folds]
        mean, se = fold_stats(fold_accuracy)
        elapsed = time.perf_counter() - started
        out = Path("runs") / "baselines" / name
        out.mkdir(parents=True, exist_ok=True)
        (out / "config.json").write_text(json.dumps({
            "name": name, "seed": SEED,
            "tie_rule": "content-hash tie break independent of ID and candidate order",
        }, indent=2) + "\n", encoding="utf-8")
        (out / "result.json").write_text(json.dumps({
            "cv_fold_accuracy": fold_accuracy, "cv_mean": mean, "cv_se": se,
            "elapsed_seconds": elapsed, "train_hard_n": len(hard_indices),
        }, indent=2) + "\n", encoding="utf-8")
    print("baseline audit recorded")


if __name__ == "__main__":
    main()

