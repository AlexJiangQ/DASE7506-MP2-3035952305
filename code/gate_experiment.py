"""Strict OOF gate experiment between plurality and the provisional comparator.

The gate is retained only if it meets all pre-declared criteria. This script
never reads test data and never changes the provisional checkpoint.
"""
from __future__ import annotations

from collections import Counter
import json
import math
from pathlib import Path

import numpy as np

from common import DATA, load_inputs, read_jsonl
from selector_features import extract_feature_blocks, hard_group_mask, labels_from_references, plurality_choice
from selector_model import choices_from_scores, load_model, score_model


RUNS = Path("runs")


def parsed_choice_answer(answers, choices):
    return [answers[i][int(choice)] for i, choice in enumerate(choices)]


def gate_features(scores, valid, answers, comparator_choices, plurality_choices, hand):
    result = np.zeros((len(scores), 10), dtype=np.float64)
    for i in range(len(scores)):
        active = np.flatnonzero(valid[i])
        if len(active) == 0:
            active = np.arange(4)
        active_scores = np.sort(scores[i, active])
        margin = active_scores[-1] - active_scores[-2] if len(active_scores) >= 2 else 0.0
        counts = Counter(answer for answer in answers[i] if answer is not None)
        count_values = sorted(counts.values(), reverse=True)
        max_count = count_values[0] if count_values else 0
        second = count_values[1] if len(count_values) > 1 else 0
        total = sum(count_values)
        cluster_entropy = 0.0
        if total:
            for count in count_values:
                p = count / total
                cluster_entropy -= p * math.log(max(p, 1e-12))
        comparator = int(comparator_choices[i])
        plurality = int(plurality_choices[i])
        # The three grounding and arithmetic summaries are from the chosen trace.
        result[i] = [
            max_count / 4.0, (max_count - second) / 4.0, len(counts) / 4.0,
            cluster_entropy, margin,
            float(answers[i][comparator] == answers[i][plurality]),
            hand[i, comparator, 20],  # reused question-number ratio
            hand[i, comparator, 26],  # arithmetic pass ratio
            hand[i, comparator, 27],  # failed equation indicator
            hand[i, comparator, 28],  # equation produces final
        ]
    return result


def actual_plurality_choices(bank):
    return np.asarray([plurality_choice(row) for row in bank], dtype=int)


def fit_logistic(x, y, l2=0.1, steps=300, lr=0.05):
    mean = x.mean(axis=0)
    scale = x.std(axis=0)
    scale[scale < 1e-8] = 1.0
    z = (x - mean) / scale
    w = np.zeros(z.shape[1], dtype=np.float64)
    b = 0.0
    for _ in range(steps):
        logits = np.clip(z @ w + b, -40, 40)
        probs = 1.0 / (1.0 + np.exp(-logits))
        error = probs - y
        w -= lr * ((z.T @ error) / len(z) + l2 * w)
        b -= lr * float(error.mean())
    return {"mean": mean, "scale": scale, "weights": w, "bias": b}


def gate_probability(model, x):
    logits = ((x - model["mean"]) / model["scale"]) @ model["weights"] + model["bias"]
    return 1.0 / (1.0 + np.exp(-np.clip(logits, -40, 40)))


def gated_choices(model, features, comparator, plurality, comparator_answers, plurality_answers):
    probability = gate_probability(model, features)
    output = comparator.copy()
    conflict = np.asarray(comparator_answers) != np.asarray(plurality_answers)
    use_plurality = conflict & (probability < 0.5)
    output[use_plurality] = plurality[use_plurality]
    return output, probability, conflict


def acc(labels, choices):
    return 100.0 * float(np.mean(labels[np.arange(len(labels)), choices]))


def main():
    train_bank, train_arrays = load_inputs("train", DATA)
    train_refs = read_jsonl(DATA / "gsm8k_train.jsonl")
    train_blocks = extract_feature_blocks(train_bank, train_arrays["embeddings"])
    train_labels = labels_from_references(train_bank, train_refs)
    hard = hard_group_mask(train_blocks["answers"])
    hard_indices = np.flatnonzero(hard)
    summary = json.loads((RUNS / "experiment_summary.json").read_text(encoding="utf-8"))
    config = summary["provisional_best"]
    with np.load(RUNS / "cv" / config["key"] / "oof_predictions.npz", allow_pickle=False) as data:
        oof_indices = data["indices"].astype(int)
        oof_scores = data["scores"].astype(np.float64)
        comparator = data["choices"].astype(int)
    if not np.array_equal(oof_indices, hard_indices):
        raise ValueError("OOF index alignment mismatch")
    folds = json.loads((RUNS / "folds.json").read_text(encoding="utf-8"))["fold_indices"]
    index_to_pos = {int(index): pos for pos, index in enumerate(oof_indices)}
    plurality_full = actual_plurality_choices(train_bank)
    plurality = plurality_full[oof_indices]
    answers = [train_blocks["answers"][int(i)] for i in oof_indices]
    valid = train_blocks["valid"][oof_indices]
    hand = train_blocks["hand"][oof_indices]
    features = gate_features(oof_scores, valid, answers, comparator, plurality, hand)
    comp_answers = parsed_choice_answer(answers, comparator)
    plur_answers = parsed_choice_answer(answers, plurality)
    comparator_correct = train_labels[oof_indices, comparator]
    plurality_correct = train_labels[oof_indices, plurality]
    informative = (np.asarray(comp_answers) != np.asarray(plur_answers)) & (comparator_correct != plurality_correct)

    fold_results = []
    fold_models = []
    for fold_no, original_eval in enumerate(folds):
        eval_pos = np.asarray([index_to_pos[int(i)] for i in original_eval], dtype=int)
        train_mask = np.ones(len(oof_indices), dtype=bool)
        train_mask[eval_pos] = False
        fit_mask = train_mask & informative
        target = comparator_correct[fit_mask].astype(np.float64)
        model = fit_logistic(features[fit_mask], target)
        gated, probability, conflict = gated_choices(
            model, features[eval_pos], comparator[eval_pos], plurality[eval_pos],
            np.asarray(comp_answers, dtype=object)[eval_pos], np.asarray(plur_answers, dtype=object)[eval_pos]
        )
        base_acc = acc(train_labels[oof_indices[eval_pos]], comparator[eval_pos])
        gate_acc = acc(train_labels[oof_indices[eval_pos]], gated)
        fold_results.append({
            "fold": fold_no, "n_gate_train": int(fit_mask.sum()), "n_eval": len(eval_pos),
            "conflicts": int(conflict.sum()), "comparator_accuracy": base_acc,
            "gate_accuracy": gate_acc, "delta": gate_acc - base_acc,
        })
        fold_models.append(model)

    # Train on every OOF informative conflict, then evaluate validation once.
    gate_model = fit_logistic(features[informative], comparator_correct[informative].astype(np.float64))
    val_bank, val_arrays = load_inputs("validation", DATA)
    val_refs = read_jsonl(DATA / "gsm8k_validation.jsonl")
    val_blocks = extract_feature_blocks(val_bank, val_arrays["embeddings"])
    val_labels = labels_from_references(val_bank, val_refs)
    val_hard = hard_group_mask(val_blocks["answers"])
    provisional = load_model(RUNS / "provisional_model_bundle")
    val_scores = score_model(provisional, val_blocks)
    val_comparator = np.asarray(choices_from_scores(val_bank, val_scores, val_blocks["valid"]), dtype=int)
    val_plurality = actual_plurality_choices(val_bank)
    val_features = gate_features(
        val_scores, val_blocks["valid"], val_blocks["answers"],
        val_comparator, val_plurality, val_blocks["hand"]
    )
    val_comp_answers = parsed_choice_answer(val_blocks["answers"], val_comparator)
    val_plur_answers = parsed_choice_answer(val_blocks["answers"], val_plurality)
    val_gated, _, val_conflict = gated_choices(
        gate_model, val_features, val_comparator, val_plurality,
        val_comp_answers, val_plur_answers,
    )
    mean_delta = float(np.mean([row["delta"] for row in fold_results]))
    nondeclining = sum(row["delta"] >= -1e-12 for row in fold_results)
    val_base_hard = acc(val_labels[val_hard], val_comparator[val_hard])
    val_gate_hard = acc(val_labels[val_hard], val_gated[val_hard])
    retained = mean_delta >= 1.0 and nondeclining >= 4 and val_gate_hard >= val_base_hard
    result = {
        "status": "retained" if retained else "rejected",
        "reason": "Gate must gain >=1 point mean, not decline in >=4/5 folds, and not reduce validation-hard.",
        "folds": fold_results, "cv_mean_delta": mean_delta,
        "nondeclining_folds": nondeclining,
        "validation_conflicts": int(val_conflict.sum()),
        "validation_comparator_all": acc(val_labels, val_comparator),
        "validation_gate_all": acc(val_labels, val_gated),
        "validation_comparator_hard": val_base_hard,
        "validation_gate_hard": val_gate_hard,
        "checkpoint_changed": False,
    }
    (RUNS / "gate_experiment.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

