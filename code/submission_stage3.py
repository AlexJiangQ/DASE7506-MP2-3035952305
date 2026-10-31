"""Stage 3 CREO selector: consensus reliability and evidence-based overturning."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from creo_features import (
    extract_group_features,
    gate_feature_matrices,
    group_to_candidate_choices,
)
from creo_model import (
    gate_group_choices,
    load_creo_bundle,
    logistic_logits,
    logistic_probabilities,
    score_group_model,
)
from selector_features import extract_feature_blocks
from selector_model import load_model, score_model


def _candidate_groups(choices, memberships):
    result = np.zeros(len(choices), dtype=np.int64)
    for i, choice in enumerate(choices):
        result[i] = next(
            g for g, members in enumerate(memberships[i]) if int(choice) in members
        )
    return result


def _masked_argmax(scores, valid):
    masked = np.where(valid, scores, -np.inf)
    choices = np.argmax(masked, axis=1).astype(np.int64)
    choices[~np.asarray(valid).any(axis=1)] = 0
    return choices


def predict(bank, arrays, checkpoint):
    """Return one Python integer candidate index per question."""
    if checkpoint is None:
        raise FileNotFoundError(
            "Stage 3 CREO requires --checkpoint runs/stage3/provisional_creo_bundle"
        )
    checkpoint = Path(checkpoint)
    if not checkpoint.is_dir():
        raise FileNotFoundError(f"CREO checkpoint directory not found: {checkpoint}")

    bundle = load_creo_bundle(checkpoint)
    config = bundle["config"]
    base_names = config.get("base_checkpoints", ["base_stage1", "base_stage2"])
    if len(base_names) != 2:
        raise ValueError("CREO expects exactly two frozen base checkpoints")

    blocks = extract_feature_blocks(bank, arrays["embeddings"])
    base_scores = []
    for name in base_names:
        base_model = load_model(checkpoint / name)
        base_scores.append(score_model(base_model, blocks))
    candidate_scores = np.stack(base_scores, axis=2)

    group_data = extract_group_features(
        bank, arrays["embeddings"], blocks, candidate_scores
    )
    group_scores = score_group_model(
        bundle["group_model"], group_data["features"], group_data["valid"]
    )
    challenger_source = config.get("challenger_source", "learned_group")
    explicit_challenger = None
    if challenger_source in {"stage1", "stage2"}:
        model_index = 0 if challenger_source == "stage1" else 1
        candidate_choices = _masked_argmax(
            candidate_scores[:, :, model_index], blocks["valid"]
        )
        explicit_challenger = _candidate_groups(
            candidate_choices, group_data["memberships"]
        )
    elif challenger_source != "learned_group":
        raise ValueError(f"Unsupported challenger source: {challenger_source}")
    gate_base = gate_feature_matrices(
        bank, group_data, group_scores, challenger_indices=explicit_challenger
    )

    reliability_probability = logistic_probabilities(
        bundle["reliability_model"], gate_base["features"]
    )
    reliability_logit = logistic_logits(
        bundle["reliability_model"], gate_base["features"]
    )
    gate_features = np.column_stack([
        gate_base["features"], reliability_probability, reliability_logit
    ])
    gate_probability = logistic_probabilities(bundle["gate_model"], gate_features)
    selected_groups = gate_group_choices(
        gate_base["plurality"],
        gate_base["challenger"],
        gate_probability,
        float(config["threshold"]),
    )
    choices = group_to_candidate_choices(
        bank, selected_groups, group_data, candidate_scores
    )
    return [int(choice) for choice in choices]
