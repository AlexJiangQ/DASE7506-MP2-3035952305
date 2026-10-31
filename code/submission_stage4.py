"""Stage 4 CREO-3W inference: plurality, candidate, and answer-group branches."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from selector_features import extract_feature_blocks, hard_group_mask
from stage4_features import (
    build_answer_groups,
    candidate_choices_to_groups,
    centered_embeddings,
    consensus_features,
    extract_group_features,
    group_argmax,
    group_choices_to_candidates,
    masked_candidate_choices,
    merge_branch_probabilities,
    plurality_groups,
)
from stage4_model import (
    binary_probabilities,
    gate_probabilities,
    load_bundle,
    score_candidate_model,
    score_group_model,
)


def predict(bank, arrays, checkpoint):
    if checkpoint is None:
        raise FileNotFoundError("Stage 4 CREO-3W requires --checkpoint runs/stage4/provisional_bundle")
    bundle = load_bundle(Path(checkpoint))
    config = bundle["config"]
    blocks = extract_feature_blocks(bank, arrays["embeddings"])
    groups = build_answer_groups(bank, blocks["answers"])
    plurality = plurality_groups(bank, groups)
    candidate_scores, projected = score_candidate_model(
        bundle["candidate_model"], centered_embeddings(arrays["embeddings"]),
        blocks["answers"], blocks["stage2_hand"],
    )
    candidate_choices = masked_candidate_choices(bank, candidate_scores, blocks["valid"])
    candidate_groups = candidate_choices_to_groups(candidate_choices, groups)
    group_features = extract_group_features(bank, groups, projected, candidate_scores)
    group_scores = score_group_model(
        bundle["group_model"], group_features["features"], group_features["valid"]
    )
    learned_groups = group_argmax(group_scores, group_features["valid"])
    branch_groups = np.column_stack([plurality, candidate_groups, learned_groups])
    base_features, _ = consensus_features(
        bank, groups, group_features, plurality, candidate_groups, learned_groups,
        candidate_scores, group_scores, hard_group_mask(blocks["answers"]),
    )
    reliabilities = np.column_stack([
        binary_probabilities(bundle["plurality_calibrator"], base_features),
        binary_probabilities(bundle["candidate_calibrator"], base_features),
        binary_probabilities(bundle["group_calibrator"], base_features),
    ])
    method = config["selected_method"]
    if method == "plurality":
        selected_groups = plurality
    elif method == "candidate":
        selected_groups = candidate_groups
    elif method == "group":
        selected_groups = learned_groups
    elif method == "fixed_reliability":
        selected_groups = merge_branch_probabilities(branch_groups, reliabilities)
    elif method.startswith("gate_"):
        probabilities = gate_probabilities(
            bundle["gate_model"], np.column_stack([base_features, reliabilities])
        )
        selected_groups = merge_branch_probabilities(branch_groups, probabilities)
    else:
        raise ValueError(f"Unsupported Stage 4 selection method: {method}")
    return [int(choice) for choice in group_choices_to_candidates(
        bank, selected_groups, groups, candidate_scores
    )]
