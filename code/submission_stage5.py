"""Stage 5 CREO-RR inference: selective revision of a frozen CREO anchor."""
from __future__ import annotations

from pathlib import Path

from stage5_features import (
    build_full_state,
    build_residual_examples,
    combine_feature_blocks,
    groups_to_candidates,
    select_revisions,
    source_example_mask,
)
from stage5_model import load_stage5_bundle, predict_probabilities


def _slice_examples(examples, row_mask):
    result = {
        "blocks": {name: values[row_mask] for name, values in examples["blocks"].items()},
        "block_names": examples["block_names"],
        "question_indices": examples["question_indices"][row_mask],
        "proposal_groups": examples["proposal_groups"][row_mask],
        "source_masks": examples["source_masks"][row_mask],
        "anchor_groups": examples["anchor_groups"],
    }
    return result


def predict(bank, arrays, checkpoint):
    if checkpoint is None:
        raise FileNotFoundError("Stage 5 CREO-RR requires --checkpoint runs/stage5/provisional_bundle")
    checkpoint = Path(checkpoint)
    bundle = load_stage5_bundle(checkpoint)
    config = bundle["config"]
    anchor_path = checkpoint / config.get("anchor_checkpoint", "anchor_stage3")
    if not anchor_path.is_dir():
        raise FileNotFoundError(f"Frozen Stage-3 anchor is missing: {anchor_path}")

    state = build_full_state(bank, arrays["embeddings"], anchor_path)
    if not bool(config.get("apply_residual", True)):
        return groups_to_candidates(state, state["anchor_groups"])

    sources = tuple(config["proposal_sources"])
    examples_all = build_residual_examples(state)
    row_mask = source_example_mask(examples_all, sources)
    examples = _slice_examples(examples_all, row_mask)
    features, names = combine_feature_blocks(examples, config["feature_set"])
    if names != config["feature_names"]:
        raise ValueError("Stage-5 feature schema does not match checkpoint")
    probabilities = predict_probabilities(
        bundle["model"], features, float(config["temperature"])
    )
    selected, _ = select_revisions(
        state, examples, probabilities, sources,
        float(config["harm_cost"]), float(config["threshold"]),
    )
    return [int(value) for value in groups_to_candidates(state, selected)]
