"""CREO-RR residual proposal and feature construction.

Features describe a proposed answer-group revision relative to the frozen
Stage-3 high-validation CREO anchor. They use only supplied inputs and frozen
model outputs and are equivariant to candidate permutation.
"""
from __future__ import annotations

import hashlib
import math
from pathlib import Path

import numpy as np

from creo_features import (
    extract_group_features,
    gate_feature_matrices,
    group_labels,
    group_to_candidate_choices,
    plurality_group_indices,
)
from creo_model import (
    gate_group_choices,
    load_creo_bundle,
    logistic_logits,
    logistic_probabilities,
    score_group_model,
)
from selector_features import extract_feature_blocks, stable_choice
from selector_model import load_model, score_model


SOURCE_NAMES = ("plurality", "stage2", "stage3_group")
FEATURE_BLOCK_ORDER = (
    "consensus_score", "reliability", "embedding", "arithmetic_grounding", "safety"
)
FEATURE_SETS = {
    "consensus_score": ("consensus_score",),
    "plus_reliability": ("consensus_score", "reliability"),
    "plus_embedding": ("consensus_score", "reliability", "embedding"),
    "plus_arithmetic": (
        "consensus_score", "reliability", "embedding", "arithmetic_grounding"
    ),
    "full": FEATURE_BLOCK_ORDER,
}


def masked_candidate_argmax(scores: np.ndarray, valid: np.ndarray) -> np.ndarray:
    scores = np.asarray(scores, dtype=np.float64)
    valid = np.asarray(valid, dtype=bool)
    result = np.argmax(np.where(valid, scores, -np.inf), axis=1).astype(np.int64)
    result[~valid.any(axis=1)] = 0
    return result


def candidate_groups(choices: np.ndarray, memberships: list[list[list[int]]]) -> np.ndarray:
    result = np.zeros(len(choices), dtype=np.int64)
    for i, choice in enumerate(choices):
        result[i] = next(
            g for g, members in enumerate(memberships[i]) if int(choice) in members
        )
    return result


def _group_argmax(scores: np.ndarray, valid: np.ndarray) -> np.ndarray:
    return np.argmax(np.where(valid, scores, -np.inf), axis=1).astype(np.int64)


def assemble_state(
    bank: list[dict],
    embeddings: np.ndarray,
    blocks: dict,
    candidate_scores: np.ndarray,
    group_data: dict,
    group_scores: np.ndarray,
    reliability: np.ndarray,
    gate_probability: np.ndarray,
    anchor_groups: np.ndarray,
) -> dict:
    plurality = plurality_group_indices(bank, group_data)
    stage2_candidates = masked_candidate_argmax(candidate_scores[:, :, 1], blocks["valid"])
    stage2_groups = candidate_groups(stage2_candidates, group_data["memberships"])
    stage3_groups = _group_argmax(group_scores, group_data["valid"])
    return {
        "bank": bank,
        "embeddings": np.asarray(embeddings, dtype=np.float64),
        "blocks": blocks,
        "candidate_scores": np.asarray(candidate_scores, dtype=np.float64),
        "group_data": group_data,
        "group_scores": np.asarray(group_scores, dtype=np.float64),
        "reliability": np.asarray(reliability, dtype=np.float64),
        "gate_probability": np.asarray(gate_probability, dtype=np.float64),
        "anchor_groups": np.asarray(anchor_groups, dtype=np.int64),
        "plurality_groups": plurality,
        "source_groups": {
            "plurality": plurality,
            "stage2": stage2_groups,
            "stage3_group": stage3_groups,
        },
    }


def build_full_state(bank: list[dict], embeddings: np.ndarray, anchor_checkpoint: Path) -> dict:
    """Run the frozen Stage-3 anchor and expose its intermediate evidence."""
    bundle = load_creo_bundle(Path(anchor_checkpoint))
    config = bundle["config"]
    base_names = config.get("base_checkpoints", ["base_stage1", "base_stage2"])
    if len(base_names) != 2:
        raise ValueError("Stage-5 anchor requires two Stage-3 base checkpoints")
    blocks = extract_feature_blocks(bank, embeddings)
    candidate_scores = np.stack([
        score_model(load_model(Path(anchor_checkpoint) / name), blocks)
        for name in base_names
    ], axis=2)
    group_data = extract_group_features(bank, embeddings, blocks, candidate_scores)
    group_scores = score_group_model(
        bundle["group_model"], group_data["features"], group_data["valid"]
    )
    gate_base = gate_feature_matrices(bank, group_data, group_scores)
    reliability = logistic_probabilities(
        bundle["reliability_model"], gate_base["features"]
    )
    reliability_logit = logistic_logits(
        bundle["reliability_model"], gate_base["features"]
    )
    gate_features = np.column_stack([
        gate_base["features"], reliability, reliability_logit
    ])
    gate_probability = logistic_probabilities(bundle["gate_model"], gate_features)
    anchor_groups = gate_group_choices(
        gate_base["plurality"], gate_base["challenger"], gate_probability,
        float(config["threshold"]),
    )
    return assemble_state(
        bank, embeddings, blocks, candidate_scores, group_data, group_scores,
        reliability, gate_probability, anchor_groups,
    )


def _softmax_rows(scores: np.ndarray, valid: np.ndarray) -> np.ndarray:
    masked = np.where(valid, scores, -np.inf)
    maximum = np.max(masked, axis=1, keepdims=True)
    values = np.where(valid, np.exp(masked - maximum), 0.0)
    return values / np.maximum(values.sum(axis=1, keepdims=True), 1e-300)


def _rank(scores: np.ndarray, valid: np.ndarray, group: int) -> float:
    active = scores[valid]
    return float(1 + np.sum(active > scores[group] + 1e-12)) / 4.0


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(a @ b) / max(float(np.linalg.norm(a) * np.linalg.norm(b)), 1e-12)


def _scalar(group_data: dict, i: int, g: int, name: str) -> float:
    index = group_data["scalar_name_to_index"][name]
    return float(group_data["scalar"][i, g, index])


def _regime_values(supports: list[int]) -> list[float]:
    pattern = "-".join(str(value) for value in sorted(supports, reverse=True))
    known = ("3-1", "2-2", "2-1-1", "1-1-1-1")
    return [float(pattern == name) for name in known] + [float(pattern not in known)]


def _proposal_source_mask(state: dict, i: int, proposal: int) -> np.ndarray:
    return np.asarray([
        float(int(state["source_groups"][name][i]) == proposal) for name in SOURCE_NAMES
    ], dtype=np.float64)


def build_residual_examples(
    state: dict,
    candidate_labels: np.ndarray | None = None,
    allowed_sources: tuple[str, ...] = SOURCE_NAMES,
) -> dict:
    """Create one row per unique proposed answer group, including no-op rows."""
    bank = state["bank"]
    group_data = state["group_data"]
    if "scalar_name_to_index" not in group_data:
        group_data["scalar_name_to_index"] = {
            name: index for index, name in enumerate(group_data["scalar_names"])
        }
    group_probs = _softmax_rows(state["group_scores"], group_data["valid"])
    labels = None
    if candidate_labels is not None:
        labels = group_labels(candidate_labels, group_data["memberships"])

    block_names = {
        "consensus_score": [
            "regime_3_1", "regime_2_2", "regime_2_1_1", "regime_1_1_1_1",
            "regime_other", "proposal_support_minus_anchor", "anchor_support",
            "proposal_support", "anchor_plurality", "proposal_plurality",
            "anchor_singleton", "proposal_singleton", "distinct_group_fraction",
            "vote_margin_fraction", "answer_entropy", "valid_fraction",
            "proposal_source_plurality", "proposal_source_stage2",
            "proposal_source_stage3_group", "proposal_source_count",
            "anchor_source_count", "source_count_difference", "same_as_anchor",
            "stage2_group_score_difference", "stage2_group_max_difference",
            "stage3_group_score_difference", "stage3_group_probability_difference",
            "anchor_stage3_rank", "proposal_stage3_rank", "absolute_stage3_gap",
            "anchor_stage2_rank", "proposal_stage2_rank",
        ],
        "reliability": [
            "plurality_reliability", "anchor_gate_probability",
            "anchor_stage3_probability", "proposal_stage3_probability",
            "proposal_minus_anchor_stage3_probability", "anchor_top1_top2_margin",
            "proposal_minus_best_other_margin", "stage2_probability_sum_difference",
        ],
        "embedding": [
            "anchor_proposal_centroid_cosine", "centroid_euclidean_distance",
            "cohesion_difference", "minimum_cohesion_difference",
            "dispersion_difference", "question_center_cosine_difference",
            "nearest_other_cosine_difference", "centroid_separation_difference",
            "anchor_singleton_embedding", "proposal_singleton_embedding",
            "embedding_peer_missing", "centered_centroid_norm_difference",
        ],
        "arithmetic_grounding": [
            "checked_equations_difference", "mismatch_fraction_difference",
            "max_residual_difference", "final_rhs_agrees_difference",
            "final_rhs_disagrees_difference", "equation_pass_ratio_difference",
            "failed_equation_difference", "equation_produces_final_difference",
            "question_number_reuse_difference", "outside_number_difference",
            "question_token_coverage_difference", "question_unit_coverage_difference",
            "outside_unit_words_difference", "tail_token_coverage_difference",
            "final_answer_near_end_difference", "answer_in_question_difference",
        ],
        "safety": [
            "anchor_group_margin", "absolute_anchor_proposal_gap",
            "anchor_multisource", "proposal_single_source", "anchor_is_plurality",
            "proposal_is_plurality", "anchor_reliability_x_plurality",
            "score_gap_x_entropy", "score_gap_x_vote_margin",
            "proposal_support_gap_x_single_source", "anchor_consensus_retention",
            "branch_disagreement_count",
        ],
    }
    rows = {name: [] for name in FEATURE_BLOCK_ORDER}
    question_indices = []
    proposal_groups = []
    source_masks = []
    outcomes = []
    anchor_correct_rows = []
    proposal_correct_rows = []

    scalar_diff_names = {
        "cohesion_difference": "within_group_mean_cosine",
        "minimum_cohesion_difference": "within_group_min_cosine",
        "dispersion_difference": "embedding_dispersion",
        "question_center_cosine_difference": "centroid_cosine_to_question_center",
        "nearest_other_cosine_difference": "nearest_other_group_centroid_cosine",
        "centroid_separation_difference": "centroid_separation",
    }
    arithmetic_names = [
        "extra_mean_log_checked_equations", "extra_mean_mismatch_fraction",
        "extra_max_log_max_relative_residual", "extra_max_final_rhs_agrees",
        "extra_max_final_rhs_disagrees", "hand_mean_equation_pass_ratio",
        "hand_max_has_failed_equation", "hand_max_equation_produces_final",
        "hand_mean_reused_question_number_ratio", "hand_mean_log_outside_numbers",
        "extra_mean_question_token_coverage", "extra_mean_question_unit_coverage",
        "extra_mean_log_outside_unit_words", "extra_mean_tail_question_token_coverage",
        "hand_mean_final_answer_near_end", "answer_in_question",
    ]

    for i, row in enumerate(bank):
        anchor = int(state["anchor_groups"][i])
        plurality = int(state["plurality_groups"][i])
        proposed = sorted({
            int(state["source_groups"][name][i]) for name in allowed_sources
        })
        valid_groups = np.flatnonzero(group_data["valid"][i]).tolist()
        supports = [int(group_data["support"][i, g]) for g in valid_groups]
        ordered = sorted(supports, reverse=True) + [0]
        valid_count = sum(supports)
        probabilities = np.asarray(supports, dtype=np.float64) / max(valid_count, 1)
        entropy = float(-np.sum(probabilities * np.log(np.maximum(probabilities, 1e-12))))
        vote_margin = (ordered[0] - ordered[1]) / 4.0
        source_group_values = [int(state["source_groups"][name][i]) for name in SOURCE_NAMES]
        anchor_source_count = sum(value == anchor for value in source_group_values)
        branch_disagreement = len(set(source_group_values)) - 1
        active_group_scores = state["group_scores"][i, valid_groups]
        sorted_group_scores = np.sort(active_group_scores)[::-1]
        top_margin = float(sorted_group_scores[0] - sorted_group_scores[1]) if len(sorted_group_scores) > 1 else 0.0
        raw_centroids = [
            state["embeddings"][i, members].mean(axis=0)
            for members in group_data["memberships"][i]
        ]
        stage2_group_mean = np.full(4, -np.inf)
        stage2_group_max = np.full(4, -np.inf)
        stage2_group_prob = np.zeros(4)
        valid_candidates = np.flatnonzero(state["blocks"]["valid"][i]).tolist()
        values = state["candidate_scores"][i, valid_candidates, 1]
        shifted = values - float(values.max()) if values.size else np.asarray([])
        candidate_probs = np.exp(shifted) if values.size else np.asarray([])
        if len(candidate_probs):
            candidate_probs /= candidate_probs.sum()
        probability_by_candidate = dict(zip(valid_candidates, candidate_probs.tolist()))
        for g, members in enumerate(group_data["memberships"][i]):
            member_values = state["candidate_scores"][i, members, 1]
            stage2_group_mean[g] = float(member_values.mean())
            stage2_group_max[g] = float(member_values.max())
            stage2_group_prob[g] = sum(probability_by_candidate.get(j, 0.0) for j in members)

        for proposal in proposed:
            mask = _proposal_source_mask(state, i, proposal)
            proposal_source_count = int(mask.sum())
            anchor_support = int(group_data["support"][i, anchor])
            proposal_support = int(group_data["support"][i, proposal])
            gap = float(state["group_scores"][i, proposal] - state["group_scores"][i, anchor])
            consensus = [
                *_regime_values(supports),
                (proposal_support - anchor_support) / 4.0,
                anchor_support / 4.0, proposal_support / 4.0,
                float(anchor == plurality), float(proposal == plurality),
                float(anchor_support == 1), float(proposal_support == 1),
                len(valid_groups) / 4.0, vote_margin, entropy, valid_count / 4.0,
                *mask.tolist(), proposal_source_count / 3.0,
                anchor_source_count / 3.0,
                (proposal_source_count - anchor_source_count) / 3.0,
                float(proposal == anchor),
                float(stage2_group_mean[proposal] - stage2_group_mean[anchor]),
                float(stage2_group_max[proposal] - stage2_group_max[anchor]),
                gap, float(group_probs[i, proposal] - group_probs[i, anchor]),
                _rank(state["group_scores"][i], group_data["valid"][i], anchor),
                _rank(state["group_scores"][i], group_data["valid"][i], proposal),
                abs(gap),
                _rank(stage2_group_max, group_data["valid"][i], anchor),
                _rank(stage2_group_max, group_data["valid"][i], proposal),
            ]
            best_other = max(
                (float(state["group_scores"][i, g]) for g in valid_groups if g != proposal),
                default=float(state["group_scores"][i, proposal]),
            )
            reliability = [
                float(state["reliability"][i]), float(state["gate_probability"][i]),
                float(group_probs[i, anchor]), float(group_probs[i, proposal]),
                float(group_probs[i, proposal] - group_probs[i, anchor]),
                top_margin,
                float(state["group_scores"][i, proposal] - best_other),
                float(stage2_group_prob[proposal] - stage2_group_prob[anchor]),
            ]
            a_centroid = raw_centroids[anchor]
            p_centroid = raw_centroids[proposal]
            embedding = [
                _cosine(a_centroid, p_centroid),
                math.log1p(float(np.linalg.norm(a_centroid - p_centroid))),
            ]
            embedding.extend([
                _scalar(group_data, i, proposal, source) - _scalar(group_data, i, anchor, source)
                for source in scalar_diff_names.values()
            ])
            embedding.extend([
                _scalar(group_data, i, anchor, "singleton_embedding_group"),
                _scalar(group_data, i, proposal, "singleton_embedding_group"),
                float(anchor == proposal),
                math.log1p(float(np.linalg.norm(group_data["group_embedding"][i, proposal])))
                - math.log1p(float(np.linalg.norm(group_data["group_embedding"][i, anchor]))),
            ])
            arithmetic = [
                _scalar(group_data, i, proposal, name) - _scalar(group_data, i, anchor, name)
                for name in arithmetic_names
            ]
            safety = [
                top_margin, abs(gap), float(anchor_source_count >= 2),
                float(proposal_source_count == 1), float(anchor == plurality),
                float(proposal == plurality),
                float(state["reliability"][i]) * float(anchor == plurality),
                gap * entropy, gap * vote_margin,
                ((proposal_support - anchor_support) / 4.0) * float(proposal_source_count == 1),
                float(anchor == plurality and anchor_support >= 2),
                branch_disagreement / 2.0,
            ]
            values_by_block = {
                "consensus_score": consensus,
                "reliability": reliability,
                "embedding": embedding,
                "arithmetic_grounding": arithmetic,
                "safety": safety,
            }
            for name in FEATURE_BLOCK_ORDER:
                rows[name].append(values_by_block[name])
            question_indices.append(i)
            proposal_groups.append(proposal)
            source_masks.append(mask)
            if labels is not None:
                anchor_correct = bool(labels[i, anchor])
                proposal_correct = bool(labels[i, proposal])
                if proposal != anchor and not anchor_correct and proposal_correct:
                    outcome = 0  # beneficial
                elif proposal != anchor and anchor_correct and not proposal_correct:
                    outcome = 1  # harmful
                else:
                    outcome = 2  # neutral / ineffective
                outcomes.append(outcome)
                anchor_correct_rows.append(anchor_correct)
                proposal_correct_rows.append(proposal_correct)

    block_arrays = {
        name: np.asarray(rows[name], dtype=np.float64) for name in FEATURE_BLOCK_ORDER
    }
    for name, array in block_arrays.items():
        if array.shape[1] != len(block_names[name]) or not np.isfinite(array).all():
            raise ValueError(f"Invalid residual feature block: {name}")
    result = {
        "blocks": block_arrays,
        "block_names": block_names,
        "question_indices": np.asarray(question_indices, dtype=np.int64),
        "proposal_groups": np.asarray(proposal_groups, dtype=np.int64),
        "source_masks": np.asarray(source_masks, dtype=np.float64),
        "anchor_groups": state["anchor_groups"],
    }
    if labels is not None:
        result.update({
            "outcomes": np.asarray(outcomes, dtype=np.int64),
            "anchor_correct": np.asarray(anchor_correct_rows, dtype=bool),
            "proposal_correct": np.asarray(proposal_correct_rows, dtype=bool),
            "group_labels": labels,
        })
    return result


def combine_feature_blocks(examples: dict, feature_set: str) -> tuple[np.ndarray, list[str]]:
    if feature_set not in FEATURE_SETS:
        raise ValueError(f"Unknown residual feature set: {feature_set}")
    selected = FEATURE_SETS[feature_set]
    arrays = [examples["blocks"][name] for name in selected]
    names = [feature for name in selected for feature in examples["block_names"][name]]
    result = np.concatenate(arrays, axis=1)
    if result.shape[1] > 96:
        raise ValueError("Residual scalar feature cap exceeded")
    return result, names


def source_example_mask(examples: dict, allowed_sources: tuple[str, ...]) -> np.ndarray:
    indices = [SOURCE_NAMES.index(name) for name in allowed_sources]
    return examples["source_masks"][:, indices].any(axis=1)


def select_revisions(
    state: dict,
    examples: dict,
    probabilities: np.ndarray,
    allowed_sources: tuple[str, ...],
    harm_cost: float,
    threshold: float,
) -> tuple[np.ndarray, dict]:
    allowed = source_example_mask(examples, allowed_sources)
    utility = probabilities[:, 0] - harm_cost * probabilities[:, 1]
    selected = np.asarray(state["anchor_groups"], dtype=np.int64).copy()
    selected_rows = np.full(len(selected), -1, dtype=np.int64)
    for i, row in enumerate(state["bank"]):
        candidate_rows = np.flatnonzero(
            allowed & (examples["question_indices"] == i)
            & (examples["proposal_groups"] != selected[i])
            & (utility > threshold)
        ).tolist()
        if not candidate_rows:
            continue
        best = max(float(utility[index]) for index in candidate_rows)
        tied_rows = [
            index for index in candidate_rows if abs(float(utility[index]) - best) <= 1e-12
        ]
        if len(tied_rows) > 1:
            texts = [str(candidate.get("text", "")) for candidate in row["candidates"]]
            representatives = [
                stable_choice(str(row.get("question", "")), texts,
                              state["group_data"]["memberships"][i][int(examples["proposal_groups"][index])])
                for index in tied_rows
            ]
            chosen_candidate = stable_choice(str(row.get("question", "")), texts, representatives)
            chosen = next(
                index for index, representative in zip(tied_rows, representatives)
                if representative == chosen_candidate
            )
        else:
            chosen = tied_rows[0]
        selected[i] = int(examples["proposal_groups"][chosen])
        selected_rows[i] = int(chosen)
    return selected, {"utility": utility, "selected_rows": selected_rows, "allowed": allowed}


def groups_to_candidates(state: dict, groups: np.ndarray) -> list[int]:
    return group_to_candidate_choices(
        state["bank"], groups, state["group_data"], state["candidate_scores"]
    )


def content_signature(state: dict, i: int, group: int) -> str:
    answer = str(state["group_data"]["answers"][i][group])
    return hashlib.sha256((str(state["bank"][i].get("question", "")) + "\0" + answer).encode()).hexdigest()
