"""Permutation-equivariant features for Stage 4 CREO-3W.

Only supplied text/embeddings, official parsing, and model scores are used.
No IDs, bank positions, test artifacts, or external resources enter features.
"""
from __future__ import annotations

from fractions import Fraction
import math

import numpy as np

from common import parse_answer
from creo_features import arithmetic_evidence, lexical_unit_evidence
from selector_features import NUMBER_RE, stable_choice


SEED = 20261022
HAND_DIM = 32


def parse_bank_answers(bank: list[dict]) -> list[list[str | None]]:
    return [
        [parse_answer(candidate.get("text", "")) for candidate in row["candidates"]]
        for row in bank
    ]


def build_answer_groups(bank: list[dict], answers: list[list[str | None]] | None = None) -> dict:
    if answers is None:
        answers = parse_bank_answers(bank)
    valid = np.zeros((len(bank), 4), dtype=bool)
    support = np.zeros((len(bank), 4), dtype=np.int64)
    memberships: list[list[list[int]]] = []
    group_answers: list[list[str | None]] = []
    candidate_to_group = np.zeros((len(bank), 4), dtype=np.int64)
    for i, row_answers in enumerate(answers):
        mapping: dict[str, list[int]] = {}
        for j, answer in enumerate(row_answers):
            if answer is not None:
                mapping.setdefault(str(answer), []).append(j)
        if not mapping:
            mapping = {f"__invalid__{j}": [j] for j in range(4)}
        row_members = list(mapping.values())
        row_group_answers = [row_answers[members[0]] for members in row_members]
        memberships.append(row_members)
        group_answers.append(row_group_answers)
        for g, members in enumerate(row_members):
            valid[i, g] = True
            support[i, g] = len(members)
            candidate_to_group[i, members] = g
    return {
        "answers": group_answers,
        "candidate_answers": answers,
        "memberships": memberships,
        "candidate_to_group": candidate_to_group,
        "valid": valid,
        "support": support,
    }


def group_labels(candidate_labels: np.ndarray, groups: dict) -> np.ndarray:
    result = np.zeros((len(candidate_labels), 4), dtype=bool)
    for i, row_groups in enumerate(groups["memberships"]):
        for g, members in enumerate(row_groups):
            result[i, g] = bool(np.any(candidate_labels[i, members]))
        if not result[i].any():
            raise ValueError("Each row must contain a correct answer group")
    return result


def plurality_groups(bank: list[dict], groups: dict) -> np.ndarray:
    choices = np.zeros(len(bank), dtype=np.int64)
    for i, row in enumerate(bank):
        active = np.flatnonzero(groups["valid"][i]).tolist()
        best_support = max(int(groups["support"][i, g]) for g in active)
        tied = [g for g in active if int(groups["support"][i, g]) == best_support]
        if len(tied) == 1:
            choices[i] = tied[0]
            continue
        texts = [str(candidate.get("text", "")) for candidate in row["candidates"]]
        representatives = [
            stable_choice(str(row.get("question", "")), texts, groups["memberships"][i][g])
            for g in tied
        ]
        selected = stable_choice(str(row.get("question", "")), texts, representatives)
        choices[i] = next(g for g in tied if selected in groups["memberships"][i][g])
    return choices


def masked_candidate_choices(bank: list[dict], scores: np.ndarray, valid: np.ndarray) -> list[int]:
    scores = np.asarray(scores, dtype=np.float64)
    result = []
    for i, row in enumerate(bank):
        active = np.flatnonzero(valid[i]).tolist() or list(range(4))
        best = max(float(scores[i, j]) for j in active)
        tied = [j for j in active if abs(float(scores[i, j]) - best) <= 1e-12]
        texts = [str(candidate.get("text", "")) for candidate in row["candidates"]]
        result.append(int(stable_choice(str(row.get("question", "")), texts, tied)))
    return result


def candidate_choices_to_groups(choices: list[int] | np.ndarray, groups: dict) -> np.ndarray:
    choices = np.asarray(choices, dtype=np.int64)
    return groups["candidate_to_group"][np.arange(len(choices)), choices].astype(np.int64)


def group_choices_to_candidates(
    bank: list[dict], group_choices: np.ndarray, groups: dict, candidate_scores: np.ndarray
) -> list[int]:
    result = []
    for i, row in enumerate(bank):
        members = groups["memberships"][i][int(group_choices[i])]
        best = max(float(candidate_scores[i, j]) for j in members)
        tied = [j for j in members if abs(float(candidate_scores[i, j]) - best) <= 1e-12]
        texts = [str(candidate.get("text", "")) for candidate in row["candidates"]]
        result.append(int(stable_choice(str(row.get("question", "")), texts, tied)))
    return result


def centered_embeddings(embeddings: np.ndarray) -> np.ndarray:
    embeddings = np.asarray(embeddings, dtype=np.float64)
    if embeddings.ndim != 3 or embeddings.shape[1:] != (4, 1024):
        raise ValueError("Expected embeddings [N,4,1024]")
    return embeddings - embeddings.mean(axis=1, keepdims=True)


def initialize_projection(input_dim: int, projection_dim: int, seed: int = SEED) -> np.ndarray:
    rng = np.random.default_rng(seed + projection_dim)
    raw = rng.normal(size=(input_dim, projection_dim))
    q, _ = np.linalg.qr(raw)
    return q[:, :projection_dim].astype(np.float64)


def project(centered: np.ndarray, projection: np.ndarray) -> np.ndarray:
    return np.einsum("nkd,dr->nkr", np.asarray(centered, dtype=np.float64), projection)


def _cosine_and_grads(a: np.ndarray, b: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    na = float(np.linalg.norm(a)); nb = float(np.linalg.norm(b))
    if na < 1e-12 or nb < 1e-12:
        return 0.0, np.zeros_like(a), np.zeros_like(b)
    cosine = float(a @ b) / (na * nb)
    grad_a = b / (na * nb) - cosine * a / (na * na)
    grad_b = a / (na * nb) - cosine * b / (nb * nb)
    return cosine, grad_a, grad_b


def interaction_feature_dim(projection_dim: int, mode: str = "interaction") -> int:
    if mode == "linear_lowrank":
        return HAND_DIM + projection_dim
    if mode == "interaction":
        return HAND_DIM + 7 * projection_dim + 4
    raise ValueError(f"Unknown candidate feature mode: {mode}")


def candidate_features_from_projected(
    projected: np.ndarray,
    answers: list[list[str | None]],
    stage2_hand: np.ndarray,
    mode: str = "interaction",
) -> np.ndarray:
    z = np.asarray(projected, dtype=np.float64)
    hand = np.asarray(stage2_hand, dtype=np.float64)
    if mode == "linear_lowrank":
        return np.concatenate([hand, z], axis=2)
    if mode != "interaction":
        raise ValueError(f"Unknown candidate feature mode: {mode}")
    n, _, d = z.shape
    result = np.zeros((n, 4, interaction_feature_dim(d, mode)), dtype=np.float64)
    for i in range(n):
        mean = z[i].mean(axis=0)
        for j in range(4):
            answer = answers[i][j]
            same = [k for k in range(4) if k != j and answer is not None and answers[i][k] == answer]
            different = [
                k for k in range(4)
                if k != j and answers[i][k] is not None and answers[i][k] != answer
            ]
            same_mean = z[i, same].mean(axis=0) if same else np.zeros(d)
            diff_mean = z[i, different].mean(axis=0) if different else np.zeros(d)
            delta = z[i, j] - mean
            cos_same = _cosine_and_grads(z[i, j], same_mean)[0]
            cos_diff = _cosine_and_grads(z[i, j], diff_mean)[0]
            result[i, j] = np.concatenate([
                hand[i, j], z[i, j], mean, delta, np.abs(delta), z[i, j] * mean,
                same_mean, diff_mean,
                np.asarray([cos_same, cos_diff, float(bool(same)), float(bool(different))]),
            ])
    return result


def candidate_feature_backward(
    upstream: np.ndarray,
    projected: np.ndarray,
    answers: list[list[str | None]],
    mode: str = "interaction",
) -> np.ndarray:
    """Backpropagate candidate-feature gradients to projected embeddings."""
    upstream = np.asarray(upstream, dtype=np.float64)
    z = np.asarray(projected, dtype=np.float64)
    n, _, d = z.shape
    grad_z = np.zeros_like(z)
    if mode == "linear_lowrank":
        grad_z += upstream[:, :, HAND_DIM:HAND_DIM + d]
        return grad_z
    if mode != "interaction":
        raise ValueError(f"Unknown candidate feature mode: {mode}")
    for i in range(n):
        mean = z[i].mean(axis=0)
        grad_mean = np.zeros(d)
        same_accum = [np.zeros(d) for _ in range(4)]
        diff_accum = [np.zeros(d) for _ in range(4)]
        for j in range(4):
            cursor = HAND_DIM
            gz = upstream[i, j, cursor:cursor + d].copy(); cursor += d
            gm = upstream[i, j, cursor:cursor + d].copy(); cursor += d
            gd = upstream[i, j, cursor:cursor + d]; cursor += d
            ga = upstream[i, j, cursor:cursor + d]; cursor += d
            gp = upstream[i, j, cursor:cursor + d]; cursor += d
            gs = upstream[i, j, cursor:cursor + d].copy(); cursor += d
            go = upstream[i, j, cursor:cursor + d].copy(); cursor += d
            scalars = upstream[i, j, cursor:cursor + 4]
            answer = answers[i][j]
            same = [k for k in range(4) if k != j and answer is not None and answers[i][k] == answer]
            different = [
                k for k in range(4)
                if k != j and answers[i][k] is not None and answers[i][k] != answer
            ]
            same_mean = z[i, same].mean(axis=0) if same else np.zeros(d)
            diff_mean = z[i, different].mean(axis=0) if different else np.zeros(d)
            delta = z[i, j] - mean
            signed = ga * np.sign(delta)
            gz += gd + signed + gp * mean
            gm += -gd - signed + gp * z[i, j]
            _, cos_gz, cos_gs = _cosine_and_grads(z[i, j], same_mean)
            gz += scalars[0] * cos_gz; gs += scalars[0] * cos_gs
            _, cos_gz, cos_go = _cosine_and_grads(z[i, j], diff_mean)
            gz += scalars[1] * cos_gz; go += scalars[1] * cos_go
            grad_z[i, j] += gz
            grad_mean += gm
            if same:
                share = gs / len(same)
                for k in same:
                    same_accum[k] += share
            if different:
                share = go / len(different)
                for k in different:
                    diff_accum[k] += share
        grad_z[i] += grad_mean[None, :] / 4.0
        for j in range(4):
            grad_z[i, j] += same_accum[j] + diff_accum[j]
    return grad_z


GROUP_BASE_NAMES = (
    "support_fraction", "plurality_member",
    "candidate_score_mean", "candidate_score_max", "candidate_score_min", "candidate_score_std",
    "projected_dispersion", "within_group_cohesion", "nearest_group_cosine",
)
GROUP_EXTRA_NAMES = tuple(
    [f"arithmetic_mean_{i}" for i in range(7)]
    + [f"arithmetic_max_{i}" for i in range(7)]
    + [f"grounding_mean_{i}" for i in range(4)]
    + [f"grounding_max_{i}" for i in range(4)]
    + ["answer_sign", "answer_log_abs", "answer_integer", "answer_in_question"]
    + ["singleton_group", "has_other_group"]
)


def _answer_numeric(answer: str | None, question: str) -> np.ndarray:
    try:
        value = Fraction(answer) if answer is not None else None
    except (ValueError, ZeroDivisionError):
        value = None
    if value is None:
        return np.zeros(4, dtype=np.float64)
    question_values = set()
    for token in NUMBER_RE.findall(question):
        try:
            question_values.add(Fraction(token.replace(",", "")))
        except (ValueError, ZeroDivisionError):
            pass
    return np.asarray([
        float(value > 0) - float(value < 0),
        math.log1p(float(min(abs(value), Fraction(10**12)))),
        float(value.denominator == 1),
        float(value in question_values),
    ])


def extract_group_features(
    bank: list[dict],
    groups: dict,
    projected: np.ndarray,
    candidate_scores: np.ndarray,
) -> dict:
    projected = np.asarray(projected, dtype=np.float64)
    candidate_scores = np.asarray(candidate_scores, dtype=np.float64)
    n, _, d = projected.shape
    scalar_dim = len(GROUP_BASE_NAMES) + len(GROUP_EXTRA_NAMES)
    scalar = np.zeros((n, 4, scalar_dim), dtype=np.float64)
    projected_mean = np.zeros((n, 4, d), dtype=np.float64)
    for i, row in enumerate(bank):
        question = str(row.get("question", ""))
        texts = [str(candidate.get("text", "")) for candidate in row["candidates"]]
        candidate_answers = groups["candidate_answers"][i]
        extras = np.asarray([
            np.concatenate([
                arithmetic_evidence(texts[j], candidate_answers[j]),
                lexical_unit_evidence(question, texts[j]),
            ])
            for j in range(4)
        ])
        active_groups = np.flatnonzero(groups["valid"][i]).tolist()
        supports = [int(groups["support"][i, g]) for g in active_groups]
        max_support = max(supports)
        centroids = np.asarray([
            projected[i, groups["memberships"][i][g]].mean(axis=0) for g in active_groups
        ])
        norms = np.linalg.norm(centroids, axis=1, keepdims=True)
        normalized = centroids / np.maximum(norms, 1e-12)
        centroid_cos = normalized @ normalized.T
        for local, g in enumerate(active_groups):
            members = groups["memberships"][i][g]
            values = candidate_scores[i, members]
            centroid = centroids[local]
            projected_mean[i, g] = centroid
            dispersion = float(np.mean(np.linalg.norm(projected[i, members] - centroid, axis=1)))
            if len(members) > 1:
                member_norm = projected[i, members] / np.maximum(
                    np.linalg.norm(projected[i, members], axis=1, keepdims=True), 1e-12
                )
                pair = member_norm @ member_norm.T
                cohesion = float(pair[np.triu_indices(len(members), 1)].mean())
            else:
                cohesion = 0.0
            other = [k for k in range(len(active_groups)) if k != local]
            nearest = max((float(centroid_cos[local, k]) for k in other), default=0.0)
            extra = extras[members]
            arithmetic = extra[:, :7]
            grounding = extra[:, 7:]
            scalar[i, g] = np.concatenate([
                np.asarray([
                    len(members) / 4.0,
                    float(len(members) == max_support),
                    float(values.mean()), float(values.max()), float(values.min()), float(values.std()),
                    dispersion, cohesion, nearest,
                ]),
                arithmetic.mean(axis=0), arithmetic.max(axis=0),
                grounding.mean(axis=0), grounding.max(axis=0),
                _answer_numeric(groups["answers"][i][g], question),
                np.asarray([float(len(members) == 1), float(bool(other))]),
            ])
    return {
        "scalar": scalar,
        "projected": projected_mean,
        "features": np.concatenate([scalar, projected_mean], axis=2),
        "scalar_dim": scalar_dim,
        "scalar_names": list(GROUP_BASE_NAMES + GROUP_EXTRA_NAMES),
        "valid": groups["valid"],
        "memberships": groups["memberships"],
        "support": groups["support"],
    }


def group_argmax(scores: np.ndarray, valid: np.ndarray) -> np.ndarray:
    masked = np.where(valid, scores, -np.inf)
    result = np.argmax(masked, axis=1).astype(np.int64)
    result[~valid.any(axis=1)] = 0
    return result


def score_margin(scores: np.ndarray, valid: np.ndarray) -> np.ndarray:
    result = np.zeros(len(scores), dtype=np.float64)
    for i in range(len(scores)):
        values = np.sort(np.asarray(scores[i, valid[i]], dtype=np.float64))
        result[i] = values[-1] - values[-2] if len(values) > 1 else 0.0
    return result


def consensus_features(
    bank: list[dict],
    groups: dict,
    group_features: dict,
    plurality: np.ndarray,
    candidate_group: np.ndarray,
    learned_group: np.ndarray,
    candidate_scores: np.ndarray,
    learned_group_scores: np.ndarray,
    hard_mask: np.ndarray,
) -> tuple[np.ndarray, list[str]]:
    candidate_valid = np.asarray([
        [answer is not None for answer in row] for row in groups["candidate_answers"]
    ], dtype=bool)
    candidate_margin = score_margin(candidate_scores, candidate_valid)
    learned_margin = score_margin(learned_group_scores, groups["valid"])
    names = [
        "pattern_3_1", "pattern_2_2", "pattern_2_1_1", "pattern_1_1_1_1", "pattern_other",
        "vote_margin", "answer_entropy", "candidate_margin", "group_margin",
        "candidate_plurality_agree", "group_plurality_agree", "candidate_group_agree",
        "plurality_support", "candidate_support", "group_support",
        "candidate_group_score_gap", "candidate_plurality_score_gap",
        "plurality_arithmetic_mismatch", "candidate_arithmetic_mismatch", "group_arithmetic_mismatch",
        "plurality_grounding", "candidate_grounding", "group_grounding", "hard_indicator",
    ]
    mismatch_index = group_features["scalar_names"].index("arithmetic_mean_2")
    grounding_index = group_features["scalar_names"].index("grounding_mean_0")
    rows = []
    for i in range(len(bank)):
        active = np.flatnonzero(groups["valid"][i]).tolist()
        supports = sorted([int(groups["support"][i, g]) for g in active], reverse=True)
        pattern = "-".join(str(x) for x in supports)
        pattern_values = [
            float(pattern == "3-1"), float(pattern == "2-2"), float(pattern == "2-1-1"),
            float(pattern == "1-1-1-1"),
            float(pattern not in {"3-1", "2-2", "2-1-1", "1-1-1-1"}),
        ]
        ordered = supports + [0]
        vote_margin = (ordered[0] - ordered[1]) / 4.0
        probabilities = np.asarray(supports, dtype=float) / max(sum(supports), 1)
        entropy = float(-np.sum(probabilities * np.log(np.maximum(probabilities, 1e-12))))
        p, c, g = int(plurality[i]), int(candidate_group[i]), int(learned_group[i])
        candidate_group_score = float(learned_group_scores[i, c])
        rows.append(pattern_values + [
            vote_margin, entropy, float(candidate_margin[i]), float(learned_margin[i]),
            float(c == p), float(g == p), float(c == g),
            groups["support"][i, p] / 4.0, groups["support"][i, c] / 4.0,
            groups["support"][i, g] / 4.0,
            float(learned_group_scores[i, g] - candidate_group_score),
            float(candidate_group_score - learned_group_scores[i, p]),
            float(group_features["scalar"][i, p, mismatch_index]),
            float(group_features["scalar"][i, c, mismatch_index]),
            float(group_features["scalar"][i, g, mismatch_index]),
            float(group_features["scalar"][i, p, grounding_index]),
            float(group_features["scalar"][i, c, grounding_index]),
            float(group_features["scalar"][i, g, grounding_index]),
            float(hard_mask[i]),
        ])
    return np.asarray(rows, dtype=np.float64), names


def merge_branch_probabilities(branch_groups: np.ndarray, branch_probabilities: np.ndarray) -> np.ndarray:
    """Merge duplicate branch choices and return the highest-mass answer group."""
    selected = np.zeros(len(branch_groups), dtype=np.int64)
    for i in range(len(branch_groups)):
        masses: dict[int, float] = {}
        for branch in range(3):
            group = int(branch_groups[i, branch])
            masses[group] = masses.get(group, 0.0) + float(branch_probabilities[i, branch])
        selected[i] = max(sorted(masses), key=lambda group: masses[group])
    return selected
