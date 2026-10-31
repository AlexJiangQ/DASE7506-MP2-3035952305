"""Answer-group and consensus-reliability features for Stage 3 CREO.

The module uses only the supplied question/candidate text, supplied embeddings,
official parsing, and frozen OOF/base scores. It is candidate-permutation
equivariant and does not use IDs or sample positions.
"""
from __future__ import annotations

import ast
from collections import Counter
from fractions import Fraction
import math
import re

import numpy as np

from common import parse_answer
from selector_features import NUMBER_RE, stable_choice


HAND_INDICES = (1, 2, 7, 9, 10, 19, 20, 21, 22, 23, 26, 27, 28)
HAND_NAMES = (
    "same_answer_count", "plurality_member", "log_chars", "log_numeric_tokens",
    "log_equals_markers", "log_reused_question_numbers", "reused_question_number_ratio",
    "log_outside_numbers", "final_answer_near_end", "log_final_answer_occurrences",
    "equation_pass_ratio", "has_failed_equation", "equation_produces_final",
)
WORD_RE = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?")
NUMERIC_EXPR_END_RE = re.compile(r"([0-9.()+\-*/\s]+)$")
NUMERIC_EXPR_START_RE = re.compile(r"^([0-9.()+\-*/\s]+)")
UNIT_WORDS = {
    "second", "seconds", "minute", "minutes", "hour", "hours", "day", "days",
    "week", "weeks", "month", "months", "year", "years", "inch", "inches",
    "foot", "feet", "yard", "yards", "mile", "miles", "meter", "meters",
    "kilometer", "kilometers", "gram", "grams", "kilogram", "kilograms",
    "ounce", "ounces", "pound", "pounds", "liter", "liters", "dollar",
    "dollars", "cent", "cents", "percent", "percentage", "mph", "kmh",
}
STOP_WORDS = {
    "the", "and", "that", "this", "with", "from", "for", "are", "was", "were",
    "has", "have", "had", "how", "what", "when", "where", "who", "does", "did",
    "into", "than", "then", "each", "there", "their",
}


def _fraction(token: str | None) -> Fraction | None:
    if token is None:
        return None
    try:
        return Fraction(str(token).replace(",", ""))
    except (ValueError, ZeroDivisionError):
        return None


def _eval_expr_node(node: ast.AST, depth: int = 0) -> Fraction:
    if depth > 10:
        raise ValueError("expression too deep")
    if isinstance(node, ast.Expression):
        return _eval_expr_node(node.body, depth + 1)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        value = Fraction(str(node.value))
        if abs(value) > 10**12:
            raise ValueError("magnitude")
        return value
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        value = _eval_expr_node(node.operand, depth + 1)
        return value if isinstance(node.op, ast.UAdd) else -value
    if isinstance(node, ast.BinOp):
        left = _eval_expr_node(node.left, depth + 1)
        right = _eval_expr_node(node.right, depth + 1)
        if isinstance(node.op, ast.Add):
            value = left + right
        elif isinstance(node.op, ast.Sub):
            value = left - right
        elif isinstance(node.op, ast.Mult):
            value = left * right
        elif isinstance(node.op, ast.Div):
            if right == 0:
                raise ValueError("division by zero")
            value = left / right
        elif isinstance(node.op, ast.Pow):
            if right.denominator != 1 or abs(right.numerator) > 6:
                raise ValueError("unsupported exponent")
            value = left ** int(right)
        else:
            raise ValueError("unsupported operator")
        if abs(value) > 10**12 or abs(value.numerator) > 10**18 or value.denominator > 10**18:
            raise ValueError("magnitude")
        return value
    raise ValueError("unsupported expression")


def _safe_eval_expr(text: str) -> Fraction | None:
    text = text.strip()
    if not text or len(text) > 160:
        return None
    try:
        tree = ast.parse(text, mode="eval")
        if sum(1 for _ in ast.walk(tree)) > 64:
            return None
        return _eval_expr_node(tree)
    except (SyntaxError, ValueError, ZeroDivisionError, OverflowError):
        return None


def _normalize_math(text: str) -> str:
    return (
        text.replace("\\times", "*").replace("\\cdot", "*")
        .replace("×", "*").replace("÷", "/").replace("−", "-")
        .replace("^", "**").replace(",", "")
    )


def arithmetic_evidence(text: str, final_answer: str | None) -> np.ndarray:
    checked = mismatches = 0
    max_relative = 0.0
    last_rhs: Fraction | None = None
    normalized = _normalize_math(text[:20000])
    for line in normalized.splitlines():
        if "=" not in line:
            continue
        parts = line.split("=")
        for left_raw, right_raw in zip(parts[:-1], parts[1:]):
            left_match = NUMERIC_EXPR_END_RE.search(left_raw)
            right_match = NUMERIC_EXPR_START_RE.search(right_raw)
            if left_match is None or right_match is None:
                continue
            left = _safe_eval_expr(left_match.group(1))
            right = _safe_eval_expr(right_match.group(1))
            if left is None or right is None:
                continue
            checked += 1
            last_rhs = right
            difference = abs(left - right)
            scale = max(abs(left), abs(right), Fraction(1))
            relative = float(difference / scale)
            max_relative = max(max_relative, relative)
            tolerance = Fraction(5, 1000) + Fraction(1, 1_000_000) * scale
            if difference > tolerance:
                mismatches += 1
            if checked >= 64:
                break
        if checked >= 64:
            break
    final_value = _fraction(final_answer)
    comparison_available = last_rhs is not None and final_value is not None
    agrees = bool(comparison_available and last_rhs == final_value)
    return np.asarray([
        math.log1p(checked),
        math.log1p(mismatches),
        mismatches / checked if checked else 0.0,
        math.log1p(min(max_relative, 1e6)),
        float(comparison_available),
        float(agrees),
        float(comparison_available and not agrees),
    ], dtype=np.float64)


ARITHMETIC_NAMES = (
    "log_checked_equations", "log_mismatched_equations", "mismatch_fraction",
    "log_max_relative_residual", "final_rhs_comparison_available",
    "final_rhs_agrees", "final_rhs_disagrees",
)


def _word_set(text: str) -> set[str]:
    return {
        token.lower() for token in WORD_RE.findall(text)
        if len(token) >= 3 and token.lower() not in STOP_WORDS
    }


def lexical_unit_evidence(question: str, text: str) -> np.ndarray:
    q_words = _word_set(question)
    c_words = _word_set(text)
    q_units = q_words & UNIT_WORDS
    c_units = c_words & UNIT_WORDS
    tail_words = _word_set(text[-240:])
    return np.asarray([
        len(c_words & q_words) / max(len(q_words), 1),
        len(c_words & q_units) / max(len(q_units), 1),
        math.log1p(len(c_units - q_units)),
        len(tail_words & q_words) / max(len(q_words), 1),
    ], dtype=np.float64)


LEXICAL_UNIT_NAMES = (
    "question_token_coverage", "question_unit_coverage",
    "log_outside_unit_words", "tail_question_token_coverage",
)


def _answer_numeric_features(answer: str | None, question: str) -> np.ndarray:
    value = _fraction(answer)
    if value is None:
        return np.zeros(4, dtype=np.float64)
    clipped = min(abs(value), Fraction(10**12))
    question_values = {_fraction(token) for token in NUMBER_RE.findall(question)}
    question_values.discard(None)
    return np.asarray([
        float(value > 0) - float(value < 0),
        math.log1p(float(clipped)),
        float(value.denominator == 1),
        float(value in question_values),
    ], dtype=np.float64)


ANSWER_NUMERIC_NAMES = ("answer_sign", "log_abs_answer", "answer_is_integer", "answer_in_question")


def _cosine_rows(array: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(array, axis=1, keepdims=True)
    normalized = array / np.maximum(norms, 1e-12)
    return normalized @ normalized.T


GLOBAL_NAMES = (
    "support_fraction", "plurality_member", "distinct_answer_fraction",
    "max_support_fraction", "vote_margin_fraction", "valid_fraction",
    "answer_entropy", "singleton_group",
)
EMBEDDING_NAMES = (
    "within_group_mean_cosine", "within_group_min_cosine", "embedding_dispersion",
    "centroid_cosine_to_question_center", "nearest_other_group_centroid_cosine",
    "centroid_separation", "singleton_embedding_group",
)


def scalar_feature_names(model_count: int) -> list[str]:
    names = list(GLOBAL_NAMES)
    for model_no in range(model_count):
        names.extend([
            f"base{model_no}_z_mean", f"base{model_no}_z_max", f"base{model_no}_z_min",
            f"base{model_no}_z_std", f"base{model_no}_probability_sum",
            f"base{model_no}_probability_max",
        ])
    names.extend(EMBEDDING_NAMES)
    for name in HAND_NAMES:
        names.extend([f"hand_mean_{name}", f"hand_max_{name}"])
    for name in ARITHMETIC_NAMES + LEXICAL_UNIT_NAMES:
        names.extend([f"extra_mean_{name}", f"extra_max_{name}"])
    names.extend(ANSWER_NUMERIC_NAMES)
    return names


def extract_group_features(
    bank: list[dict],
    embeddings: np.ndarray,
    candidate_blocks: dict,
    candidate_scores: np.ndarray,
) -> dict:
    """Return padded answer-group features and candidate membership mappings."""
    embeddings = np.asarray(embeddings, dtype=np.float64)
    candidate_scores = np.asarray(candidate_scores, dtype=np.float64)
    if embeddings.shape[:2] != (len(bank), 4) or embeddings.shape[2] != 1024:
        raise ValueError("Expected embeddings [N,4,1024]")
    if candidate_scores.ndim != 3 or candidate_scores.shape[:2] != (len(bank), 4):
        raise ValueError("Expected candidate_scores [N,4,M]")
    model_count = candidate_scores.shape[2]
    names = scalar_feature_names(model_count)
    scalar = np.zeros((len(bank), 4, len(names)), dtype=np.float64)
    group_embedding = np.zeros((len(bank), 4, 1024), dtype=np.float64)
    group_valid = np.zeros((len(bank), 4), dtype=bool)
    group_support = np.zeros((len(bank), 4), dtype=np.int64)
    memberships: list[list[list[int]]] = []
    group_answers: list[list[str | None]] = []

    centered = embeddings - embeddings.mean(axis=1, keepdims=True)
    hand = np.asarray(candidate_blocks["hand"], dtype=np.float64)
    for i, row in enumerate(bank):
        question = str(row.get("question", ""))
        texts = [str(candidate.get("text", "")) for candidate in row["candidates"]]
        answers = [parse_answer(text) for text in texts]
        valid_candidates = [j for j, answer in enumerate(answers) if answer is not None]
        answer_to_members: dict[str, list[int]] = {}
        for j in valid_candidates:
            answer_to_members.setdefault(str(answers[j]), []).append(j)
        if not answer_to_members:
            # All-invalid audit inputs retain four singleton fallback groups.
            answer_to_members = {f"__invalid__{j}": [j] for j in range(4)}
        row_members = list(answer_to_members.values())
        row_answers = [answers[members[0]] for members in row_members]
        memberships.append(row_members)
        group_answers.append(row_answers)
        supports = [len(members) for members in row_members]
        max_support = max(supports)
        ordered = sorted(supports, reverse=True)
        second_support = ordered[1] if len(ordered) > 1 else 0
        total_valid = max(len(valid_candidates), 1)
        probabilities = np.asarray(supports, dtype=np.float64) / total_valid
        entropy = float(-np.sum(probabilities * np.log(np.maximum(probabilities, 1e-12))))

        score_z = np.zeros_like(candidate_scores[i])
        score_prob = np.zeros_like(candidate_scores[i])
        active = valid_candidates if valid_candidates else list(range(4))
        for m in range(model_count):
            values = candidate_scores[i, active, m]
            mean = float(values.mean())
            std = float(values.std())
            score_z[:, m] = (candidate_scores[i, :, m] - mean) / max(std, 1e-8)
            shifted = values - float(values.max())
            probs = np.exp(shifted)
            probs /= probs.sum()
            score_prob[active, m] = probs

        candidate_extra = np.asarray([
            np.concatenate([
                arithmetic_evidence(texts[j], answers[j]),
                lexical_unit_evidence(question, texts[j]),
            ])
            for j in range(4)
        ])
        centroid_vectors = []
        for members in row_members:
            centroid_vectors.append(embeddings[i, members].mean(axis=0))
        centroid_vectors_array = np.asarray(centroid_vectors)
        centroid_norm = centroid_vectors_array / np.maximum(
            np.linalg.norm(centroid_vectors_array, axis=1, keepdims=True), 1e-12
        )
        centroid_cosines = centroid_norm @ centroid_norm.T
        question_center = embeddings[i].mean(axis=0)
        question_center_norm = max(float(np.linalg.norm(question_center)), 1e-12)

        for g, members in enumerate(row_members):
            group_valid[i, g] = True
            group_support[i, g] = len(members)
            group_embedding[i, g] = centered[i, members].mean(axis=0)
            member_embeddings = embeddings[i, members]
            if len(members) > 1:
                pair_cos = _cosine_rows(member_embeddings)
                off_diagonal = pair_cos[np.triu_indices(len(members), 1)]
                mean_cos = float(off_diagonal.mean())
                min_cos = float(off_diagonal.min())
            else:
                mean_cos = min_cos = 0.0
            centroid = centroid_vectors_array[g]
            dispersion = float(np.mean(np.linalg.norm(member_embeddings - centroid, axis=1)))
            center_cos = float(centroid @ question_center) / max(
                float(np.linalg.norm(centroid)) * question_center_norm, 1e-12
            )
            other = [k for k in range(len(row_members)) if k != g]
            nearest_other = max((float(centroid_cosines[g, k]) for k in other), default=0.0)

            values: list[float] = [
                len(members) / 4.0,
                float(len(members) == max_support),
                len(row_members) / 4.0,
                max_support / 4.0,
                (max_support - second_support) / 4.0,
                len(valid_candidates) / 4.0,
                entropy,
                float(len(members) == 1),
            ]
            for m in range(model_count):
                z = score_z[members, m]
                p = score_prob[members, m]
                values.extend([
                    float(z.mean()), float(z.max()), float(z.min()), float(z.std()),
                    float(p.sum()), float(p.max()),
                ])
            values.extend([
                mean_cos, min_cos, dispersion, center_cos, nearest_other,
                1.0 - nearest_other, float(len(members) == 1),
            ])
            for hand_index in HAND_INDICES:
                selected = hand[i, members, hand_index]
                values.extend([float(selected.mean()), float(selected.max())])
            for feature_index in range(candidate_extra.shape[1]):
                selected = candidate_extra[members, feature_index]
                values.extend([float(selected.mean()), float(selected.max())])
            values.extend(_answer_numeric_features(row_answers[g], question).tolist())
            scalar[i, g] = values

    return {
        "scalar": scalar,
        "group_embedding": group_embedding,
        "features": np.concatenate([scalar, group_embedding], axis=2),
        "scalar_names": names,
        "valid": group_valid,
        "support": group_support,
        "memberships": memberships,
        "answers": group_answers,
    }


def group_labels(candidate_labels: np.ndarray, memberships: list[list[list[int]]]) -> np.ndarray:
    labels = np.zeros((len(memberships), 4), dtype=bool)
    for i, row_groups in enumerate(memberships):
        for g, members in enumerate(row_groups):
            labels[i, g] = bool(np.any(candidate_labels[i, members]))
        if not labels[i].any():
            raise ValueError("Every group row needs a correct answer group")
    return labels


def plurality_group_indices(bank: list[dict], group_data: dict) -> np.ndarray:
    result = np.zeros(len(bank), dtype=np.int64)
    for i, row in enumerate(bank):
        valid_groups = np.flatnonzero(group_data["valid"][i]).tolist()
        max_support = max(int(group_data["support"][i, g]) for g in valid_groups)
        tied_groups = [g for g in valid_groups if int(group_data["support"][i, g]) == max_support]
        if len(tied_groups) == 1:
            result[i] = tied_groups[0]
            continue
        texts = [str(candidate.get("text", "")) for candidate in row["candidates"]]
        representatives = [
            stable_choice(str(row.get("question", "")), texts, group_data["memberships"][i][g])
            for g in tied_groups
        ]
        chosen_candidate = stable_choice(str(row.get("question", "")), texts, representatives)
        result[i] = next(g for g in tied_groups if chosen_candidate in group_data["memberships"][i][g])
    return result


def group_to_candidate_choices(
    bank: list[dict],
    group_indices: np.ndarray,
    group_data: dict,
    candidate_scores: np.ndarray,
) -> list[int]:
    choices = []
    candidate_scores = np.asarray(candidate_scores)
    if candidate_scores.ndim == 3:
        candidate_scores = candidate_scores[:, :, -1]
    for i, row in enumerate(bank):
        g = int(group_indices[i])
        members = group_data["memberships"][i][g]
        best = max(float(candidate_scores[i, j]) for j in members)
        tied = [j for j in members if abs(float(candidate_scores[i, j]) - best) <= 1e-12]
        texts = [str(candidate.get("text", "")) for candidate in row["candidates"]]
        choices.append(int(stable_choice(str(row.get("question", "")), texts, tied)))
    return choices


def group_argmax_choices(
    bank: list[dict], group_scores: np.ndarray, group_data: dict, candidate_scores: np.ndarray
) -> tuple[np.ndarray, list[int]]:
    selected_groups = np.zeros(len(bank), dtype=np.int64)
    for i, row in enumerate(bank):
        valid_groups = np.flatnonzero(group_data["valid"][i]).tolist()
        best = max(float(group_scores[i, g]) for g in valid_groups)
        tied_groups = [g for g in valid_groups if abs(float(group_scores[i, g]) - best) <= 1e-12]
        if len(tied_groups) == 1:
            selected_groups[i] = tied_groups[0]
        else:
            texts = [str(candidate.get("text", "")) for candidate in row["candidates"]]
            representatives = [
                stable_choice(str(row.get("question", "")), texts, group_data["memberships"][i][g])
                for g in tied_groups
            ]
            chosen = stable_choice(str(row.get("question", "")), texts, representatives)
            selected_groups[i] = next(
                g for g in tied_groups if chosen in group_data["memberships"][i][g]
            )
    return selected_groups, group_to_candidate_choices(
        bank, selected_groups, group_data, candidate_scores
    )


def shrinkage_pool_choices(
    bank: list[dict], candidate_scores: np.ndarray, group_data: dict, alpha: float
) -> list[int]:
    scores = np.asarray(candidate_scores, dtype=np.float64)
    if scores.ndim == 3:
        scores = scores[:, :, -1]
    pooled = scores.copy()
    valid_candidates = np.zeros_like(scores, dtype=bool)
    for i, row_groups in enumerate(group_data["memberships"]):
        for members in row_groups:
            mean = float(scores[i, members].mean())
            pooled[i, members] = (1.0 - alpha) * scores[i, members] + alpha * mean
            valid_candidates[i, members] = True
    pooled = np.where(valid_candidates, pooled, -np.inf)
    choices = []
    for i, row in enumerate(bank):
        best = float(np.max(pooled[i]))
        tied = np.flatnonzero(np.abs(pooled[i] - best) <= 1e-12).tolist()
        texts = [str(candidate.get("text", "")) for candidate in row["candidates"]]
        choices.append(int(stable_choice(str(row.get("question", "")), texts, tied)))
    return choices


def gate_feature_matrices(
    bank: list[dict],
    group_data: dict,
    group_scores: np.ndarray,
    group_label_array: np.ndarray | None = None,
    challenger_indices: np.ndarray | None = None,
) -> dict:
    names = list(group_data["scalar_names"])
    name_to_index = {name: index for index, name in enumerate(names)}
    plurality = plurality_group_indices(bank, group_data)
    challenger = np.zeros(len(bank), dtype=np.int64)
    selected_scalar_names = [
        "support_fraction", "answer_entropy", "base0_z_mean", "base0_z_max",
        "base0_probability_sum", "base1_z_mean", "base1_z_max", "base1_probability_sum",
        "within_group_mean_cosine", "within_group_min_cosine", "embedding_dispersion",
        "centroid_separation", "hand_mean_equation_pass_ratio", "hand_max_has_failed_equation",
        "hand_max_equation_produces_final", "extra_mean_mismatch_fraction",
        "extra_max_log_max_relative_residual", "extra_max_final_rhs_agrees",
        "extra_max_final_rhs_disagrees", "extra_mean_question_token_coverage",
        "extra_mean_question_unit_coverage", "extra_mean_log_outside_unit_words",
        "extra_mean_tail_question_token_coverage",
    ]
    missing = [name for name in selected_scalar_names if name not in name_to_index]
    if missing:
        raise KeyError(f"Missing gate feature names: {missing}")
    feature_rows = []
    feature_names = [
        "pattern_3_1", "pattern_2_2", "pattern_2_1_1", "pattern_1_1_1_1", "pattern_other",
        "distinct_group_fraction", "vote_margin_fraction", "answer_entropy",
        "group_score_challenger_minus_plurality", "group_score_plurality",
        "group_score_challenger", "group_score_softmax_challenger_minus_plurality",
    ]
    for name in selected_scalar_names:
        feature_names.extend([f"plurality_{name}", f"challenger_{name}", f"diff_{name}"])
    feature_names.extend([
        "interaction_score_gap_entropy", "interaction_score_gap_vote_margin",
        "interaction_cohesion_gap_entropy", "interaction_arithmetic_gap_score",
        "interaction_support_gap_score",
    ])
    for i in range(len(bank)):
        p = int(plurality[i])
        valid_groups = np.flatnonzero(group_data["valid"][i]).tolist()
        alternatives = [g for g in valid_groups if g != p]
        if challenger_indices is not None:
            c = int(challenger_indices[i])
            if c not in valid_groups:
                raise ValueError("Explicit challenger is not a valid answer group")
        elif alternatives:
            c = max(alternatives, key=lambda g: float(group_scores[i, g]))
        else:
            c = p
        challenger[i] = c
        supports = sorted(
            [int(group_data["support"][i, g]) for g in valid_groups], reverse=True
        )
        pattern = "-".join(str(value) for value in supports)
        pattern_values = [
            float(pattern == "3-1"), float(pattern == "2-2"),
            float(pattern == "2-1-1"), float(pattern == "1-1-1-1"),
            float(pattern not in {"3-1", "2-2", "2-1-1", "1-1-1-1"}),
        ]
        ordered_supports = supports + [0]
        vote_margin = (ordered_supports[0] - ordered_supports[1]) / 4.0
        probabilities = np.asarray(ordered_supports[:-1], dtype=np.float64) / max(sum(supports), 1)
        entropy = float(-np.sum(probabilities * np.log(np.maximum(probabilities, 1e-12))))
        active_scores = np.asarray([group_scores[i, g] for g in valid_groups], dtype=np.float64)
        shifted = active_scores - float(active_scores.max())
        active_probs = np.exp(shifted); active_probs /= active_probs.sum()
        probability_map = {g: active_probs[k] for k, g in enumerate(valid_groups)}
        score_gap = float(group_scores[i, c] - group_scores[i, p])
        values = pattern_values + [
            len(valid_groups) / 4.0, vote_margin, entropy, score_gap,
            float(group_scores[i, p]), float(group_scores[i, c]),
            float(probability_map[c] - probability_map[p]),
        ]
        p_selected = []
        c_selected = []
        for name in selected_scalar_names:
            index = name_to_index[name]
            p_value = float(group_data["scalar"][i, p, index])
            c_value = float(group_data["scalar"][i, c, index])
            p_selected.append(p_value); c_selected.append(c_value)
            values.extend([p_value, c_value, c_value - p_value])
        cohesion_index = selected_scalar_names.index("within_group_mean_cosine")
        arithmetic_index = selected_scalar_names.index("extra_mean_mismatch_fraction")
        support_index = selected_scalar_names.index("support_fraction")
        values.extend([
            score_gap * entropy,
            score_gap * vote_margin,
            (c_selected[cohesion_index] - p_selected[cohesion_index]) * entropy,
            (c_selected[arithmetic_index] - p_selected[arithmetic_index]) * score_gap,
            (c_selected[support_index] - p_selected[support_index]) * score_gap,
        ])
        feature_rows.append(values)
    result = {
        "features": np.asarray(feature_rows, dtype=np.float64),
        "names": feature_names,
        "plurality": plurality,
        "challenger": challenger,
    }
    if group_label_array is not None:
        p_correct = group_label_array[np.arange(len(bank)), plurality]
        c_correct = group_label_array[np.arange(len(bank)), challenger]
        result.update({
            "plurality_correct": p_correct,
            "challenger_correct": c_correct,
            "informative": p_correct ^ c_correct,
            "overturn_label": (~p_correct & c_correct).astype(np.float64),
        })
    return result
