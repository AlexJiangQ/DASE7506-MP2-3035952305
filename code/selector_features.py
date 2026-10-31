"""Permutation-equivariant features for MP2 candidate selection.

Only the supplied bank, supplied embeddings, and common.parse_answer are used.
The 32 hand features are intentionally compact and interpretable; arithmetic
checks are conservative soft signals rather than hard rejection rules.
"""
from __future__ import annotations

from collections import Counter
from fractions import Fraction
import hashlib
import math
import re

import numpy as np

from common import parse_answer


NUMBER_RE = re.compile(r"[-+]?\d[\d,]*(?:\.\d+)?(?:/\d+)?")
SIMPLE_EQ_RE = re.compile(
    r"(?<![\w.])([-+]?\d[\d,]*(?:\.\d+)?(?:/\d+)?)\s*"
    r"([+\-*/×÷])\s*"
    r"([-+]?\d[\d,]*(?:\.\d+)?(?:/\d+)?)\s*=\s*"
    r"([-+]?\d[\d,]*(?:\.\d+)?(?:/\d+)?)(?![\w.])"
)
FINAL_RE = re.compile(r"(?:final\s+answer|answer\s*:)", re.IGNORECASE)
WORD_RE = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?")

# Small, task-generic control vocabularies. They are fixed in code and are not
# learned from validation/test data or sourced from an external corpus.
STOP_WORDS = {
    "the", "and", "that", "this", "with", "from", "for", "are", "was",
    "were", "has", "have", "had", "how", "what", "when", "where", "who",
    "does", "did", "into", "than", "then", "each", "there", "their",
}
MATH_WORDS = {
    "sum", "total", "difference", "product", "divide", "divided", "average",
    "percent", "percentage", "ratio", "rate", "remaining", "increase",
    "decrease", "times", "more", "less", "equal", "cost", "price",
}
UNIT_WORDS = {
    "second", "seconds", "minute", "minutes", "hour", "hours", "day", "days",
    "week", "weeks", "month", "months", "year", "years", "inch", "inches",
    "foot", "feet", "yard", "yards", "mile", "miles", "meter", "meters",
    "kilometer", "kilometers", "gram", "grams", "kilogram", "kilograms",
    "ounce", "ounces", "pound", "pounds", "liter", "liters", "dollar",
    "dollars", "cent", "cents", "percent", "percentage",
}

HAND_FEATURE_NAMES = [
    "parse_valid", "same_answer_count", "plurality_member",
    "distinct_valid_answers", "max_cluster_size", "cluster_margin",
    "valid_fraction", "log_chars", "log_lines", "log_numeric_tokens",
    "log_equals_markers", "has_hash_answer", "has_boxed",
    "has_final_answer_marker", "centered_log_chars", "centered_log_lines",
    "centered_log_numeric_tokens", "centered_log_equals_markers",
    "log_question_number_count", "log_reused_question_numbers",
    "reused_question_number_ratio", "log_outside_numbers",
    "final_answer_near_end", "log_final_answer_occurrences",
    "log_checkable_equations", "log_passed_equations", "equation_pass_ratio",
    "has_failed_equation", "equation_produces_final", "log_abs_final_answer",
    "log_unique_candidate_numbers", "numeric_token_density",
]

REL_FEATURE_NAMES = [
    "cosine_to_group_center", "nearest_candidate_cosine",
    "same_answer_peer_cosine", "different_answer_peer_cosine",
    "has_same_answer_peer", "has_different_answer_peer",
]

# Twenty-four stage-1 controls plus eight question-candidate lexical
# interactions. Keeping the block at 32 features preserves the declared cap.
STAGE2_BASE_HAND_INDICES = (
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
    14, 15, 16, 17,
    18, 19, 20, 21, 22, 23,
    26, 27, 28,
)
LEXICAL_FEATURE_NAMES = [
    "question_token_coverage", "log_shared_question_tokens",
    "question_token_precision", "question_math_word_coverage",
    "question_unit_coverage", "log_outside_unit_words",
    "tail_question_token_coverage", "long_question_token_coverage",
]
STAGE2_HAND_FEATURE_NAMES = [HAND_FEATURE_NAMES[i] for i in STAGE2_BASE_HAND_INDICES] + LEXICAL_FEATURE_NAMES


def _fraction(token: str) -> Fraction | None:
    try:
        return Fraction(token.replace(",", ""))
    except (ValueError, ZeroDivisionError):
        return None


def _number_set(text: str) -> set[str]:
    values = set()
    for token in NUMBER_RE.findall(text):
        value = _fraction(token)
        if value is not None:
            values.add(str(value))
    return values


def _word_set(text: str) -> set[str]:
    return {
        token.lower() for token in WORD_RE.findall(text)
        if len(token) >= 3 and token.lower() not in STOP_WORDS
    }


def _safe_log_abs_answer(answer: str | None) -> float:
    if answer is None:
        return 0.0
    value = _fraction(answer)
    if value is None:
        return 0.0
    # Avoid converting enormous rationals to float before clipping.
    magnitude = abs(value)
    if magnitude > 10**12:
        magnitude = Fraction(10**12)
    return math.log1p(float(magnitude))


def _arithmetic_features(text: str, final_answer: str | None) -> tuple[int, int, bool, bool]:
    checked = passed = 0
    failed = produces_final = False
    final_value = _fraction(final_answer) if final_answer is not None else None
    for match in SIMPLE_EQ_RE.finditer(text):
        left, op, right, shown = match.groups()
        a, b, c = _fraction(left), _fraction(right), _fraction(shown)
        if a is None or b is None or c is None:
            continue
        if op in ("/", "÷") and b == 0:
            continue
        checked += 1
        if op == "+":
            expected = a + b
        elif op == "-":
            expected = a - b
        elif op in ("*", "×"):
            expected = a * b
        else:
            expected = a / b
        if expected == c:
            passed += 1
            produces_final |= final_value is not None and c == final_value
        else:
            failed = True
    return checked, passed, failed, produces_final


def stable_choice(question: str, candidate_texts: list[str], eligible: list[int]) -> int:
    """Content-based deterministic tie break, independent of IDs and order."""
    if not eligible:
        return 0
    return min(
        eligible,
        key=lambda j: hashlib.sha256(
            (question + "\0" + candidate_texts[j]).encode("utf-8")
        ).digest(),
    )


def plurality_choice(row: dict) -> int:
    texts = [str(candidate.get("text", "")) for candidate in row["candidates"]]
    answers = [parse_answer(text) for text in texts]
    counts = Counter(answer for answer in answers if answer is not None)
    if not counts:
        return stable_choice(row.get("question", ""), texts, list(range(4)))
    best_count = max(counts.values())
    tied_answers = {answer for answer, count in counts.items() if count == best_count}
    eligible = [j for j, answer in enumerate(answers) if answer in tied_answers]
    return stable_choice(row.get("question", ""), texts, eligible)


def deterministic_random_choice(row: dict) -> int:
    texts = [str(candidate.get("text", "")) for candidate in row["candidates"]]
    return stable_choice(row.get("question", ""), texts, list(range(4)))


def longest_choice(row: dict) -> int:
    texts = [str(candidate.get("text", "")) for candidate in row["candidates"]]
    longest = max(len(text) for text in texts)
    eligible = [j for j, text in enumerate(texts) if len(text) == longest]
    return stable_choice(row.get("question", ""), texts, eligible)


def extract_feature_blocks(bank: list[dict], embeddings: np.ndarray) -> dict:
    embeddings = np.asarray(embeddings, dtype=np.float64)
    if embeddings.ndim != 3 or embeddings.shape[1:] != (4, 1024):
        raise ValueError("Expected embeddings with shape [N, 4, 1024]")
    if len(bank) != embeddings.shape[0]:
        raise ValueError("Bank and embedding counts differ")

    n = len(bank)
    hand = np.zeros((n, 4, len(HAND_FEATURE_NAMES)), dtype=np.float64)
    stage2_hand = np.zeros((n, 4, len(STAGE2_HAND_FEATURE_NAMES)), dtype=np.float64)
    rel = np.zeros((n, 4, len(REL_FEATURE_NAMES)), dtype=np.float64)
    answers: list[list[str | None]] = []
    valid = np.zeros((n, 4), dtype=bool)

    centered = embeddings - embeddings.mean(axis=1, keepdims=True)
    norms = np.linalg.norm(embeddings, axis=2, keepdims=True)
    normalized = embeddings / np.maximum(norms, 1e-12)
    centers = embeddings.mean(axis=1)
    center_norm = np.linalg.norm(centers, axis=1, keepdims=True)

    for i, row in enumerate(bank):
        texts = [str(candidate.get("text", "")) for candidate in row["candidates"]]
        row_answers = [parse_answer(text) for text in texts]
        answers.append(row_answers)
        valid[i] = [answer is not None for answer in row_answers]
        counts = Counter(answer for answer in row_answers if answer is not None)
        distinct = len(counts)
        max_cluster = max(counts.values(), default=0)
        ordered_counts = sorted(counts.values(), reverse=True)
        second = ordered_counts[1] if len(ordered_counts) > 1 else 0
        plurality_answers = {a for a, c in counts.items() if c == max_cluster}
        valid_fraction = float(valid[i].mean())
        question = str(row.get("question", ""))
        q_numbers = _number_set(question)
        q_words = _word_set(question)
        q_math = q_words & MATH_WORDS
        q_units = q_words & UNIT_WORDS
        q_long = {word for word in q_words if len(word) >= 6}

        raw_style = np.zeros((4, 4), dtype=np.float64)
        for j, text in enumerate(texts):
            numeric_tokens = NUMBER_RE.findall(text)
            raw_style[j] = [
                math.log1p(len(text)),
                math.log1p(text.count("\n") + (1 if text else 0)),
                math.log1p(len(numeric_tokens)),
                math.log1p(text.count("=") + text.count("≈")),
            ]
        centered_style = raw_style - raw_style.mean(axis=0, keepdims=True)

        pair_cos = normalized[i] @ normalized[i].T
        center_cos = (embeddings[i] @ centers[i]) / np.maximum(
            np.linalg.norm(embeddings[i], axis=1) * float(center_norm[i, 0]), 1e-12
        )

        for j, text in enumerate(texts):
            answer = row_answers[j]
            numeric_tokens = NUMBER_RE.findall(text)
            candidate_numbers = _number_set(text)
            reused = candidate_numbers & q_numbers
            outside = candidate_numbers - q_numbers
            checked, passed, failed, produces = _arithmetic_features(text, answer)
            answer_occurrences = text.count(answer) if answer is not None else 0
            tail = text[-160:]
            hand[i, j] = [
                float(answer is not None),
                counts.get(answer, 0) / 4.0 if answer is not None else 0.0,
                float(answer is not None and answer in plurality_answers),
                distinct / 4.0,
                max_cluster / 4.0,
                (max_cluster - second) / 4.0,
                valid_fraction,
                *raw_style[j].tolist(),
                float("####" in text),
                float("\\boxed" in text),
                float(bool(FINAL_RE.search(text))),
                *centered_style[j].tolist(),
                math.log1p(len(q_numbers)),
                math.log1p(len(reused)),
                len(reused) / max(len(q_numbers), 1),
                math.log1p(len(outside)),
                float(answer is not None and answer in tail),
                math.log1p(answer_occurrences),
                math.log1p(checked),
                math.log1p(passed),
                passed / checked if checked else 0.0,
                float(failed),
                float(produces),
                _safe_log_abs_answer(answer),
                math.log1p(len(candidate_numbers)),
                len(numeric_tokens) / max(len(text), 1),
            ]

            candidate_words = _word_set(text)
            shared_words = candidate_words & q_words
            candidate_units = candidate_words & UNIT_WORDS
            tail_words = _word_set(text[-240:])
            lexical = np.asarray([
                len(shared_words) / max(len(q_words), 1),
                math.log1p(len(shared_words)),
                len(shared_words) / max(len(candidate_words), 1),
                len(candidate_words & q_math) / max(len(q_math), 1),
                len(candidate_words & q_units) / max(len(q_units), 1),
                math.log1p(len(candidate_units - q_units)),
                len(tail_words & q_words) / max(len(q_words), 1),
                len(candidate_words & q_long) / max(len(q_long), 1),
            ], dtype=np.float64)
            stage2_hand[i, j] = np.concatenate([
                hand[i, j, list(STAGE2_BASE_HAND_INDICES)], lexical
            ])

            peers = [k for k in range(4) if k != j]
            same = [k for k in peers if answer is not None and row_answers[k] == answer]
            different = [k for k in peers if row_answers[k] != answer]
            rel[i, j] = [
                center_cos[j],
                max((pair_cos[j, k] for k in peers), default=0.0),
                float(np.mean([pair_cos[j, k] for k in same])) if same else 0.0,
                float(np.mean([pair_cos[j, k] for k in different])) if different else 0.0,
                float(bool(same)),
                float(bool(different)),
            ]

    return {
        "hand": hand,
        "stage2_hand": stage2_hand,
        "relative": rel,
        "raw": embeddings,
        "centered": centered,
        "answers": answers,
        "valid": valid,
    }


def labels_from_references(bank: list[dict], references: list[dict]) -> np.ndarray:
    if len(bank) != len(references):
        raise ValueError("Bank and reference counts differ")
    labels = np.zeros((len(bank), 4), dtype=bool)
    for i, (row, ref) in enumerate(zip(bank, references)):
        if row["id"] != ref["id"] or row["question"] != ref["question"]:
            raise ValueError("Reference alignment mismatch")
        gold = parse_answer(ref["answer"])
        for j, candidate in enumerate(row["candidates"]):
            answer = parse_answer(candidate.get("text", ""))
            labels[i, j] = answer is not None and answer == gold
        if not labels[i].any():
            raise ValueError("Supplied group has no correct parsed candidate")
    return labels


def hard_group_mask(answers: list[list[str | None]]) -> np.ndarray:
    return np.asarray([
        len({answer for answer in row if answer is not None}) >= 2 for row in answers
    ], dtype=bool)

