"""Pure NumPy linear listwise models used by training and submission."""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from common import parse_answer
from selector_features import stable_choice


VARIANT_BLOCKS = {
    "structure_only": ("hand",),
    "raw_embedding_only": ("raw",),
    "centered_embedding_only": ("centered",),
    "hand_centered": ("hand", "relative", "centered"),
    "hand_raw_centered": ("hand", "relative", "raw", "centered"),
    "stage2_hand_centered": ("stage2_hand", "relative", "centered"),
}


def _selected_arrays(blocks: dict, variant: str) -> list[tuple[str, np.ndarray]]:
    if variant not in VARIANT_BLOCKS:
        raise ValueError(f"Unknown variant: {variant}")
    return [(name, np.asarray(blocks[name], dtype=np.float64)) for name in VARIANT_BLOCKS[variant]]


def fit_transform(blocks: dict, variant: str, train_indices: np.ndarray) -> tuple[np.ndarray, dict]:
    arrays = _selected_arrays(blocks, variant)
    transformed = []
    means = []
    scales = []
    slices = {}
    cursor = 0
    for name, array in arrays:
        train_flat = array[train_indices].reshape(-1, array.shape[-1])
        mean = train_flat.mean(axis=0)
        scale = train_flat.std(axis=0)
        scale = np.where(scale < 1e-8, 1.0, scale)
        transformed.append((array - mean) / scale)
        means.append(mean)
        scales.append(scale)
        slices[name] = [cursor, cursor + array.shape[-1]]
        cursor += array.shape[-1]
    return np.concatenate(transformed, axis=2), {
        "variant": variant,
        "mean": np.concatenate(means),
        "scale": np.concatenate(scales),
        "slices": slices,
    }


def transform_with_model(blocks: dict, model: dict) -> np.ndarray:
    raw = np.concatenate([blocks[name] for name in VARIANT_BLOCKS[model["variant"]]], axis=2)
    return (np.asarray(raw, dtype=np.float64) - model["mean"]) / model["scale"]


def _masked_softmax(scores: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    masked = np.where(mask, scores, -np.inf)
    row_max = np.max(masked, axis=1, keepdims=True)
    shifted = np.where(mask, masked - row_max, -np.inf)
    exp_scores = np.where(mask, np.exp(shifted), 0.0)
    sums = exp_scores.sum(axis=1, keepdims=True)
    probs = exp_scores / np.maximum(sums, 1e-300)
    logsumexp = row_max[:, 0] + np.log(np.maximum(sums[:, 0], 1e-300))
    return probs, logsumexp


def loss_and_gradient(
    weights: np.ndarray,
    features: np.ndarray,
    labels: np.ndarray,
    valid: np.ndarray,
    l2_vector: np.ndarray | None = None,
    loss_kind: str = "listwise",
    group_weights: np.ndarray | None = None,
    pairwise_weight: float = 0.0,
) -> tuple[float, np.ndarray]:
    scores = np.einsum("nkd,d->nk", features, weights)
    active = np.asarray(valid, dtype=bool).copy()
    active[~active.any(axis=1)] = True
    positives = np.asarray(labels, dtype=bool) & active
    if np.any(~positives.any(axis=1)):
        raise ValueError("Each training group needs at least one active positive")

    if group_weights is None:
        group_weights = np.ones(features.shape[0], dtype=np.float64)
    group_weights = np.asarray(group_weights, dtype=np.float64)
    if group_weights.shape != (features.shape[0],) or np.any(group_weights < 0):
        raise ValueError("group_weights must be a non-negative vector with one value per group")
    weight_sum = float(group_weights.sum())
    if weight_sum <= 0:
        raise ValueError("group_weights must have positive total weight")
    normalized_weights = group_weights / weight_sum

    p_all, lse_all = _masked_softmax(scores, active)
    if loss_kind == "listwise":
        p_pos, lse_pos = _masked_softmax(scores, positives)
        loss = float(np.sum(normalized_weights * (lse_all - lse_pos)))
        grad_scores = (p_all - p_pos) * normalized_weights[:, None]
    elif loss_kind == "uniform_positive":
        target = positives / positives.sum(axis=1, keepdims=True)
        logp = np.where(active, scores - lse_all[:, None], 0.0)
        per_group = -np.sum(target * logp, axis=1)
        loss = float(np.sum(normalized_weights * per_group))
        grad_scores = (p_all - target) * normalized_weights[:, None]
    else:
        raise ValueError(f"Unknown loss: {loss_kind}")

    if pairwise_weight < 0:
        raise ValueError("pairwise_weight must be non-negative")
    if pairwise_weight:
        pair_loss = 0.0
        pair_grad = np.zeros_like(scores)
        for i in range(features.shape[0]):
            positive_indices = np.flatnonzero(positives[i])
            negative_indices = np.flatnonzero(active[i] & ~positives[i])
            pair_count = len(positive_indices) * len(negative_indices)
            if pair_count == 0:
                continue
            scale = normalized_weights[i] / pair_count
            for positive in positive_indices:
                for negative in negative_indices:
                    delta = float(scores[i, negative] - scores[i, positive])
                    pair_loss += scale * float(np.logaddexp(0.0, delta))
                    if delta >= 0:
                        sigmoid = 1.0 / (1.0 + math.exp(-delta))
                    else:
                        exp_delta = math.exp(delta)
                        sigmoid = exp_delta / (1.0 + exp_delta)
                    pair_grad[i, positive] -= scale * sigmoid
                    pair_grad[i, negative] += scale * sigmoid
        loss += pairwise_weight * pair_loss
        grad_scores += pairwise_weight * pair_grad

    grad = np.einsum("nk,nkd->d", grad_scores, features)
    if l2_vector is not None:
        loss += 0.5 * float(np.sum(l2_vector * weights * weights))
        grad += l2_vector * weights
    return loss, grad


def _l2_vector(meta: dict, hand_l2: float, embedding_l2: float) -> np.ndarray:
    result = np.zeros_like(meta["mean"], dtype=np.float64)
    for name, (start, end) in meta["slices"].items():
        result[start:end] = embedding_l2 if name in ("raw", "centered") else hand_l2
    return result


def train_linear_model(
    features: np.ndarray,
    labels: np.ndarray,
    valid: np.ndarray,
    meta: dict,
    hand_l2: float,
    embedding_l2: float,
    seed: int,
    loss_kind: str = "listwise",
    max_steps: int = 140,
    learning_rate: float = 0.03,
    group_weights: np.ndarray | None = None,
    pairwise_weight: float = 0.0,
    weight_scheme: str = "uniform",
) -> dict:
    # Zero initialization is deterministic and appropriate for a convex model.
    del seed
    weights = np.zeros(features.shape[-1], dtype=np.float64)
    l2 = _l2_vector(meta, hand_l2, embedding_l2)
    m = np.zeros_like(weights)
    v = np.zeros_like(weights)
    beta1, beta2 = 0.9, 0.999
    history = []
    best_loss = float("inf")
    best_weights = weights.copy()
    stale = 0
    for step in range(1, max_steps + 1):
        loss, grad = loss_and_gradient(
            weights, features, labels, valid, l2, loss_kind,
            group_weights=group_weights, pairwise_weight=pairwise_weight,
        )
        grad_norm = float(np.linalg.norm(grad))
        if grad_norm > 5.0:
            grad *= 5.0 / grad_norm
        m = beta1 * m + (1.0 - beta1) * grad
        v = beta2 * v + (1.0 - beta2) * (grad * grad)
        m_hat = m / (1.0 - beta1**step)
        v_hat = v / (1.0 - beta2**step)
        weights -= learning_rate * m_hat / (np.sqrt(v_hat) + 1e-8)
        history.append({"step": step, "loss": loss, "grad_norm": grad_norm})
        if loss < best_loss - 1e-8:
            best_loss = loss
            best_weights = weights.copy()
            stale = 0
        else:
            stale += 1
        if stale >= 25:
            break
    return {
        **meta,
        "weights": best_weights,
        "bias": 0.0,
        "hand_l2": float(hand_l2),
        "embedding_l2": float(embedding_l2),
        "loss_kind": loss_kind,
        "pairwise_weight": float(pairwise_weight),
        "weight_scheme": weight_scheme,
        "loss_history": history,
    }


def score_model(model: dict, blocks: dict) -> np.ndarray:
    features = transform_with_model(blocks, model)
    return np.einsum("nkd,d->nk", features, model["weights"]) + float(model.get("bias", 0.0))


def choices_from_scores(bank: list[dict], scores: np.ndarray, valid: np.ndarray) -> list[int]:
    choices = []
    for i, row in enumerate(bank):
        eligible = np.flatnonzero(valid[i]).tolist()
        if not eligible:
            eligible = list(range(4))
        best = max(float(scores[i, j]) for j in eligible)
        tied = [j for j in eligible if abs(float(scores[i, j]) - best) <= 1e-12]
        texts = [str(candidate.get("text", "")) for candidate in row["candidates"]]
        choices.append(int(stable_choice(str(row.get("question", "")), texts, tied)))
    return choices


def cluster_choices_from_scores(
    bank: list[dict],
    scores: np.ndarray,
    valid: np.ndarray,
    rule: str = "candidate",
    alpha: float = 0.0,
) -> list[int]:
    """Choose an answer cluster, then its highest-scoring candidate.

    Invalid candidates remain masked. If all candidates are invalid, this
    safely falls back to ordinary candidate scoring.
    """
    if rule == "candidate":
        return choices_from_scores(bank, scores, valid)
    if rule not in ("cluster_mean", "cluster_logsumexp", "cluster_max_plus_log_count"):
        raise ValueError(f"Unknown decision rule: {rule}")
    choices = []
    for i, row in enumerate(bank):
        texts = [str(candidate.get("text", "")) for candidate in row["candidates"]]
        answers = [parse_answer(text) for text in texts]
        if not valid[i].any():
            choices.extend(choices_from_scores([row], scores[i:i + 1], valid[i:i + 1]))
            continue
        clusters: dict[str, list[int]] = {}
        for j, answer in enumerate(answers):
            if valid[i, j] and answer is not None:
                clusters.setdefault(answer, []).append(j)
        cluster_scores = {}
        for answer, members in clusters.items():
            values = np.asarray([scores[i, j] for j in members], dtype=np.float64)
            if rule == "cluster_mean":
                aggregate = float(values.mean())
            elif rule == "cluster_logsumexp":
                maximum = float(values.max())
                aggregate = maximum + math.log(float(np.exp(values - maximum).sum()))
            else:
                aggregate = float(values.max()) + float(alpha) * math.log(len(members))
            cluster_scores[answer] = aggregate
        best_cluster_score = max(cluster_scores.values())
        tied_answers = [
            answer for answer, value in cluster_scores.items()
            if abs(value - best_cluster_score) <= 1e-12
        ]
        representatives = [stable_choice(str(row.get("question", "")), texts, clusters[a]) for a in tied_answers]
        representative = stable_choice(str(row.get("question", "")), texts, representatives)
        selected_answer = next(answer for answer in tied_answers if representative in clusters[answer])
        members = clusters[selected_answer]
        best_candidate_score = max(float(scores[i, j]) for j in members)
        tied_members = [j for j in members if abs(float(scores[i, j]) - best_candidate_score) <= 1e-12]
        choices.append(int(stable_choice(str(row.get("question", "")), texts, tied_members)))
    return choices


def save_model(
    model: dict,
    directory: Path,
    temperature: float = 1.0,
    decision_rule: str = "candidate",
    cluster_alpha: float = 0.0,
) -> None:
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    np.savez(
        directory / "model.npz",
        weights=np.asarray(model["weights"], dtype=np.float32),
        mean=np.asarray(model["mean"], dtype=np.float32),
        scale=np.asarray(model["scale"], dtype=np.float32),
    )
    config = {
        "format": "dase7506-mp2-linear-v1",
        "variant": model["variant"],
        "slices": model["slices"],
        "bias": float(model.get("bias", 0.0)),
        "hand_l2": float(model["hand_l2"]),
        "embedding_l2": float(model["embedding_l2"]),
        "loss_kind": model["loss_kind"],
        "pairwise_weight": float(model.get("pairwise_weight", 0.0)),
        "weight_scheme": model.get("weight_scheme", "uniform"),
        "decision_rule": decision_rule,
        "cluster_alpha": float(cluster_alpha),
        "temperature": float(temperature),
    }
    (directory / "config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    (directory / "loss_history.json").write_text(
        json.dumps(model["loss_history"], indent=2) + "\n", encoding="utf-8"
    )


def load_model(directory: Path) -> dict:
    directory = Path(directory)
    config_path = directory / "config.json"
    weights_path = directory / "model.npz"
    if not config_path.is_file() or not weights_path.is_file():
        raise FileNotFoundError(
            f"Checkpoint must contain config.json and model.npz: {directory}"
        )
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("format") != "dase7506-mp2-linear-v1":
        raise ValueError("Unsupported checkpoint format")
    with np.load(weights_path, allow_pickle=False) as data:
        model = {
            **config,
            "weights": data["weights"].astype(np.float64),
            "mean": data["mean"].astype(np.float64),
            "scale": data["scale"].astype(np.float64),
        }
    expected = sum(
        32 if name in ("hand", "stage2_hand") else 6 if name == "relative" else 1024
        for name in VARIANT_BLOCKS[model["variant"]]
    )
    if any(model[key].shape != (expected,) for key in ("weights", "mean", "scale")):
        raise ValueError("Checkpoint array shape mismatch")
    return model

