"""Deterministic NumPy models for Stage 3 CREO."""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from selector_model import loss_and_gradient


def fit_group_scaler(
    features: np.ndarray, valid: np.ndarray, train_indices: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    mask = valid[train_indices]
    flat = features[train_indices][mask]
    mean = flat.mean(axis=0)
    scale = flat.std(axis=0)
    scale = np.where(scale < 1e-8, 1.0, scale)
    return mean, scale


def transform_groups(features: np.ndarray, mean: np.ndarray, scale: np.ndarray) -> np.ndarray:
    return (np.asarray(features, dtype=np.float64) - mean) / scale


def train_group_model(
    features: np.ndarray,
    labels: np.ndarray,
    valid: np.ndarray,
    train_indices: np.ndarray,
    scalar_dim: int,
    scalar_l2: float,
    embedding_l2: float,
    pairwise_weight: float,
    max_steps: int = 140,
    learning_rate: float = 0.03,
) -> dict:
    mean, scale = fit_group_scaler(features, valid, train_indices)
    transformed = transform_groups(features, mean, scale)
    x = transformed[train_indices]
    y = labels[train_indices]
    active = valid[train_indices]
    weights = np.zeros(features.shape[-1], dtype=np.float64)
    l2 = np.full(features.shape[-1], embedding_l2, dtype=np.float64)
    l2[:scalar_dim] = scalar_l2
    m = np.zeros_like(weights); v = np.zeros_like(weights)
    beta1, beta2 = 0.9, 0.999
    best_loss = float("inf"); best_weights = weights.copy(); stale = 0
    history = []
    for step in range(1, max_steps + 1):
        loss, gradient = loss_and_gradient(
            weights,
            x,
            y,
            active,
            l2_vector=l2,
            loss_kind="listwise",
            pairwise_weight=pairwise_weight,
        )
        norm = float(np.linalg.norm(gradient))
        if norm > 5.0:
            gradient *= 5.0 / norm
        m = beta1 * m + (1.0 - beta1) * gradient
        v = beta2 * v + (1.0 - beta2) * (gradient * gradient)
        m_hat = m / (1.0 - beta1**step)
        v_hat = v / (1.0 - beta2**step)
        weights -= learning_rate * m_hat / (np.sqrt(v_hat) + 1e-8)
        history.append({"step": step, "loss": loss, "gradient_norm": norm})
        if loss < best_loss - 1e-8:
            best_loss = loss; best_weights = weights.copy(); stale = 0
        else:
            stale += 1
        if stale >= 25:
            break
    return {
        "weights": best_weights,
        "mean": mean,
        "scale": scale,
        "scalar_dim": int(scalar_dim),
        "scalar_l2": float(scalar_l2),
        "embedding_l2": float(embedding_l2),
        "pairwise_weight": float(pairwise_weight),
        "loss_history": history,
    }


def score_group_model(model: dict, features: np.ndarray, valid: np.ndarray) -> np.ndarray:
    transformed = transform_groups(features, model["mean"], model["scale"])
    scores = np.einsum("ngd,d->ng", transformed, model["weights"])
    return np.where(valid, scores, -np.inf)


def _sigmoid(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    output = np.empty_like(values)
    positive = values >= 0
    output[positive] = 1.0 / (1.0 + np.exp(-values[positive]))
    exp_values = np.exp(values[~positive])
    output[~positive] = exp_values / (1.0 + exp_values)
    return output


def fit_logistic(
    features: np.ndarray,
    labels: np.ndarray,
    train_mask: np.ndarray,
    l2: float,
    class_balance: bool = False,
    max_steps: int = 500,
    learning_rate: float = 0.03,
) -> dict:
    train_mask = np.asarray(train_mask, dtype=bool)
    x_raw = np.asarray(features[train_mask], dtype=np.float64)
    y = np.asarray(labels[train_mask], dtype=np.float64)
    if len(np.unique(y)) < 2:
        raise ValueError("Logistic training needs both classes")
    mean = x_raw.mean(axis=0)
    scale = x_raw.std(axis=0)
    scale = np.where(scale < 1e-8, 1.0, scale)
    x = (x_raw - mean) / scale
    x = np.concatenate([x, np.ones((len(x), 1), dtype=np.float64)], axis=1)
    weights = np.zeros(x.shape[1], dtype=np.float64)
    if class_balance:
        positives = max(float(y.sum()), 1.0)
        negatives = max(float(len(y) - y.sum()), 1.0)
        sample_weights = np.where(y > 0.5, len(y) / (2.0 * positives), len(y) / (2.0 * negatives))
    else:
        sample_weights = np.ones(len(y), dtype=np.float64)
    sample_weights /= sample_weights.sum()
    m = np.zeros_like(weights); v = np.zeros_like(weights)
    beta1, beta2 = 0.9, 0.999
    best_loss = float("inf"); best_weights = weights.copy(); stale = 0
    history = []
    for step in range(1, max_steps + 1):
        logits = x @ weights
        probs = _sigmoid(logits)
        losses = np.logaddexp(0.0, logits) - y * logits
        loss = float(np.sum(sample_weights * losses) + 0.5 * l2 * np.sum(weights[:-1] ** 2))
        gradient = x.T @ (sample_weights * (probs - y))
        gradient[:-1] += l2 * weights[:-1]
        norm = float(np.linalg.norm(gradient))
        if norm > 5.0:
            gradient *= 5.0 / norm
        m = beta1 * m + (1.0 - beta1) * gradient
        v = beta2 * v + (1.0 - beta2) * (gradient * gradient)
        m_hat = m / (1.0 - beta1**step)
        v_hat = v / (1.0 - beta2**step)
        weights -= learning_rate * m_hat / (np.sqrt(v_hat) + 1e-8)
        history.append({"step": step, "loss": loss, "gradient_norm": norm})
        if loss < best_loss - 1e-9:
            best_loss = loss; best_weights = weights.copy(); stale = 0
        else:
            stale += 1
        if stale >= 50:
            break
    return {
        "weights": best_weights,
        "mean": mean,
        "scale": scale,
        "l2": float(l2),
        "class_balance": bool(class_balance),
        "loss_history": history,
    }


def logistic_logits(model: dict, features: np.ndarray) -> np.ndarray:
    x = (np.asarray(features, dtype=np.float64) - model["mean"]) / model["scale"]
    return x @ model["weights"][:-1] + model["weights"][-1]


def logistic_probabilities(model: dict, features: np.ndarray) -> np.ndarray:
    return _sigmoid(logistic_logits(model, features))


def gate_group_choices(
    plurality: np.ndarray,
    challenger: np.ndarray,
    probabilities: np.ndarray,
    threshold: float,
) -> np.ndarray:
    plurality = np.asarray(plurality, dtype=np.int64)
    challenger = np.asarray(challenger, dtype=np.int64)
    return np.where(probabilities >= threshold, challenger, plurality).astype(np.int64)


def save_creo_bundle(
    directory: Path,
    group_model: dict,
    reliability_model: dict,
    gate_model: dict,
    config: dict,
) -> None:
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    np.savez(
        directory / "creo_model.npz",
        group_weights=group_model["weights"].astype(np.float32),
        group_mean=group_model["mean"].astype(np.float32),
        group_scale=group_model["scale"].astype(np.float32),
        reliability_weights=reliability_model["weights"].astype(np.float32),
        reliability_mean=reliability_model["mean"].astype(np.float32),
        reliability_scale=reliability_model["scale"].astype(np.float32),
        gate_weights=gate_model["weights"].astype(np.float32),
        gate_mean=gate_model["mean"].astype(np.float32),
        gate_scale=gate_model["scale"].astype(np.float32),
    )
    payload = {
        "format": "dase7506-mp2-creo-v1",
        **config,
        "group_model": {
            key: group_model[key] for key in
            ("scalar_dim", "scalar_l2", "embedding_l2", "pairwise_weight")
        },
        "reliability_model": {
            "l2": reliability_model["l2"],
            "class_balance": reliability_model["class_balance"],
        },
        "gate_model": {
            "l2": gate_model["l2"],
            "class_balance": gate_model["class_balance"],
        },
    }
    (directory / "config.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    (directory / "loss_histories.json").write_text(json.dumps({
        "group": group_model["loss_history"],
        "reliability": reliability_model["loss_history"],
        "gate": gate_model["loss_history"],
    }, indent=2) + "\n", encoding="utf-8")


def load_creo_bundle(directory: Path) -> dict:
    directory = Path(directory)
    config = json.loads((directory / "config.json").read_text(encoding="utf-8"))
    if config.get("format") != "dase7506-mp2-creo-v1":
        raise ValueError("Unsupported CREO bundle")
    with np.load(directory / "creo_model.npz", allow_pickle=False) as data:
        group_model = {
            **config["group_model"],
            "weights": data["group_weights"].astype(np.float64),
            "mean": data["group_mean"].astype(np.float64),
            "scale": data["group_scale"].astype(np.float64),
        }
        reliability_model = {
            **config["reliability_model"],
            "weights": data["reliability_weights"].astype(np.float64),
            "mean": data["reliability_mean"].astype(np.float64),
            "scale": data["reliability_scale"].astype(np.float64),
        }
        gate_model = {
            **config["gate_model"],
            "weights": data["gate_weights"].astype(np.float64),
            "mean": data["gate_mean"].astype(np.float64),
            "scale": data["gate_scale"].astype(np.float64),
        }
    return {"config": config, "group_model": group_model,
            "reliability_model": reliability_model, "gate_model": gate_model}
