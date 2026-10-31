"""Deterministic NumPy models and calibration for CREO-RR."""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np


OUTCOME_NAMES = ("beneficial", "harmful", "neutral")


def softmax(logits: np.ndarray) -> np.ndarray:
    values = np.asarray(logits, dtype=np.float64)
    shifted = values - values.max(axis=1, keepdims=True)
    exp_values = np.exp(shifted)
    return exp_values / exp_values.sum(axis=1, keepdims=True)


def class_weights(labels: np.ndarray, mode: str) -> np.ndarray:
    labels = np.asarray(labels, dtype=np.int64)
    if mode == "uniform":
        return np.ones(3, dtype=np.float64)
    if mode != "balanced-capped":
        raise ValueError(f"Unknown class weight mode: {mode}")
    counts = np.bincount(labels, minlength=3).astype(np.float64)
    if np.any(counts == 0):
        raise ValueError("Every residual class must occur in the training portion")
    raw = len(labels) / (3.0 * counts)
    floor = float(raw.min())
    capped = np.minimum(raw, 4.0 * floor)
    return capped / capped.mean()


def fit_scaler(features: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    selected = np.asarray(features, dtype=np.float64)[np.asarray(mask, dtype=bool)]
    mean = selected.mean(axis=0)
    scale = selected.std(axis=0)
    return mean, np.where(scale < 1e-8, 1.0, scale)


def multinomial_loss_and_gradient(
    parameters: np.ndarray,
    features: np.ndarray,
    labels: np.ndarray,
    example_weights: np.ndarray,
    l2: float,
    harm_aux_weight: float,
) -> tuple[float, np.ndarray]:
    """Weighted three-class CE with an optional aggregated harmful BCE."""
    x = np.asarray(features, dtype=np.float64)
    y = np.asarray(labels, dtype=np.int64)
    weights = np.asarray(example_weights, dtype=np.float64)
    weights = weights / weights.sum()
    logits = x @ parameters
    probabilities = softmax(logits)
    loss = -float(np.sum(weights * np.log(np.maximum(probabilities[np.arange(len(y)), y], 1e-300))))
    gradient_logits = probabilities.copy()
    gradient_logits[np.arange(len(y)), y] -= 1.0
    gradient_logits *= weights[:, None]

    if harm_aux_weight:
        harmful = (y == 1).astype(np.float64)
        p_harm = np.clip(probabilities[:, 1], 1e-12, 1 - 1e-12)
        harm_loss = -float(np.sum(weights * (
            harmful * np.log(p_harm) + (1.0 - harmful) * np.log(1.0 - p_harm)
        )))
        aux = np.zeros_like(probabilities)
        aux[:, 1] = p_harm - harmful
        nonharm_scale = (
            harmful - (1.0 - harmful) * p_harm / np.maximum(1.0 - p_harm, 1e-12)
        )
        aux[:, 0] = probabilities[:, 0] * nonharm_scale
        aux[:, 2] = probabilities[:, 2] * nonharm_scale
        gradient_logits += harm_aux_weight * weights[:, None] * aux
        loss += harm_aux_weight * harm_loss

    gradient = x.T @ gradient_logits
    if l2:
        # The last feature is the explicit bias and is not regularized.
        loss += 0.5 * l2 * float(np.sum(parameters[:-1] ** 2))
        gradient[:-1] += l2 * parameters[:-1]
    return loss, gradient


def train_multinomial(
    features: np.ndarray,
    labels: np.ndarray,
    question_indices: np.ndarray,
    train_questions: np.ndarray,
    class_weight_mode: str,
    harm_aux_weight: float,
    l2: float = 0.1,
    max_steps: int = 500,
    learning_rate: float = 0.03,
) -> dict:
    train_questions = np.asarray(train_questions, dtype=np.int64)
    row_mask = np.isin(question_indices, train_questions)
    mean, scale = fit_scaler(features, row_mask)
    x = (np.asarray(features[row_mask], dtype=np.float64) - mean) / scale
    x = np.column_stack([x, np.ones(len(x), dtype=np.float64)])
    y = np.asarray(labels[row_mask], dtype=np.int64)
    class_weight = class_weights(y, class_weight_mode)
    example_weight = class_weight[y]
    parameters = np.zeros((x.shape[1], 3), dtype=np.float64)
    first = np.zeros_like(parameters)
    second = np.zeros_like(parameters)
    beta1, beta2 = 0.9, 0.999
    best_loss = float("inf")
    best = parameters.copy()
    stale = 0
    history = []
    for step in range(1, max_steps + 1):
        loss, gradient = multinomial_loss_and_gradient(
            parameters, x, y, example_weight, l2, harm_aux_weight
        )
        norm = float(np.linalg.norm(gradient))
        if norm > 5.0:
            gradient *= 5.0 / norm
        first = beta1 * first + (1.0 - beta1) * gradient
        second = beta2 * second + (1.0 - beta2) * (gradient * gradient)
        first_hat = first / (1.0 - beta1**step)
        second_hat = second / (1.0 - beta2**step)
        parameters -= learning_rate * first_hat / (np.sqrt(second_hat) + 1e-8)
        history.append({"step": step, "loss": loss, "gradient_norm": norm})
        if loss < best_loss - 1e-9:
            best_loss = loss
            best = parameters.copy()
            stale = 0
        else:
            stale += 1
        if stale >= 60:
            break
    return {
        "parameters": best,
        "mean": mean,
        "scale": scale,
        "class_weights": class_weight,
        "class_weight_mode": class_weight_mode,
        "harm_aux_weight": float(harm_aux_weight),
        "l2": float(l2),
        "loss_history": history,
    }


def predict_logits(model: dict, features: np.ndarray) -> np.ndarray:
    x = (np.asarray(features, dtype=np.float64) - model["mean"]) / model["scale"]
    x = np.column_stack([x, np.ones(len(x), dtype=np.float64)])
    return x @ model["parameters"]


def predict_probabilities(model: dict, features: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    return softmax(predict_logits(model, features) / float(temperature))


def multiclass_nll(logits: np.ndarray, labels: np.ndarray, temperature: float = 1.0) -> float:
    probabilities = softmax(np.asarray(logits, dtype=np.float64) / float(temperature))
    return -float(np.mean(np.log(np.maximum(probabilities[np.arange(len(labels)), labels], 1e-300))))


def fit_temperature(logits: np.ndarray, labels: np.ndarray) -> float:
    """Deterministic bounded one-dimensional calibration search."""
    log_temperatures = np.linspace(math.log(0.25), math.log(4.0), 121)
    losses = np.asarray([
        multiclass_nll(logits, labels, math.exp(value)) for value in log_temperatures
    ])
    best = int(np.argmin(losses))
    return float(math.exp(log_temperatures[best]))


def cross_fitted_temperature(
    logits: np.ndarray,
    labels: np.ndarray,
    question_indices: np.ndarray,
    folds: list[np.ndarray],
) -> tuple[np.ndarray, list[float]]:
    probabilities = np.zeros_like(logits, dtype=np.float64)
    temperatures = []
    all_questions = np.arange(max(int(question_indices.max()) + 1, 1))
    for fold in folds:
        eval_mask = np.isin(question_indices, fold)
        train_mask = np.isin(question_indices, np.setdiff1d(all_questions, fold))
        temperature = fit_temperature(logits[train_mask], labels[train_mask])
        probabilities[eval_mask] = softmax(logits[eval_mask] / temperature)
        temperatures.append(temperature)
    return probabilities, temperatures


def calibration_metrics(probabilities: np.ndarray, labels: np.ndarray, bins: int = 10) -> dict:
    probabilities = np.asarray(probabilities, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    chosen = np.argmax(probabilities, axis=1)
    confidence = np.max(probabilities, axis=1)
    correct = chosen == labels
    ece = 0.0
    rows = []
    edges = np.linspace(0.0, 1.0, bins + 1)
    for index in range(bins):
        mask = (confidence >= edges[index]) & (
            confidence <= edges[index + 1] if index == bins - 1 else confidence < edges[index + 1]
        )
        if not mask.any():
            rows.append({"lower": edges[index], "upper": edges[index + 1], "n": 0})
            continue
        accuracy = float(correct[mask].mean())
        mean_confidence = float(confidence[mask].mean())
        ece += float(mask.mean()) * abs(accuracy - mean_confidence)
        rows.append({
            "lower": float(edges[index]), "upper": float(edges[index + 1]),
            "n": int(mask.sum()), "accuracy": accuracy, "confidence": mean_confidence,
        })
    nll = -float(np.mean(np.log(np.maximum(probabilities[np.arange(len(labels)), labels], 1e-300))))
    return {"multiclass_nll": nll, "ece": ece, "bins": rows}


def save_stage5_bundle(directory: Path, model: dict, temperature: float, config: dict) -> None:
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    np.savez(
        directory / "residual_model.npz",
        parameters=np.asarray(model["parameters"], dtype=np.float32),
        mean=np.asarray(model["mean"], dtype=np.float32),
        scale=np.asarray(model["scale"], dtype=np.float32),
        class_weights=np.asarray(model["class_weights"], dtype=np.float32),
    )
    payload = {
        "format": "dase7506-mp2-creo-rr-v1",
        **config,
        "temperature": float(temperature),
        "class_weight_mode": model["class_weight_mode"],
        "harm_aux_weight": model["harm_aux_weight"],
        "l2": model["l2"],
    }
    (directory / "config.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    (directory / "loss_history.json").write_text(
        json.dumps(model["loss_history"], indent=2) + "\n", encoding="utf-8"
    )


def load_stage5_bundle(directory: Path) -> dict:
    directory = Path(directory)
    config_path = directory / "config.json"
    model_path = directory / "residual_model.npz"
    if not config_path.is_file() or not model_path.is_file():
        raise FileNotFoundError(f"Stage-5 checkpoint is incomplete: {directory}")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("format") != "dase7506-mp2-creo-rr-v1":
        raise ValueError("Unsupported Stage-5 checkpoint format")
    with np.load(model_path, allow_pickle=False) as data:
        model = {
            "parameters": data["parameters"].astype(np.float64),
            "mean": data["mean"].astype(np.float64),
            "scale": data["scale"].astype(np.float64),
            "class_weights": data["class_weights"].astype(np.float64),
            "class_weight_mode": config["class_weight_mode"],
            "harm_aux_weight": float(config["harm_aux_weight"]),
            "l2": float(config["l2"]),
        }
    if model["parameters"].shape != (len(model["mean"]) + 1, 3):
        raise ValueError("Stage-5 residual model shape mismatch")
    return {"config": config, "model": model}
