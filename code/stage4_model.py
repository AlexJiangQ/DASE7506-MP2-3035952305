"""Pure NumPy models and losses for Stage 4 CREO-3W."""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from stage4_features import (
    candidate_feature_backward,
    candidate_features_from_projected,
    initialize_projection,
    interaction_feature_dim,
    project,
)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    out = np.empty_like(x)
    positive = x >= 0
    out[positive] = 1.0 / (1.0 + np.exp(-x[positive]))
    expx = np.exp(x[~positive])
    out[~positive] = expx / (1.0 + expx)
    return out


def _masked_softmax(scores: np.ndarray, mask: np.ndarray) -> np.ndarray:
    masked = np.where(mask, scores, -np.inf)
    maximum = np.max(masked, axis=1, keepdims=True)
    exponent = np.where(mask, np.exp(masked - maximum), 0.0)
    return exponent / np.maximum(exponent.sum(axis=1, keepdims=True), 1e-300)


def ranking_score_loss_gradient(
    scores: np.ndarray,
    labels: np.ndarray,
    valid: np.ndarray,
    pairwise_weight: float = 0.0,
) -> tuple[float, np.ndarray, dict]:
    scores = np.asarray(scores, dtype=np.float64)
    active = np.asarray(valid, dtype=bool).copy()
    active[~active.any(axis=1)] = True
    positive = np.asarray(labels, dtype=bool) & active
    if np.any(~positive.any(axis=1)):
        raise ValueError("Every ranking row needs an active positive")
    p_all = _masked_softmax(scores, active)
    p_positive = _masked_softmax(scores, positive)
    maximum_all = np.max(np.where(active, scores, -np.inf), axis=1)
    maximum_positive = np.max(np.where(positive, scores, -np.inf), axis=1)
    lse_all = maximum_all + np.log(np.sum(np.where(active, np.exp(scores - maximum_all[:, None]), 0.0), axis=1))
    lse_positive = maximum_positive + np.log(
        np.sum(np.where(positive, np.exp(scores - maximum_positive[:, None]), 0.0), axis=1)
    )
    listwise = float(np.mean(lse_all - lse_positive))
    gradient = (p_all - p_positive) / len(scores)
    pair_loss = 0.0
    pair_gradient = np.zeros_like(scores)
    if pairwise_weight:
        for i in range(len(scores)):
            positives = np.flatnonzero(positive[i])
            negatives = np.flatnonzero(active[i] & ~positive[i])
            count = len(positives) * len(negatives)
            if not count:
                continue
            scale = 1.0 / (len(scores) * count)
            for pos in positives:
                for neg in negatives:
                    delta = float(scores[i, neg] - scores[i, pos])
                    pair_loss += scale * float(np.logaddexp(0.0, delta))
                    sigmoid = 1.0 / (1.0 + math.exp(-delta)) if delta >= 0 else math.exp(delta) / (1 + math.exp(delta))
                    pair_gradient[i, pos] -= scale * sigmoid
                    pair_gradient[i, neg] += scale * sigmoid
        gradient += pairwise_weight * pair_gradient
    total = listwise + pairwise_weight * pair_loss
    return total, gradient, {"listwise": listwise, "pairwise": pair_loss}


def weighted_bce_score_gradient(
    scores: np.ndarray,
    labels: np.ndarray,
    hard_mask: np.ndarray,
) -> tuple[float, np.ndarray, dict]:
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.float64)
    hard_mask = np.asarray(hard_mask, dtype=bool)
    row_weights = np.zeros(len(scores), dtype=np.float64)
    hard_count = max(int(hard_mask.sum()), 1)
    easy_count = max(int((~hard_mask).sum()), 1)
    if hard_mask.any() and (~hard_mask).any():
        row_weights[hard_mask] = 0.5 / hard_count
        row_weights[~hard_mask] = 0.5 / easy_count
    else:
        row_weights[:] = 1.0 / len(scores)
    element_weights = row_weights[:, None] / scores.shape[1]
    losses = np.logaddexp(0.0, scores) - labels * scores
    loss = float(np.sum(element_weights * losses))
    gradient = element_weights * (_sigmoid(scores) - labels)
    hard_loss = float(np.mean(losses[hard_mask])) if hard_mask.any() else 0.0
    easy_loss = float(np.mean(losses[~hard_mask])) if (~hard_mask).any() else 0.0
    return loss, gradient, {"bce_hard": hard_loss, "bce_easy": easy_loss}


def mlp_forward(features: np.ndarray, model: dict) -> tuple[np.ndarray, np.ndarray]:
    hidden = np.tanh(np.einsum("nkd,dh->nkh", features, model["w1"]) + model["b1"])
    scores = np.einsum("nkh,h->nk", hidden, model["w2"]) + float(model["b2"])
    return scores, hidden


def mlp_parameter_gradients(
    features: np.ndarray, hidden: np.ndarray, score_gradient: np.ndarray, model: dict
) -> tuple[dict, np.ndarray]:
    grad_w2 = np.einsum("nkh,nk->h", hidden, score_gradient)
    grad_b2 = float(score_gradient.sum())
    grad_hidden = score_gradient[:, :, None] * model["w2"][None, None, :]
    grad_pre = grad_hidden * (1.0 - hidden * hidden)
    grad_w1 = np.einsum("nkd,nkh->dh", features, grad_pre)
    grad_b1 = grad_pre.sum(axis=(0, 1))
    grad_features = np.einsum("nkh,dh->nkd", grad_pre, model["w1"])
    return {"w1": grad_w1, "b1": grad_b1, "w2": grad_w2, "b2": grad_b2}, grad_features


def _adam_update(parameters: dict, gradients: dict, state: dict, step: int, learning_rate: float) -> None:
    for name, value in parameters.items():
        gradient = np.asarray(gradients[name], dtype=np.float64)
        norm = float(np.linalg.norm(gradient))
        if norm > 5.0:
            gradient = gradient * (5.0 / norm)
        state["m"][name] = 0.9 * state["m"][name] + 0.1 * gradient
        state["v"][name] = 0.999 * state["v"][name] + 0.001 * gradient * gradient
        m_hat = state["m"][name] / (1.0 - 0.9**step)
        v_hat = state["v"][name] / (1.0 - 0.999**step)
        parameters[name] -= learning_rate * m_hat / (np.sqrt(v_hat) + 1e-8)


def train_candidate_model(
    centered: np.ndarray,
    answers: list[list[str | None]],
    stage2_hand: np.ndarray,
    labels: np.ndarray,
    valid: np.ndarray,
    hard_mask: np.ndarray,
    train_indices: np.ndarray,
    config: dict,
    seed: int,
) -> dict:
    train_indices = np.asarray(train_indices, dtype=np.int64)
    hand_mean = stage2_hand[train_indices].reshape(-1, stage2_hand.shape[-1]).mean(axis=0)
    hand_scale = stage2_hand[train_indices].reshape(-1, stage2_hand.shape[-1]).std(axis=0)
    hand_scale = np.where(hand_scale < 1e-8, 1.0, hand_scale)
    projection_dim = int(config["projection_dim"])
    mode = config["mode"]
    projection = initialize_projection(centered.shape[-1], projection_dim, seed)
    feature_dim = interaction_feature_dim(projection_dim, mode)
    rng = np.random.default_rng(seed + feature_dim + int(config["hidden_dim"]))
    parameters: dict[str, np.ndarray] = {"projection": projection}
    if mode == "linear_lowrank":
        parameters["linear"] = rng.normal(scale=1.0 / math.sqrt(feature_dim), size=feature_dim)
        parameters["bias"] = np.asarray(0.0)
    else:
        hidden_dim = int(config["hidden_dim"])
        parameters["w1"] = rng.normal(scale=1.0 / math.sqrt(feature_dim), size=(feature_dim, hidden_dim))
        parameters["b1"] = np.zeros(hidden_dim)
        parameters["w2"] = rng.normal(scale=1.0 / math.sqrt(hidden_dim), size=hidden_dim)
        parameters["b2"] = np.asarray(0.0)
    state = {
        "m": {name: np.zeros_like(value, dtype=np.float64) for name, value in parameters.items()},
        "v": {name: np.zeros_like(value, dtype=np.float64) for name, value in parameters.items()},
    }
    epochs = int(config.get("epochs", 16))
    batch_size = int(config.get("batch_size", 256))
    pairwise_weight = float(config.get("pairwise_weight", 0.0))
    bce_weight = float(config.get("bce_weight", 0.0))
    learning_rate = float(config.get("learning_rate", 0.01))
    l2 = float(config.get("l2", 1e-4))
    history = []
    step = 0
    for epoch in range(epochs):
        if bce_weight:
            order = rng.permutation(train_indices)
        else:
            order = rng.permutation(train_indices[hard_mask[train_indices]])
        totals = {"loss": 0.0, "listwise": 0.0, "pairwise": 0.0, "bce_hard": 0.0, "bce_easy": 0.0}
        batches = 0
        for start in range(0, len(order), batch_size):
            indices = order[start:start + batch_size]
            if not len(indices):
                continue
            local_answers = [answers[int(i)] for i in indices]
            hand = (stage2_hand[indices] - hand_mean) / hand_scale
            z = project(centered[indices], parameters["projection"])
            features = candidate_features_from_projected(z, local_answers, hand, mode)
            if mode == "linear_lowrank":
                scores = np.einsum("nkd,d->nk", features, parameters["linear"]) + float(parameters["bias"])
            else:
                scores, hidden = mlp_forward(features, parameters)
            score_gradient = np.zeros_like(scores)
            hard_local = hard_mask[indices]
            rank_info = {"listwise": 0.0, "pairwise": 0.0}
            rank_loss = 0.0
            if hard_local.any():
                rank_loss, rank_gradient, rank_info = ranking_score_loss_gradient(
                    scores[hard_local], labels[indices][hard_local], valid[indices][hard_local], pairwise_weight
                )
                score_gradient[hard_local] += rank_gradient
            bce_loss = 0.0
            bce_info = {"bce_hard": 0.0, "bce_easy": 0.0}
            if bce_weight:
                bce_loss, bce_gradient, bce_info = weighted_bce_score_gradient(
                    scores, labels[indices], hard_local
                )
                score_gradient += bce_weight * bce_gradient
            gradients: dict[str, np.ndarray] = {}
            if mode == "linear_lowrank":
                gradients["linear"] = np.einsum("nkd,nk->d", features, score_gradient) + l2 * parameters["linear"]
                gradients["bias"] = np.asarray(score_gradient.sum())
                grad_features = score_gradient[:, :, None] * parameters["linear"][None, None, :]
            else:
                mlp_grad, grad_features = mlp_parameter_gradients(features, hidden, score_gradient, parameters)
                gradients.update(mlp_grad)
                for name in ("w1", "w2"):
                    gradients[name] += l2 * parameters[name]
            grad_z = candidate_feature_backward(grad_features, z, local_answers, mode)
            gradients["projection"] = np.einsum("nkd,nkr->dr", centered[indices], grad_z) + l2 * parameters["projection"]
            total_loss = rank_loss + bce_weight * bce_loss + 0.5 * l2 * sum(
                float(np.sum(parameters[name] ** 2))
                for name in parameters if name not in {"bias", "b1", "b2"}
            )
            step += 1
            _adam_update(parameters, gradients, state, step, learning_rate)
            totals["loss"] += total_loss
            for name in rank_info:
                totals[name] += rank_info[name]
            for name in bce_info:
                totals[name] += bce_info[name]
            batches += 1
        history.append({"epoch": epoch + 1, **{name: value / max(batches, 1) for name, value in totals.items()}})
    return {
        **parameters,
        "hand_mean": hand_mean,
        "hand_scale": hand_scale,
        "mode": mode,
        "projection_dim": projection_dim,
        "hidden_dim": int(config["hidden_dim"]),
        "pairwise_weight": pairwise_weight,
        "bce_weight": bce_weight,
        "loss_history": history,
    }


def score_candidate_model(
    model: dict,
    centered: np.ndarray,
    answers: list[list[str | None]],
    stage2_hand: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    hand = (stage2_hand - model["hand_mean"]) / model["hand_scale"]
    z = project(centered, model["projection"])
    features = candidate_features_from_projected(z, answers, hand, model["mode"])
    if model["mode"] == "linear_lowrank":
        scores = np.einsum("nkd,d->nk", features, model["linear"]) + float(model["bias"])
    else:
        scores = mlp_forward(features, model)[0]
    return scores, z


def _select_group_features(features: np.ndarray, scalar_dim: int, mode: str) -> np.ndarray:
    if mode == "scalar_only":
        return features[:, :, :scalar_dim]
    if mode == "projected_only":
        return np.concatenate([features[:, :, :2], features[:, :, scalar_dim:]], axis=2)
    if mode == "combined":
        return features
    raise ValueError(f"Unknown group mode: {mode}")


def train_group_model(
    features: np.ndarray,
    labels: np.ndarray,
    valid: np.ndarray,
    train_indices: np.ndarray,
    scalar_dim: int,
    config: dict,
    seed: int,
) -> dict:
    selected = _select_group_features(np.asarray(features, dtype=np.float64), scalar_dim, config["mode"])
    flat = selected[train_indices][valid[train_indices]]
    mean = flat.mean(axis=0); scale = flat.std(axis=0)
    scale = np.where(scale < 1e-8, 1.0, scale)
    x = (selected[train_indices] - mean) / scale
    y = labels[train_indices]; active = valid[train_indices]
    rng = np.random.default_rng(seed + selected.shape[-1])
    hidden_dim = int(config.get("hidden_dim", 16))
    model = {
        "w1": rng.normal(scale=1.0 / math.sqrt(selected.shape[-1]), size=(selected.shape[-1], hidden_dim)),
        "b1": np.zeros(hidden_dim),
        "w2": rng.normal(scale=1.0 / math.sqrt(hidden_dim), size=hidden_dim),
        "b2": np.asarray(0.0),
    }
    state = {"m": {k: np.zeros_like(v) for k, v in model.items()}, "v": {k: np.zeros_like(v) for k, v in model.items()}}
    history = []
    l2 = float(config.get("l2", 1e-3)); learning_rate = float(config.get("learning_rate", 0.02))
    for step in range(1, int(config.get("steps", 140)) + 1):
        scores, hidden = mlp_forward(x, model)
        loss, score_gradient, parts = ranking_score_loss_gradient(
            scores, y, active, float(config.get("pairwise_weight", 0.0))
        )
        gradients, _ = mlp_parameter_gradients(x, hidden, score_gradient, model)
        for name in ("w1", "w2"):
            loss += 0.5 * l2 * float(np.sum(model[name] ** 2))
            gradients[name] += l2 * model[name]
        _adam_update(model, gradients, state, step, learning_rate)
        if step == 1 or step % 10 == 0:
            history.append({"step": step, "loss": loss, **parts})
    return {**model, "mean": mean, "scale": scale, "mode": config["mode"],
            "scalar_dim": int(scalar_dim), "pairwise_weight": float(config.get("pairwise_weight", 0.0)),
            "loss_history": history}


def score_group_model(model: dict, features: np.ndarray, valid: np.ndarray) -> np.ndarray:
    selected = _select_group_features(features, int(model["scalar_dim"]), model["mode"])
    transformed = (selected - model["mean"]) / model["scale"]
    scores = mlp_forward(transformed, model)[0]
    return np.where(valid, scores, -np.inf)


def fit_binary_logistic(
    features: np.ndarray,
    labels: np.ndarray,
    train_mask: np.ndarray,
    sample_weights: np.ndarray | None = None,
    l2: float = 0.1,
    steps: int = 300,
) -> dict:
    mask = np.asarray(train_mask, dtype=bool)
    raw = np.asarray(features[mask], dtype=np.float64); y = np.asarray(labels[mask], dtype=float)
    mean = raw.mean(axis=0); scale = raw.std(axis=0); scale = np.where(scale < 1e-8, 1.0, scale)
    x = np.column_stack([(raw - mean) / scale, np.ones(len(raw))])
    weights = np.zeros(x.shape[1]); m = np.zeros_like(weights); v = np.zeros_like(weights)
    sw = np.ones(len(raw)) if sample_weights is None else np.asarray(sample_weights[mask], dtype=float)
    sw /= sw.sum()
    history = []
    for step in range(1, steps + 1):
        logits = x @ weights; probabilities = _sigmoid(logits)
        loss = float(np.sum(sw * (np.logaddexp(0, logits) - y * logits)) + 0.5 * l2 * np.sum(weights[:-1] ** 2))
        gradient = x.T @ (sw * (probabilities - y)); gradient[:-1] += l2 * weights[:-1]
        m = 0.9 * m + 0.1 * gradient; v = 0.999 * v + 0.001 * gradient * gradient
        weights -= 0.03 * (m / (1 - 0.9**step)) / (np.sqrt(v / (1 - 0.999**step)) + 1e-8)
        if step == 1 or step % 25 == 0:
            history.append({"step": step, "loss": loss})
    return {"weights": weights, "mean": mean, "scale": scale, "l2": l2, "loss_history": history}


def binary_probabilities(model: dict, features: np.ndarray) -> np.ndarray:
    x = (np.asarray(features, dtype=float) - model["mean"]) / model["scale"]
    return _sigmoid(x @ model["weights"][:-1] + model["weights"][-1])


def fit_multinomial_gate(
    features: np.ndarray,
    positive_branches: np.ndarray,
    train_mask: np.ndarray,
    l2: float = 0.1,
    steps: int = 400,
) -> dict:
    positive_branches = np.asarray(positive_branches, dtype=bool)
    mask = np.asarray(train_mask, dtype=bool) & positive_branches.any(axis=1)
    raw = np.asarray(features[mask], dtype=float); positive = np.asarray(positive_branches[mask], dtype=bool)
    if not len(raw):
        raise ValueError("Gate training needs at least one row with a correct branch")
    mean = raw.mean(axis=0); scale = raw.std(axis=0); scale = np.where(scale < 1e-8, 1.0, scale)
    x = np.column_stack([(raw - mean) / scale, np.ones(len(raw))])
    weights = np.zeros((x.shape[1], 3)); m = np.zeros_like(weights); v = np.zeros_like(weights)
    history = []
    for step in range(1, steps + 1):
        logits = x @ weights
        p_all = _masked_softmax(logits, np.ones_like(positive, dtype=bool))
        p_positive = _masked_softmax(logits, positive)
        max_all = logits.max(axis=1); max_pos = np.max(np.where(positive, logits, -np.inf), axis=1)
        loss = float(np.mean(
            max_all + np.log(np.exp(logits - max_all[:, None]).sum(axis=1))
            - max_pos - np.log(np.where(positive, np.exp(logits - max_pos[:, None]), 0).sum(axis=1))
        ) + 0.5 * l2 * np.sum(weights[:-1] ** 2))
        gradient = x.T @ ((p_all - p_positive) / len(x)); gradient[:-1] += l2 * weights[:-1]
        m = 0.9 * m + 0.1 * gradient; v = 0.999 * v + 0.001 * gradient * gradient
        weights -= 0.03 * (m / (1 - 0.9**step)) / (np.sqrt(v / (1 - 0.999**step)) + 1e-8)
        if step == 1 or step % 25 == 0:
            history.append({"step": step, "loss": loss})
    return {"weights": weights, "mean": mean, "scale": scale, "l2": l2, "loss_history": history}


def gate_probabilities(model: dict, features: np.ndarray) -> np.ndarray:
    x = np.column_stack([
        (np.asarray(features, dtype=float) - model["mean"]) / model["scale"],
        np.ones(len(features)),
    ])
    return _masked_softmax(x @ model["weights"], np.ones((len(features), 3), dtype=bool))


def calibration_metrics(labels: np.ndarray, probabilities: np.ndarray) -> dict:
    labels = np.asarray(labels, dtype=float); p = np.clip(np.asarray(probabilities, dtype=float), 1e-8, 1 - 1e-8)
    nll = float(np.mean(-(labels * np.log(p) + (1 - labels) * np.log(1 - p))))
    ece = 0.0
    bins = []
    for lower in np.linspace(0, 0.9, 10):
        upper = lower + 0.1
        mask = (p >= lower) & (p < upper if upper < 1 else p <= upper)
        if mask.any():
            accuracy = float(labels[mask].mean()); confidence = float(p[mask].mean())
            ece += float(mask.mean()) * abs(accuracy - confidence)
            bins.append({"lower": float(lower), "upper": float(upper), "n": int(mask.sum()),
                         "accuracy": accuracy, "confidence": confidence})
    order = np.argsort(-p); sorted_errors = 1.0 - labels[order]
    risks = np.cumsum(sorted_errors) / np.arange(1, len(labels) + 1)
    aurc = float(np.mean(risks))
    quantile_bins = []
    for indices in np.array_split(np.argsort(p), 5):
        quantile_bins.append({"n": len(indices), "accuracy": float(labels[indices].mean()),
                              "confidence": float(p[indices].mean()),
                              "min_probability": float(p[indices].min()), "max_probability": float(p[indices].max())})
    return {"nll": nll, "ece": float(ece), "aurc": aurc, "bins": bins, "quintiles": quantile_bins}


def _float32_copy(value):
    return np.asarray(value, dtype=np.float32)


def save_bundle(directory: Path, bundle: dict, config: dict) -> None:
    directory = Path(directory); directory.mkdir(parents=True, exist_ok=True)
    arrays = {}
    for prefix, model_name in (("candidate", "candidate_model"), ("group", "group_model"),
                               ("plurality_cal", "plurality_calibrator"),
                               ("candidate_cal", "candidate_calibrator"), ("group_cal", "group_calibrator"),
                               ("selected_cal", "selected_calibrator"), ("gate", "gate_model")):
        model = bundle[model_name]
        for key, value in model.items():
            if isinstance(value, np.ndarray):
                arrays[f"{prefix}__{key}"] = _float32_copy(value)
    np.savez(directory / "model.npz", **arrays)
    metadata = {
        "format": "dase7506-mp2-creo-3w-v1",
        **config,
        "model_metadata": {},
    }
    for model_name in ("candidate_model", "group_model", "plurality_calibrator", "candidate_calibrator", "group_calibrator", "selected_calibrator", "gate_model"):
        metadata["model_metadata"][model_name] = {
            key: value for key, value in bundle[model_name].items()
            if not isinstance(value, np.ndarray) and key != "loss_history"
        }
    (directory / "config.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    (directory / "loss_histories.json").write_text(json.dumps({
        name: bundle[name].get("loss_history", []) for name in metadata["model_metadata"]
    }, indent=2) + "\n", encoding="utf-8")


def load_bundle(directory: Path) -> dict:
    directory = Path(directory)
    config_path = directory / "config.json"; model_path = directory / "model.npz"
    if not config_path.is_file() or not model_path.is_file():
        raise FileNotFoundError(f"Stage 4 checkpoint missing config.json or model.npz: {directory}")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("format") != "dase7506-mp2-creo-3w-v1":
        raise ValueError("Unsupported Stage 4 checkpoint format")
    result = {"config": config}
    prefix_map = {
        "candidate": "candidate_model", "group": "group_model", "plurality_cal": "plurality_calibrator",
        "candidate_cal": "candidate_calibrator", "group_cal": "group_calibrator",
        "selected_cal": "selected_calibrator", "gate": "gate_model",
    }
    with np.load(model_path, allow_pickle=False) as data:
        for prefix, model_name in prefix_map.items():
            model = dict(config["model_metadata"][model_name])
            marker = prefix + "__"
            for key in data.files:
                if key.startswith(marker):
                    model[key[len(marker):]] = data[key].astype(np.float64)
            result[model_name] = model
    return result
