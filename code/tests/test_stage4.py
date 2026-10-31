import copy
import tempfile
import unittest
from pathlib import Path

import numpy as np

import common
import stage4_features
import submission_stage4
from selector_features import extract_feature_blocks
from stage4_features import (
    build_answer_groups,
    candidate_feature_backward,
    candidate_features_from_projected,
    centered_embeddings,
    extract_group_features,
    initialize_projection,
    project,
)
from stage4_model import (
    load_bundle,
    mlp_forward,
    mlp_parameter_gradients,
    ranking_score_loss_gradient,
    save_bundle,
    weighted_bce_score_gradient,
)


def bank_of(n=4):
    texts = [
        "2 + 3 = 5. Final Answer: \\boxed{5}",
        "Carefully count the items. #### 5",
        "2 + 3 = 6. Final Answer: \\boxed{6}",
        "No parseable final result.",
    ]
    return [{
        "id": f"arbitrary::{i * 91 + 7}",
        "question": f"There are 2 red and 3 blue balls. How many? version {i}",
        "candidates": [{"index": j, "text": text} for j, text in enumerate(texts)],
    } for i in range(n)]


class Stage4Tests(unittest.TestCase):
    def setUp(self):
        self.bank = bank_of()
        self.rng = np.random.default_rng(13)
        self.embeddings = self.rng.normal(size=(4, 4, 1024)).astype(np.float32)

    def test_official_parser_binding(self):
        self.assertIs(stage4_features.parse_answer, common.parse_answer)

    def test_candidate_permutation_equivariance(self):
        blocks = extract_feature_blocks(self.bank, self.embeddings)
        answers = blocks["answers"]
        centered = centered_embeddings(self.embeddings)
        projection = initialize_projection(1024, 8)
        original = candidate_features_from_projected(
            project(centered, projection), answers, blocks["stage2_hand"], "interaction"
        )
        permutation = np.asarray([2, 0, 3, 1])
        permuted_bank = copy.deepcopy(self.bank)
        for row in permuted_bank:
            row["candidates"] = [row["candidates"][int(j)] for j in permutation]
            for j, candidate in enumerate(row["candidates"]):
                candidate["index"] = j
        permuted_embeddings = self.embeddings[:, permutation]
        permuted_blocks = extract_feature_blocks(permuted_bank, permuted_embeddings)
        permuted_features = candidate_features_from_projected(
            project(centered_embeddings(permuted_embeddings), projection),
            permuted_blocks["answers"], permuted_blocks["stage2_hand"], "interaction",
        )
        np.testing.assert_allclose(permuted_features, original[:, permutation], atol=1e-10)

    def test_answer_group_permutation_equivariance(self):
        blocks = extract_feature_blocks(self.bank, self.embeddings)
        groups = build_answer_groups(self.bank, blocks["answers"])
        projected = project(centered_embeddings(self.embeddings), initialize_projection(1024, 8))
        scores = self.rng.normal(size=(4, 4))
        original = extract_group_features(self.bank, groups, projected, scores)
        permutation = np.asarray([2, 0, 3, 1])
        permuted_bank = copy.deepcopy(self.bank)
        for row in permuted_bank:
            row["candidates"] = [row["candidates"][int(j)] for j in permutation]
            for j, candidate in enumerate(row["candidates"]):
                candidate["index"] = j
        permuted_blocks = extract_feature_blocks(permuted_bank, self.embeddings[:, permutation])
        permuted_groups = build_answer_groups(permuted_bank, permuted_blocks["answers"])
        permuted_features = extract_group_features(
            permuted_bank, permuted_groups, projected[:, permutation], scores[:, permutation]
        )
        for i in range(len(self.bank)):
            first = {answer: original["features"][i, g] for g, answer in enumerate(groups["answers"][i])}
            second = {answer: permuted_features["features"][i, g] for g, answer in enumerate(permuted_groups["answers"][i])}
            self.assertEqual(set(first), set(second))
            for answer in first:
                np.testing.assert_allclose(first[answer], second[answer], atol=1e-10)

    def test_interaction_projection_gradient(self):
        bank = bank_of(1)
        embeddings = self.rng.normal(size=(1, 4, 6))
        centered = embeddings - embeddings.mean(axis=1, keepdims=True)
        projection = self.rng.normal(scale=0.2, size=(6, 3))
        hand = self.rng.normal(size=(1, 4, 32))
        answers = [["5", "5", "6", None]]
        z = project(centered, projection)
        features = candidate_features_from_projected(z, answers, hand)
        upstream = self.rng.normal(size=features.shape)
        grad_z = candidate_feature_backward(upstream, z, answers)
        analytic = np.einsum("nkd,nkr->dr", centered, grad_z)
        numerical = np.zeros_like(projection)
        eps = 1e-6
        for r in range(projection.shape[0]):
            for c in range(projection.shape[1]):
                plus = projection.copy(); plus[r, c] += eps
                minus = projection.copy(); minus[r, c] -= eps
                fp = candidate_features_from_projected(project(centered, plus), answers, hand)
                fm = candidate_features_from_projected(project(centered, minus), answers, hand)
                numerical[r, c] = (np.sum(fp * upstream) - np.sum(fm * upstream)) / (2 * eps)
        self.assertLess(float(np.max(np.abs(analytic - numerical))), 2e-5)

    def test_listwise_pairwise_and_bce_gradients(self):
        scores = self.rng.normal(size=(3, 4))
        labels = np.asarray([[1, 1, 0, 0], [0, 1, 0, 1], [1, 0, 0, 0]], dtype=bool)
        valid = np.asarray([[1, 1, 1, 1], [1, 1, 0, 1], [1, 1, 1, 0]], dtype=bool)
        loss, gradient, _ = ranking_score_loss_gradient(scores, labels, valid, 0.1)
        numerical = np.zeros_like(scores); eps = 1e-6
        for i in range(3):
            for j in range(4):
                plus = scores.copy(); plus[i, j] += eps
                minus = scores.copy(); minus[i, j] -= eps
                numerical[i, j] = (
                    ranking_score_loss_gradient(plus, labels, valid, 0.1)[0]
                    - ranking_score_loss_gradient(minus, labels, valid, 0.1)[0]
                ) / (2 * eps)
        self.assertTrue(np.isfinite(loss))
        self.assertLess(float(np.max(np.abs(gradient - numerical))), 2e-6)

        bce_loss, bce_gradient, _ = weighted_bce_score_gradient(scores, labels, np.asarray([1, 1, 0]))
        for i in range(3):
            for j in range(4):
                plus = scores.copy(); plus[i, j] += eps
                minus = scores.copy(); minus[i, j] -= eps
                numerical[i, j] = (
                    weighted_bce_score_gradient(plus, labels, np.asarray([1, 1, 0]))[0]
                    - weighted_bce_score_gradient(minus, labels, np.asarray([1, 1, 0]))[0]
                ) / (2 * eps)
        self.assertTrue(np.isfinite(bce_loss))
        self.assertLess(float(np.max(np.abs(bce_gradient - numerical))), 2e-6)

    def test_mlp_finite_difference_gradient(self):
        features = self.rng.normal(size=(2, 4, 5))
        model = {
            "w1": self.rng.normal(scale=0.2, size=(5, 3)), "b1": np.zeros(3),
            "w2": self.rng.normal(scale=0.2, size=3), "b2": np.asarray(0.0),
        }
        scores, hidden = mlp_forward(features, model)
        upstream = self.rng.normal(size=scores.shape)
        gradients, _ = mlp_parameter_gradients(features, hidden, upstream, model)
        eps = 1e-6
        numerical = np.zeros_like(model["w1"])
        for i in range(5):
            for j in range(3):
                plus = {k: np.array(v, copy=True) for k, v in model.items()}; plus["w1"][i, j] += eps
                minus = {k: np.array(v, copy=True) for k, v in model.items()}; minus["w1"][i, j] -= eps
                numerical[i, j] = (
                    np.sum(mlp_forward(features, plus)[0] * upstream)
                    - np.sum(mlp_forward(features, minus)[0] * upstream)
                ) / (2 * eps)
        self.assertLess(float(np.max(np.abs(gradients["w1"] - numerical))), 2e-6)

    def test_arbitrary_length_empty_and_unparseable(self):
        bank = bank_of(7)
        for row in bank:
            row["id"] = "unexpected/" + row["id"]
            for candidate in row["candidates"]:
                candidate["text"] = ""
        embeddings = np.zeros((7, 4, 1024), dtype=np.float32)
        blocks = extract_feature_blocks(bank, embeddings)
        groups = build_answer_groups(bank, blocks["answers"])
        features = candidate_features_from_projected(
            project(centered_embeddings(embeddings), initialize_projection(1024, 8)),
            blocks["answers"], blocks["stage2_hand"],
        )
        self.assertEqual(groups["valid"].sum(axis=1).tolist(), [4] * 7)
        self.assertTrue(np.isfinite(features).all())
        checkpoint = Path(__file__).resolve().parents[1] / "runs" / "stage4" / "provisional_bundle"
        if checkpoint.is_dir():
            choices = submission_stage4.predict(bank, {"embeddings": embeddings}, checkpoint)
            self.assertEqual(len(choices), 7)
            self.assertTrue(all(type(choice) is int and 0 <= choice < 4 for choice in choices))

    def test_checkpoint_roundtrip_and_float32_error(self):
        candidate = {
            "projection": self.rng.normal(size=(5, 2)), "w1": self.rng.normal(size=(6, 3)),
            "b1": np.zeros(3), "w2": self.rng.normal(size=3), "b2": np.asarray(0.0),
            "hand_mean": np.zeros(32), "hand_scale": np.ones(32), "mode": "interaction",
            "projection_dim": 2, "hidden_dim": 3, "pairwise_weight": 0.1, "bce_weight": 0.05,
            "loss_history": [],
        }
        group = {"w1": self.rng.normal(size=(4, 2)), "b1": np.zeros(2), "w2": self.rng.normal(size=2),
                 "b2": np.asarray(0.0), "mean": np.zeros(4), "scale": np.ones(4), "mode": "combined",
                 "scalar_dim": 2, "pairwise_weight": 0.0, "loss_history": []}
        logistic = {"weights": self.rng.normal(size=5), "mean": np.zeros(4), "scale": np.ones(4),
                    "l2": 0.1, "loss_history": []}
        gate = {"weights": self.rng.normal(size=(5, 3)), "mean": np.zeros(4), "scale": np.ones(4),
                "l2": 0.1, "loss_history": []}
        bundle = {"candidate_model": candidate, "group_model": group,
                  "plurality_calibrator": logistic,
                  "candidate_calibrator": logistic, "group_calibrator": logistic,
                  "selected_calibrator": logistic, "gate_model": gate}
        with tempfile.TemporaryDirectory() as tmp:
            save_bundle(Path(tmp), bundle, {"seed": 20261022})
            loaded = load_bundle(Path(tmp))
            for key in ("projection", "w1", "w2"):
                np.testing.assert_allclose(candidate[key], loaded["candidate_model"][key], rtol=2e-7, atol=2e-7)
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(FileNotFoundError):
                load_bundle(Path(tmp) / "missing")

    def test_submission_missing_checkpoint_and_python_ints(self):
        with self.assertRaises(FileNotFoundError):
            submission_stage4.predict(self.bank, {"embeddings": self.embeddings}, None)
        checkpoint = Path(__file__).resolve().parents[1] / "runs" / "stage4" / "provisional_bundle"
        if checkpoint.is_dir():
            choices = submission_stage4.predict(
                self.bank, {"embeddings": self.embeddings}, checkpoint
            )
            self.assertEqual(len(choices), len(self.bank))
            self.assertTrue(all(type(choice) is int and 0 <= choice < 4 for choice in choices))


if __name__ == "__main__":
    unittest.main()
