import copy
import tempfile
import unittest
from pathlib import Path

import numpy as np

import common
import selector_features
from selector_features import extract_feature_blocks
from selector_model import (
    cluster_choices_from_scores,
    choices_from_scores,
    fit_transform,
    load_model,
    loss_and_gradient,
    save_model,
    score_model,
    train_linear_model,
)
import submission
import submission_stage3


def synthetic_bank(n=3):
    rows = []
    templates = [
        "1 + 1 = 2. Final Answer: \\boxed{2}",
        "We compute carefully. #### 2",
        "1 + 1 = 3. Final Answer: \\boxed{3}",
        "No numeric answer is available.",
    ]
    for i in range(n):
        rows.append({
            "id": f"arbitrary-{1000 + i * 17}",
            "question": f"What is 1 + 1? Variant {i + 5}",
            "candidates": [
                {"index": j, "text": text + (" extra" * i)} for j, text in enumerate(templates)
            ],
        })
    return rows


class SelectorTests(unittest.TestCase):
    def setUp(self):
        self.bank = synthetic_bank(3)
        rng = np.random.default_rng(11)
        self.embeddings = rng.normal(size=(3, 4, 1024)).astype(np.float32)

    def _trained_model(self):
        blocks = extract_feature_blocks(self.bank, self.embeddings)
        labels = np.asarray([[1, 1, 0, 0]] * len(self.bank), dtype=bool)
        indices = np.arange(len(self.bank))
        features, meta = fit_transform(blocks, "hand_centered", indices)
        model = train_linear_model(
            features, labels, blocks["valid"], meta,
            hand_l2=1e-2, embedding_l2=1e-1, seed=20261022, max_steps=20,
        )
        return blocks, model

    def test_official_parser_is_used(self):
        self.assertIs(selector_features.parse_answer, common.parse_answer)

    def test_multi_positive_loss_and_gradient(self):
        rng = np.random.default_rng(7)
        features = rng.normal(size=(3, 4, 5))
        labels = np.asarray([
            [1, 1, 0, 0], [0, 1, 0, 1], [1, 0, 0, 0]
        ], dtype=bool)
        valid = np.asarray([
            [1, 1, 1, 1], [1, 1, 0, 1], [1, 1, 1, 0]
        ], dtype=bool)
        weights = rng.normal(scale=0.1, size=5)
        l2 = np.full(5, 0.03)
        loss, grad = loss_and_gradient(weights, features, labels, valid, l2)
        self.assertTrue(np.isfinite(loss))
        eps = 1e-6
        numerical = np.zeros_like(weights)
        for d in range(len(weights)):
            plus = weights.copy(); plus[d] += eps
            minus = weights.copy(); minus[d] -= eps
            numerical[d] = (
                loss_and_gradient(plus, features, labels, valid, l2)[0]
                - loss_and_gradient(minus, features, labels, valid, l2)[0]
            ) / (2 * eps)
        self.assertLess(float(np.max(np.abs(grad - numerical))), 2e-6)

    def test_weighted_hybrid_loss_and_gradient(self):
        rng = np.random.default_rng(19)
        features = rng.normal(size=(4, 4, 6))
        labels = np.asarray([
            [1, 0, 0, 0], [1, 1, 0, 0], [0, 1, 0, 1], [1, 1, 1, 0]
        ], dtype=bool)
        valid = np.ones((4, 4), dtype=bool)
        weights = rng.normal(scale=0.08, size=6)
        group_weights = np.asarray([2.0, 1.0, 1.5, 0.75])
        l2 = np.full(6, 0.02)
        kwargs = {"group_weights": group_weights, "pairwise_weight": 0.25}
        loss, grad = loss_and_gradient(weights, features, labels, valid, l2, **kwargs)
        self.assertTrue(np.isfinite(loss))
        eps = 1e-6
        numerical = np.zeros_like(weights)
        for d in range(len(weights)):
            plus = weights.copy(); plus[d] += eps
            minus = weights.copy(); minus[d] -= eps
            numerical[d] = (
                loss_and_gradient(plus, features, labels, valid, l2, **kwargs)[0]
                - loss_and_gradient(minus, features, labels, valid, l2, **kwargs)[0]
            ) / (2 * eps)
        self.assertLess(float(np.max(np.abs(grad - numerical))), 3e-6)

    def test_stage2_hand_cap(self):
        blocks = extract_feature_blocks(self.bank, self.embeddings)
        self.assertEqual(blocks["stage2_hand"].shape, (3, 4, 32))
        self.assertTrue(np.isfinite(blocks["stage2_hand"]).all())

    def test_permutation_equivariance(self):
        blocks, model = self._trained_model()
        original_scores = score_model(model, blocks)
        original_choices = choices_from_scores(self.bank, original_scores, blocks["valid"])
        permutation = np.asarray([2, 0, 3, 1])
        inverse = np.argsort(permutation)
        permuted_bank = copy.deepcopy(self.bank)
        for row in permuted_bank:
            row["candidates"] = [row["candidates"][int(j)] for j in permutation]
            for j, candidate in enumerate(row["candidates"]):
                candidate["index"] = j
        permuted_embeddings = self.embeddings[:, permutation]
        permuted_blocks = extract_feature_blocks(permuted_bank, permuted_embeddings)
        permuted_scores = score_model(model, permuted_blocks)
        np.testing.assert_allclose(permuted_scores, original_scores[:, permutation], atol=1e-9)
        permuted_choices = choices_from_scores(
            permuted_bank, permuted_scores, permuted_blocks["valid"]
        )
        expected = [int(inverse[choice]) for choice in original_choices]
        self.assertEqual(permuted_choices, expected)

        original_cluster = cluster_choices_from_scores(
            self.bank, original_scores, blocks["valid"],
            rule="cluster_max_plus_log_count", alpha=0.5,
        )
        permuted_cluster = cluster_choices_from_scores(
            permuted_bank, permuted_scores, permuted_blocks["valid"],
            rule="cluster_max_plus_log_count", alpha=0.5,
        )
        self.assertEqual(permuted_cluster, [int(inverse[choice]) for choice in original_cluster])

    def test_checkpoint_roundtrip_and_python_ints(self):
        blocks, model = self._trained_model()
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint = Path(tmp) / "bundle"
            save_model(model, checkpoint)
            restored = load_model(checkpoint)
            original_scores = score_model(model, blocks)
            restored_scores = score_model(restored, blocks)
            np.testing.assert_allclose(original_scores, restored_scores, atol=2e-7)
            self.assertEqual(
                choices_from_scores(self.bank, original_scores, blocks["valid"]),
                choices_from_scores(self.bank, restored_scores, blocks["valid"]),
            )
            choices = choices_from_scores(
                self.bank, restored_scores, blocks["valid"]
            )
            self.assertEqual(len(choices), len(self.bank))
            self.assertTrue(all(type(choice) is int for choice in choices))

    def test_submission_entry_uses_frozen_stage3_method(self):
        self.assertIs(submission.predict, submission_stage3.predict)

    def test_missing_checkpoint_is_explicit(self):
        with self.assertRaises(FileNotFoundError):
            submission.predict(self.bank, {"embeddings": self.embeddings}, None)
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(FileNotFoundError):
                load_model(Path(tmp) / "missing")

    def test_arbitrary_length_ids_empty_and_unparseable(self):
        bank = synthetic_bank(7)
        for row in bank:
            row["id"] = "unexpected-id::" + row["id"]
            for candidate in row["candidates"]:
                candidate["text"] = ""
        embeddings = np.zeros((7, 4, 1024), dtype=np.float32)
        blocks = extract_feature_blocks(bank, embeddings)
        self.assertFalse(blocks["valid"].any())
        scores = np.zeros((7, 4), dtype=np.float64)
        choices = choices_from_scores(bank, scores, blocks["valid"])
        self.assertEqual(len(choices), 7)
        self.assertTrue(all(type(choice) is int and 0 <= choice < 4 for choice in choices))


if __name__ == "__main__":
    unittest.main()

