import copy
import unittest

import numpy as np

import common
import creo_features
from creo_features import arithmetic_evidence, extract_group_features
from creo_model import (
    fit_logistic,
    logistic_probabilities,
    score_group_model,
    train_group_model,
)
from selector_features import extract_feature_blocks


def synthetic_bank(n=6):
    templates = [
        "We calculate 2 + 3 = 5. Final Answer: \\boxed{5}",
        "Using the given 2 and 3, the answer is 5. #### 5",
        "A mistaken step says 2 + 3 = 6. Final Answer: \\boxed{6}",
        "Check 2 + 3 = 5, then 5 * 2 = 10. Answer: 10",
    ]
    return [
        {
            "id": f"nonstandard::{i * 37 + 11}",
            "question": f"If a box has 2 red and 3 blue balls, how many balls are there? v{i}",
            "candidates": [
                {"index": j, "text": text + (" detail" * (i % 3))}
                for j, text in enumerate(templates)
            ],
        }
        for i in range(n)
    ]


class CreoTests(unittest.TestCase):
    def setUp(self):
        self.bank = synthetic_bank()
        rng = np.random.default_rng(20261022)
        self.embeddings = rng.normal(size=(len(self.bank), 4, 1024)).astype(np.float32)
        self.scores = rng.normal(size=(len(self.bank), 4, 2))

    def test_official_parser_binding(self):
        self.assertIs(creo_features.parse_answer, common.parse_answer)

    def test_safe_arithmetic_is_finite_and_conservative(self):
        good = arithmetic_evidence("First 2 + 3 = 5. Answer: 5", "5")
        bad = arithmetic_evidence("First 2 + 3 = 9. Answer: 9", "9")
        hostile = arithmetic_evidence("__import__('os').system('echo no') = 0", "0")
        self.assertTrue(np.isfinite(good).all())
        self.assertTrue(np.isfinite(bad).all())
        self.assertTrue(np.isfinite(hostile).all())
        self.assertEqual(good[0], bad[0])
        self.assertGreater(bad[1], good[1])
        self.assertGreater(bad[2], good[2])

    def test_group_features_are_finite_and_permutation_equivariant(self):
        blocks = extract_feature_blocks(self.bank, self.embeddings)
        original = extract_group_features(self.bank, self.embeddings, blocks, self.scores)
        self.assertEqual(original["features"].shape[:2], (len(self.bank), 4))
        self.assertTrue(np.isfinite(original["features"]).all())

        permutation = np.asarray([2, 0, 3, 1])
        permuted_bank = copy.deepcopy(self.bank)
        for row in permuted_bank:
            row["candidates"] = [row["candidates"][int(j)] for j in permutation]
            for j, candidate in enumerate(row["candidates"]):
                candidate["index"] = j
        permuted_embeddings = self.embeddings[:, permutation]
        permuted_scores = self.scores[:, permutation]
        permuted_blocks = extract_feature_blocks(permuted_bank, permuted_embeddings)
        permuted = extract_group_features(
            permuted_bank, permuted_embeddings, permuted_blocks, permuted_scores
        )
        for i in range(len(self.bank)):
            original_by_answer = {
                answer: original["features"][i, g]
                for g, answer in enumerate(original["answers"][i])
            }
            permuted_by_answer = {
                answer: permuted["features"][i, g]
                for g, answer in enumerate(permuted["answers"][i])
            }
            self.assertEqual(set(original_by_answer), set(permuted_by_answer))
            for answer in original_by_answer:
                np.testing.assert_allclose(
                    original_by_answer[answer], permuted_by_answer[answer], atol=1e-10
                )

    def test_group_and_logistic_training_are_deterministic(self):
        blocks = extract_feature_blocks(self.bank, self.embeddings)
        groups = extract_group_features(self.bank, self.embeddings, blocks, self.scores)
        labels = np.zeros((len(self.bank), 4), dtype=bool)
        for i, answers in enumerate(groups["answers"]):
            labels[i, answers.index("5")] = True
        indices = np.arange(len(self.bank))
        kwargs = dict(
            features=groups["features"], labels=labels, valid=groups["valid"],
            train_indices=indices, scalar_dim=groups["scalar"].shape[2],
            scalar_l2=0.1, embedding_l2=1.0, pairwise_weight=0.1, max_steps=12,
        )
        first = train_group_model(**kwargs)
        second = train_group_model(**kwargs)
        np.testing.assert_array_equal(first["weights"], second["weights"])
        self.assertTrue(np.isfinite(score_group_model(first, groups["features"], groups["valid"])[groups["valid"]]).all())

        x = np.column_stack([np.linspace(-2, 2, 8), np.arange(8) % 3])
        y = np.asarray([0, 0, 0, 1, 0, 1, 1, 1], dtype=float)
        model = fit_logistic(x, y, np.ones(8, dtype=bool), l2=0.1, max_steps=30)
        probabilities = logistic_probabilities(model, x)
        self.assertTrue(np.isfinite(probabilities).all())
        self.assertTrue(((probabilities >= 0) & (probabilities <= 1)).all())

    def test_arbitrary_ids_and_all_unparseable_are_safe(self):
        bank = synthetic_bank(7)
        for row in bank:
            row["id"] = "arbitrary-bank-id/" + row["id"]
            for candidate in row["candidates"]:
                candidate["text"] = ""
        embeddings = np.zeros((7, 4, 1024), dtype=np.float32)
        scores = np.zeros((7, 4, 2), dtype=np.float64)
        blocks = extract_feature_blocks(bank, embeddings)
        groups = extract_group_features(bank, embeddings, blocks, scores)
        self.assertEqual(groups["valid"].sum(axis=1).tolist(), [4] * 7)
        self.assertTrue(np.isfinite(groups["features"]).all())


if __name__ == "__main__":
    unittest.main()
