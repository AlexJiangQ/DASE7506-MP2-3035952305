import copy
import tempfile
import unittest
from pathlib import Path

import numpy as np

import common
import creo_features
import submission_stage5
from creo_features import extract_group_features
from selector_features import extract_feature_blocks
from stage5_features import (
    SOURCE_NAMES,
    assemble_state,
    build_residual_examples,
    combine_feature_blocks,
    groups_to_candidates,
)
from stage5_model import (
    load_stage5_bundle,
    multinomial_loss_and_gradient,
    predict_logits,
    save_stage5_bundle,
    train_multinomial,
)


def synthetic_bank(n=5):
    templates = [
        "2 + 3 = 5. Final Answer: \\boxed{5}",
        "Careful calculation gives the total. #### 5",
        "2 + 3 = 6. Final Answer: \\boxed{6}",
        "Double the sum: 5 * 2 = 10. #### 10",
    ]
    return [
        {
            "id": f"arbitrary::{i * 19 + 7}",
            "question": f"There are 2 red and 3 blue items. How many total? v{i}",
            "candidates": [
                {"index": j, "text": text + (" detail" * (i % 2))}
                for j, text in enumerate(templates)
            ],
        }
        for i in range(n)
    ]


def synthetic_state(bank, embeddings, candidate_scores):
    blocks = extract_feature_blocks(bank, embeddings)
    groups = extract_group_features(bank, embeddings, blocks, candidate_scores)
    group_scores = np.full((len(bank), 4), -np.inf)
    anchor = np.zeros(len(bank), dtype=np.int64)
    for i, answers in enumerate(groups["answers"]):
        group_scores[i, :len(answers)] = [
            {"5": 0.0, "6": 1.0, "10": 2.0}.get(str(answer), -1.0)
            for answer in answers
        ]
        anchor[i] = answers.index("5") if "5" in answers else 0
    state = assemble_state(
        bank, embeddings, blocks, candidate_scores, groups, group_scores,
        np.full(len(bank), 0.7), np.full(len(bank), 0.6), anchor,
    )
    return state


class Stage5Tests(unittest.TestCase):
    def setUp(self):
        self.bank = synthetic_bank()
        rng = np.random.default_rng(20261022)
        self.embeddings = rng.normal(size=(len(self.bank), 4, 1024)).astype(np.float32)
        base = np.asarray([[0.1, 0.0], [0.2, 0.1], [0.0, 2.0], [-0.1, 1.0]])
        self.candidate_scores = np.repeat(base[None, :, :], len(self.bank), axis=0)

    def test_official_parser_binding(self):
        self.assertIs(creo_features.parse_answer, common.parse_answer)

    def test_proposal_deduplication_and_residual_labels(self):
        state = synthetic_state(self.bank, self.embeddings, self.candidate_scores)
        labels = np.asarray([[1, 1, 0, 0]] * len(self.bank), dtype=bool)
        examples = build_residual_examples(state, labels)
        for i in range(len(self.bank)):
            rows = examples["question_indices"] == i
            proposed = examples["proposal_groups"][rows]
            self.assertEqual(len(proposed), len(set(proposed.tolist())))
            self.assertLessEqual(len(proposed), len(SOURCE_NAMES))
            answers = [state["group_data"]["answers"][i][g] for g in proposed]
            outcome_by_answer = dict(zip(answers, examples["outcomes"][rows].tolist()))
            self.assertEqual(outcome_by_answer["5"], 2)
            self.assertEqual(outcome_by_answer["6"], 1)
            self.assertEqual(outcome_by_answer["10"], 1)

    def test_multinomial_and_harm_aux_gradient(self):
        rng = np.random.default_rng(31)
        x = rng.normal(size=(9, 6))
        x[:, -1] = 1.0
        y = np.asarray([0, 1, 2, 0, 1, 2, 0, 1, 2])
        sample_weights = np.linspace(0.5, 1.5, len(y))
        parameters = rng.normal(scale=0.05, size=(6, 3))
        for harm_weight in (0.0, 0.1):
            loss, gradient = multinomial_loss_and_gradient(
                parameters, x, y, sample_weights, l2=0.03,
                harm_aux_weight=harm_weight,
            )
            self.assertTrue(np.isfinite(loss))
            numerical = np.zeros_like(parameters)
            epsilon = 1e-6
            for row in range(parameters.shape[0]):
                for column in range(parameters.shape[1]):
                    plus = parameters.copy(); plus[row, column] += epsilon
                    minus = parameters.copy(); minus[row, column] -= epsilon
                    numerical[row, column] = (
                        multinomial_loss_and_gradient(
                            plus, x, y, sample_weights, 0.03, harm_weight
                        )[0]
                        - multinomial_loss_and_gradient(
                            minus, x, y, sample_weights, 0.03, harm_weight
                        )[0]
                    ) / (2 * epsilon)
            self.assertLess(float(np.max(np.abs(gradient - numerical))), 4e-6)

    def test_scaler_is_fold_isolated(self):
        features = np.asarray([
            [0.0, 1.0], [1.0, 2.0], [2.0, 3.0],
            [1000.0, -1000.0], [2000.0, -2000.0], [3000.0, -3000.0],
        ])
        labels = np.asarray([0, 1, 2, 0, 1, 2])
        questions = np.arange(6)
        model = train_multinomial(
            features, labels, questions, np.asarray([0, 1, 2]),
            class_weight_mode="uniform", harm_aux_weight=0.0, max_steps=5,
        )
        np.testing.assert_allclose(model["mean"], features[:3].mean(axis=0))
        self.assertFalse(np.allclose(model["mean"], features.mean(axis=0)))

    def test_candidate_and_answer_group_permutation_equivariance(self):
        original_state = synthetic_state(self.bank, self.embeddings, self.candidate_scores)
        original_examples = build_residual_examples(original_state)
        original_features, _ = combine_feature_blocks(original_examples, "full")
        permutation = np.asarray([2, 0, 3, 1])
        permuted_bank = copy.deepcopy(self.bank)
        for row in permuted_bank:
            row["candidates"] = [row["candidates"][int(j)] for j in permutation]
            for j, candidate in enumerate(row["candidates"]):
                candidate["index"] = j
        permuted_state = synthetic_state(
            permuted_bank, self.embeddings[:, permutation], self.candidate_scores[:, permutation]
        )
        permuted_examples = build_residual_examples(permuted_state)
        permuted_features, _ = combine_feature_blocks(permuted_examples, "full")
        for i in range(len(self.bank)):
            original_rows = np.flatnonzero(original_examples["question_indices"] == i)
            permuted_rows = np.flatnonzero(permuted_examples["question_indices"] == i)
            original_by_answer = {
                original_state["group_data"]["answers"][i][int(original_examples["proposal_groups"][r])]:
                original_features[r] for r in original_rows
            }
            permuted_by_answer = {
                permuted_state["group_data"]["answers"][i][int(permuted_examples["proposal_groups"][r])]:
                permuted_features[r] for r in permuted_rows
            }
            self.assertEqual(set(original_by_answer), set(permuted_by_answer))
            for answer in original_by_answer:
                np.testing.assert_allclose(
                    original_by_answer[answer], permuted_by_answer[answer], atol=1e-10
                )

    def test_checkpoint_roundtrip_float_tolerance_and_missing(self):
        rng = np.random.default_rng(9)
        features = rng.normal(size=(12, 4))
        labels = np.asarray([0, 1, 2] * 4)
        questions = np.arange(12)
        model = train_multinomial(
            features, labels, questions, questions, "balanced-capped", 0.1,
            max_steps=20,
        )
        with tempfile.TemporaryDirectory() as temporary:
            checkpoint = Path(temporary) / "bundle"
            save_stage5_bundle(checkpoint, model, 1.2, {
                "feature_set": "consensus_score", "proposal_sources": ["plurality"],
                "harm_cost": 2.0, "threshold": 0.05, "apply_residual": True,
                "anchor_checkpoint": "anchor_stage3", "seed": 20261022,
            })
            restored = load_stage5_bundle(checkpoint)
            np.testing.assert_allclose(
                predict_logits(model, features),
                predict_logits(restored["model"], features), atol=2e-6,
            )
            with self.assertRaises(FileNotFoundError):
                load_stage5_bundle(Path(temporary) / "missing")
        with self.assertRaises(FileNotFoundError):
            submission_stage5.predict(
                self.bank, {"embeddings": self.embeddings}, Path("definitely-missing-stage5")
            )

    def test_provisional_bundle_determinism_and_python_ints(self):
        checkpoint = Path("runs/stage5/bugfix2/provisional_bundle")
        if not checkpoint.is_dir():
            self.skipTest("Stage-5 provisional bundle has not been trained")
        first = submission_stage5.predict(
            self.bank, {"embeddings": self.embeddings}, checkpoint
        )
        second = submission_stage5.predict(
            self.bank, {"embeddings": self.embeddings}, checkpoint
        )
        self.assertEqual(first, second)
        self.assertTrue(all(type(choice) is int and 0 <= choice < 4 for choice in first))

    def test_arbitrary_ids_lengths_empty_and_all_unparseable(self):
        bank = synthetic_bank(7)
        for row in bank:
            row["id"] = "unexpected/" + row["id"]
            for candidate in row["candidates"]:
                candidate["text"] = ""
        embeddings = np.zeros((7, 4, 1024), dtype=np.float32)
        scores = np.zeros((7, 4, 2), dtype=np.float64)
        blocks = extract_feature_blocks(bank, embeddings)
        groups = extract_group_features(bank, embeddings, blocks, scores)
        group_scores = np.zeros((7, 4), dtype=np.float64)
        state = assemble_state(
            bank, embeddings, blocks, scores, groups, group_scores,
            np.full(7, 0.5), np.full(7, 0.5), np.zeros(7, dtype=int),
        )
        examples = build_residual_examples(state)
        features, _ = combine_feature_blocks(examples, "full")
        self.assertTrue(np.isfinite(features).all())
        choices = groups_to_candidates(state, np.zeros(7, dtype=int))
        self.assertEqual(len(choices), 7)
        self.assertTrue(all(type(choice) is int and 0 <= choice < 4 for choice in choices))


if __name__ == "__main__":
    unittest.main()
