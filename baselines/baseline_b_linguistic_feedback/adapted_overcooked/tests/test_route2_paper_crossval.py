from __future__ import annotations

import sys
import unittest
from unittest import mock
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.neural_inference import (  # noqa: E402
    ENSEMBLE_SCHEMA_VERSION,
    TrajectoryFeedbackRewardPredictor,
    load_predictor,
    predict_reward_distribution,
    predictor_sha256,
)
from src.paper_cross_validation import make_paper_cross_validation_manifest  # noqa: E402
from scripts.prepare_route2_dataset import audit_full_reward_supervision  # noqa: E402
from scripts.evaluate_route2_paper_crossval import (  # noqa: E402
    _fold_behavior,
    _verify_manifest_hash,
)
from scripts.select_route2_candidate_on_dev import (  # noqa: E402
    _choose_fold_winner,
    _selection_view,
)


class PaperCrossValidationTests(unittest.TestCase):
    def test_ten_fold_rotation_holds_out_both_axes(self) -> None:
        examples = [
            {
                "teacher_id": f"teacher_{teacher:02d}",
                "reward_config_id": f"reward_{reward:02d}",
            }
            for teacher in range(10)
            for reward in range(20)
        ]
        manifest = make_paper_cross_validation_manifest(examples, n_folds=10, seed=7)
        self.assertEqual(len(manifest["folds"]), 10)
        self.assertEqual(manifest["teacher_identity_field"], "teacher_id")
        self.assertEqual(manifest["coverage"]["corpus_examples"], len(examples))
        for fold in manifest["folds"]:
            self.assertEqual(fold["test_axis_fold"], (fold["fold"] + 1) % 10)
            self.assertFalse(set(fold["train_teachers"]) & set(fold["dev_teachers"]))
            self.assertFalse(set(fold["train_teachers"]) & set(fold["test_teachers"]))
            self.assertFalse(set(fold["dev_teachers"]) & set(fold["test_teachers"]))
            self.assertFalse(set(fold["train_rewards"]) & set(fold["dev_rewards"]))
            self.assertFalse(set(fold["train_rewards"]) & set(fold["test_rewards"]))
            self.assertFalse(set(fold["dev_rewards"]) & set(fold["test_rewards"]))
            self.assertTrue(fold["train_indices"])
            self.assertTrue(fold["dev_indices"])
            self.assertTrue(fold["test_indices"])
            self.assertEqual(
                fold["coverage"]["corpus_examples"],
                len(examples),
            )
            self.assertIn("excluded_fraction", fold["coverage"])
            self.assertIn("heldout_overlap_audit", fold)

    def test_sparse_components_still_produce_nonempty_folds(self) -> None:
        examples = []
        for component, teacher_count, reward_count in (("a", 8, 24), ("b", 2, 6), ("c", 2, 6)):
            for teacher in range(teacher_count):
                for reward in range(reward_count):
                    examples.append(
                        {
                            "teacher_id": f"{component}_teacher_{teacher}",
                            "reward_config_id": f"{component}_reward_{reward}",
                        }
                    )
        manifest = make_paper_cross_validation_manifest(examples, n_folds=10, seed=42)
        for fold in manifest["folds"]:
            self.assertGreater(len(fold["dev_indices"]), 0)
            self.assertGreater(len(fold["test_indices"]), 0)

    def test_row_hash_proxy_is_not_accepted_as_teacher_identity(self) -> None:
        examples = [
            {
                "cv_teacher_id": f"row_hash_{index}",
                "reward_config_id": f"reward_{index}",
            }
            for index in range(10)
        ]
        with self.assertRaisesRegex(ValueError, "stable teacher_id"):
            make_paper_cross_validation_manifest(examples, n_folds=10)


class FullRewardSupervisionAuditTests(unittest.TestCase):
    def test_identical_input_with_different_complete_reward_fails_closed(self) -> None:
        rows = [
            {
                "feedback_id": "first",
                "text": "Please keep the lane clear.",
                "route2_trajectory_features": {"blocks_human_path": 1.0},
                "teacher_reward_weights": {"blocks_human_path": -1.0},
            },
            {
                "feedback_id": "second",
                "text": "  please KEEP the lane clear. ",
                "route2_trajectory_features": {"blocks_human_path": 1.0},
                "teacher_reward_weights": {"blocks_human_path": -2.0},
            },
        ]
        with self.assertRaisesRegex(ValueError, "conflicting complete Route 2 supervision"):
            audit_full_reward_supervision(rows, ["blocks_human_path"])

    def test_identical_input_with_same_complete_reward_is_audited(self) -> None:
        row = {
            "feedback_id": "same",
            "text": "Keep the lane clear.",
            "route2_trajectory_features": {"blocks_human_path": 1.0},
            "teacher_reward_weights": {"blocks_human_path": -1.0},
        }
        audit = audit_full_reward_supervision([row, dict(row)], ["blocks_human_path"])
        self.assertEqual(audit["consistent_duplicate_inputs"], 1)
        self.assertEqual(audit["supervision_conflict_count"], 0)


class DevOnlySelectionTests(unittest.TestCase):
    def test_selection_view_physically_omits_test_examples(self) -> None:
        dataset = {
            "examples": [
                {"feedback_id": name, "tokens": [name]}
                for name in ("train", "dev", "test")
            ]
        }
        selected = _selection_view(
            dataset,
            {"train_indices": [0], "dev_indices": [1], "test_indices": [2]},
            min_freq=1,
        )
        self.assertEqual(
            [example["feedback_id"] for example in selected["examples"]],
            ["train", "dev"],
        )
        self.assertEqual(selected["split"]["test_indices"], [])

    def test_fold_winner_uses_mse_shortlist_then_tie_break(self) -> None:
        reports = [
            {
                "candidate_id": "sgd_paper_accumulated_mse",
                "metrics": {
                    "varying_dimension_mse": 1.005,
                    "nearest_reward_config_accuracy": 0.8,
                    "preference_sensitive_cooking_accuracy": 0.5,
                },
            },
            {
                "candidate_id": "adam_fixed_budget_mse",
                "metrics": {
                    "varying_dimension_mse": 1.0,
                    "nearest_reward_config_accuracy": 0.7,
                    "preference_sensitive_cooking_accuracy": 0.9,
                },
            },
        ]
        winner, rule = _choose_fold_winner(reports)
        self.assertEqual(winner["candidate_id"], "sgd_paper_accumulated_mse")
        self.assertIn("sgd_paper_accumulated_mse", rule["shortlisted_candidates"])


class FrozenEvaluationGuardTests(unittest.TestCase):
    def test_manifest_self_hash_is_recomputed(self) -> None:
        from src.evaluation_splits import canonical_sha256

        manifest = {"folds": [{"fold": 0}]}
        manifest["manifest_sha256"] = canonical_sha256(manifest)
        self.assertEqual(
            _verify_manifest_hash(manifest, label="test"),
            manifest["manifest_sha256"],
        )
        manifest["folds"][0]["fold"] = 1
        with self.assertRaisesRegex(ValueError, "manifest hash mismatch"):
            _verify_manifest_hash(manifest, label="test")

    def test_reward_tie_cannot_become_correct_majority_vote(self) -> None:
        example = {
            "reward_config_id": "reward",
            "group_id": "context",
            "context": {},
            "feasible_subgoals": ["GET_ONION", "WAIT"],
            "acceptable_subgoals": ["GET_ONION"],
        }
        tied_decision = {
            "is_tie": True,
            "chosen_subgoal": "GET_ONION",
        }
        with mock.patch(
            "scripts.evaluate_route2_paper_crossval.choose_subgoal",
            return_value=tied_decision,
        ):
            metrics = _fold_behavior([example], torch.zeros((1, 1)), ["feature"])
        self.assertEqual(metrics["per_utterance"]["correct"], 0)
        self.assertEqual(metrics["per_reward_context_majority"]["correct"], 0)
        self.assertEqual(metrics["unresolved_reward_tie_votes"], 1)


class EnsembleInferenceTests(unittest.TestCase):
    @staticmethod
    def _constant_model(values: list[float]) -> TrajectoryFeedbackRewardPredictor:
        model = TrajectoryFeedbackRewardPredictor(vocab_size=2, n_features=len(values))
        with torch.no_grad():
            for parameter in model.parameters():
                parameter.zero_()
            model.fc2.bias.copy_(torch.tensor(values, dtype=torch.float32))
        return model

    def test_manifest_loads_members_and_averages_reward_vectors(self) -> None:
        manifest_path = ROOT / "tests" / "fixtures" / "route2_ensemble_manifest.json"
        features = ["a", "b"]
        vocab = {"<pad>": 0, "<unk>": 1}

        def fake_load(path):
            values = [1.0, 3.0] if "member_0" in str(path) else [3.0, 7.0]
            return self._constant_model(values), vocab, features, True

        with mock.patch("src.neural_inference.load_checkpoint", side_effect=fake_load):
            predictor = load_predictor(manifest_path)
            result = predict_reward_distribution(
                predictor, "anything", feature_counts=[0.0, 0.0]
            )
            self.assertEqual(result["ensemble_size"], 2)
            self.assertEqual(result["predictor_kind"], "ensemble")
            self.assertAlmostEqual(result["weights"]["a"], 2.0)
            self.assertAlmostEqual(result["weights"]["b"], 5.0)
            self.assertAlmostEqual(result["uncertainty"]["a"], 1.0)
            self.assertAlmostEqual(result["uncertainty"]["b"], 2.0)
            self.assertEqual(len(predictor_sha256(manifest_path)), 64)


if __name__ == "__main__":
    unittest.main()
