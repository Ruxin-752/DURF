from __future__ import annotations

import json
import shutil
import sys
import unittest
import uuid
from contextlib import contextmanager
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.evaluate_route2_online_adaptation_simulation import _summarize  # noqa: E402
from scripts.evaluate_route2_trajectory_required_benchmark import (  # noqa: E402
    _metrics,
    _validity_gate,
    run_evaluation,
)
from scripts.generate_route2_trajectory_required_benchmark import (  # noqa: E402
    benchmark_contexts,
    context_reward_basis,
    generate_benchmark,
)
from src.feature_schema import load_features  # noqa: E402
from src.subgoal_featurizer import featurize_subgoal  # noqa: E402


TEST_TMP = ROOT / "outputs" / "_trajectory_required_test_tmp"


@contextmanager
def workspace_temp():
    TEST_TMP.mkdir(parents=True, exist_ok=True)
    path = TEST_TMP / uuid.uuid4().hex
    path.mkdir()
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)


class TrajectoryRequiredBenchmarkTests(unittest.TestCase):
    def test_reward_basis_can_prefer_reference_with_subset_features(self) -> None:
        spec = next(
            item
            for item in benchmark_contexts()
            if item["context_id"] == "yield_human_tomato"
        )
        reference = featurize_subgoal(spec["context"], spec["referenced_subgoal"])
        alternatives = [
            featurize_subgoal(spec["context"], subgoal)
            for subgoal in spec["feasible_subgoals"]
            if subgoal != spec["referenced_subgoal"]
        ]
        self.assertFalse(
            any(
                value > 0.0
                and all(other.get(feature, 0.0) == 0.0 for other in alternatives)
                for feature, value in reference.items()
            )
        )

        weights = context_reward_basis(spec, sorted(load_features()))

        def score(vector: dict[str, float]) -> float:
            return sum(weights.get(feature, 0.0) * value for feature, value in vector.items())

        self.assertTrue(any(value < 0.0 for value in weights.values()))
        self.assertTrue(all(score(reference) > score(other) for other in alternatives))

    def test_generator_crosses_both_inputs_and_seals_test(self) -> None:
        with workspace_temp() as output:
            manifest = generate_benchmark(
                output,
                training_config={
                    "seeds": [1],
                    "epochs": 2,
                    "learning_rate": 0.01,
                    "weight_decay": 0.0001,
                    "batch_size": 64,
                    "patience": 1,
                    "optimizer": "adam",
                    "dimension_weighting": "varying_only",
                },
            )
            self.assertEqual(manifest["split_counts"], {"train": 160, "dev": 60, "test": 60})
            self.assertEqual(manifest["context_count"], 10)
            self.assertEqual(manifest["reward_config_count"], 20)
            self.assertEqual(manifest["audit"]["exact_input_target_conflicts"], 0)
            self.assertEqual(
                manifest["audit"]["same_text_multi_trajectory_target_groups"],
                28,
            )
            self.assertEqual(
                manifest["audit"]["same_trajectory_opposite_polarity_groups"],
                30,
            )
            self.assertTrue(
                all(
                    value == 0
                    for value in manifest["audit"]["normalized_text_split_overlap"].values()
                )
            )
            train_dev = json.loads((output / "train_dev.json").read_text(encoding="utf-8"))
            test = json.loads((output / "test.json").read_text(encoding="utf-8"))
            self.assertEqual({row["split"] for row in train_dev["examples"]}, {"train", "dev"})
            self.assertEqual({row["split"] for row in test["examples"]}, {"test"})
            with self.assertRaises(FileExistsError):
                generate_benchmark(output)

    def test_metrics_and_gate_require_full_to_beat_both_ablations(self) -> None:
        examples = [
            {"target_reward": [1.0, 0.0]},
            {"target_reward": [-1.0, 0.0]},
        ]
        catalog = np.asarray([[1.0, 0.0], [-1.0, 0.0]])
        perfect = _metrics(
            np.asarray([[1.0, 0.0], [-1.0, 0.0]]),
            examples,
            varying_indices=[0],
            target_catalog=catalog,
        )
        weak = _metrics(
            np.zeros((2, 2)),
            examples,
            varying_indices=[0],
            target_catalog=catalog,
        )

        def aggregate(metrics):
            return {
                name: {"mean": float(value), "std": 0.0, "values": [float(value)]}
                for name, value in metrics.items()
                if isinstance(value, float)
            }

        manifest = {
            "validity_gate": {
                "minimum_full_relative_gain_over_text_only": 0.30,
                "minimum_full_relative_gain_over_trajectory_only": 0.30,
                "minimum_full_nearest_config_accuracy": 0.85,
                "minimum_full_active_sign_accuracy": 0.90,
                "minimum_config_accuracy_gain_over_each_ablation": 0.25,
            }
        }
        gate = _validity_gate(
            {
                "full": aggregate(perfect),
                "text_only": aggregate(weak),
                "trajectory_only": aggregate(weak),
            },
            manifest,
        )
        self.assertEqual(gate["status"], "passed")
        failed = _validity_gate(
            {"full": aggregate(weak), "text_only": aggregate(weak), "trajectory_only": aggregate(weak)},
            manifest,
        )
        self.assertEqual(failed["status"], "failed")

    def test_dev_evaluator_does_not_open_sealed_test_payload(self) -> None:
        with workspace_temp() as output:
            manifest = generate_benchmark(
                output,
                training_config={
                    "seeds": [1],
                    "epochs": 2,
                    "learning_rate": 0.01,
                    "weight_decay": 0.0001,
                    "batch_size": 64,
                    "patience": 1,
                    "optimizer": "adam",
                    "dimension_weighting": "varying_only",
                },
            )
            (output / "test.json").unlink()
            evaluation_dir = output / "dev_evaluation"
            report = run_evaluation(
                manifest["manifest_path"],
                partition="dev",
                output_dir=evaluation_dir,
            )
            self.assertEqual(report["partition"], "dev")
            self.assertIsNone(report["test_sha256"])
            self.assertFalse(report["formal_seed137_test_touched"])
            self.assertFalse((evaluation_dir / "test_evaluation_receipt.json").exists())

    def test_online_summary_reports_utility_regret_task_and_safety(self) -> None:
        summary = _summarize(
            [
                {
                    "chosen_subgoal": "WAIT",
                    "teacher_utility": 1.0,
                    "regret": 0.0,
                    "task_proxy": 0.0,
                    "safety_violation": False,
                },
                {
                    "chosen_subgoal": "GET_ONION",
                    "teacher_utility": -1.0,
                    "regret": 2.0,
                    "task_proxy": 1.0,
                    "safety_violation": True,
                },
            ]
        )
        self.assertEqual(summary["cumulative_teacher_utility"], 0.0)
        self.assertEqual(summary["cumulative_regret"], 2.0)
        self.assertEqual(summary["preference_satisfaction_rate"], 0.5)
        self.assertEqual(summary["task_proxy_total"], 1.0)
        self.assertEqual(summary["safety_violation_count"], 1)


if __name__ == "__main__":
    unittest.main()
