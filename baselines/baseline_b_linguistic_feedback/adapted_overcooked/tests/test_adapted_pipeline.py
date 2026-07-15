from __future__ import annotations

import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.run_baseline_b_pipeline import run_pipeline  # noqa: E402
from src.feature_schema import (  # noqa: E402
    empty_weights,
    load_weights,
    read_json,
    validate_feedback_examples,
)
from src.overcooked_grounding import ground_feedback  # noqa: E402
from src.probe_evaluator import evaluate_probes, load_probe_states  # noqa: E402
from src.session_bridge import build_session_feedback_examples  # noqa: E402


class FeedbackContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.probes = load_probe_states()
        cls.feedback = read_json(ROOT / "data" / "feedback_examples.json")

    def test_all_88_feedback_examples_validate(self) -> None:
        self.assertEqual(len(self.feedback), 88)
        validate_feedback_examples(self.feedback, probe_states=self.probes)

    def test_unknown_feature_is_rejected_with_feedback_id(self) -> None:
        invalid = [copy.deepcopy(self.feedback[0])]
        invalid[0]["trajectory_features"]["not_in_schema"] = 1
        with self.assertRaisesRegex(ValueError, invalid[0]["feedback_id"]):
            validate_feedback_examples(invalid, probe_states=self.probes)

    def test_every_example_grounds_to_nonempty_features(self) -> None:
        action_library = {
            action["action_id"]: action["features"]
            for probe in self.probes
            for action in probe["available_actions"]
        }
        for feedback in self.feedback:
            result = ground_feedback(
                feedback,
                feedback_type=feedback["expected_feedback_type"],
                action_feature_library=action_library,
            )
            self.assertTrue(result["target_features"], feedback["feedback_id"])


class ProbeEvaluationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.probes = load_probe_states()

    def test_action_order_does_not_change_result(self) -> None:
        weights = load_weights()
        original = evaluate_probes(self.probes, weights)
        reversed_probes = copy.deepcopy(self.probes)
        for probe in reversed_probes:
            probe["available_actions"].reverse()
        reversed_result = evaluate_probes(reversed_probes, weights)
        self.assertEqual(
            [item["chosen_action"] for item in original["results"]],
            [item["chosen_action"] for item in reversed_result["results"]],
        )
        self.assertEqual(original["correct"], reversed_result["correct"])

    def test_zero_weight_ties_are_failures(self) -> None:
        evaluation = evaluate_probes(self.probes, empty_weights())
        self.assertEqual(evaluation["correct"], 0)
        self.assertEqual(evaluation["tie_count"], len(self.probes))
        self.assertTrue(all(item["chosen_action"] is None for item in evaluation["results"]))

    def test_leave_one_probe_out_excludes_derived_examples(self) -> None:
        feedback = read_json(ROOT / "data" / "feedback_examples.json")
        result = run_pipeline(
            feedback_examples=feedback,
            probe_states=self.probes,
            initial_weights=load_weights(),
            learning_rate=1.0,
        )
        holdout = result["probe_evaluations"]["leave_one_probe_out"]
        for item in holdout["results"]:
            expected_excluded = sum(
                example.get("probe_id") == item["probe_id"]
                for example in feedback
            )
            self.assertEqual(item["excluded_examples"], expected_excluded)
            self.assertEqual(
                item["training_examples"],
                len(feedback) - expected_excluded,
            )


class SessionBridgeTests(unittest.TestCase):
    @staticmethod
    def _write_jsonl(path: Path, records: list[dict]) -> None:
        path.write_text(
            "".join(json.dumps(record) + "\n" for record in records),
            encoding="utf-8",
        )

    def test_attributed_window_becomes_provenance_rich_example(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            session = Path(directory) / "session_001"
            session.mkdir()
            timestamp = "2026-07-14T10:00:00+00:00"
            self._write_jsonl(
                session / "trajectory.jsonl",
                [
                    {
                        "total_step": 5,
                        "layout": "ring_tomato_onion_10x6",
                        "ai_action_name": "interact",
                        "human_action_name": "east",
                        "environment_reward": 0,
                        "state_facts": {
                            "ai_pos": [2, 1],
                            "human_pos": [1, 1],
                            "ai_held_object": {"name": "tomato"},
                            "pot_states": {},
                        },
                        "extra": {
                            "state_before": {
                                "ai_pos": [2, 1],
                                "human_pos": [1, 1],
                                "ai_held_object": None,
                            }
                        },
                    }
                ],
            )
            self._write_jsonl(
                session / "feedback_events.jsonl",
                [
                    {
                        "source": "chat_messages.csv",
                        "timestamp_utc": timestamp,
                        "episode": 0,
                        "total_step": 5,
                        "feedback_text": "You blocked me.",
                        "role": "human_language",
                        "extra": {"layout": "ring_tomato_onion_10x6"},
                    }
                ],
            )
            self._write_jsonl(
                session / "attribution_preview.jsonl",
                [
                    {
                        "feedback_event_id": f"chat_messages.csv:5:{timestamp}",
                        "feedback_type": "evaluative",
                        "target_time_window": [5, 5],
                        "target_event": "AI_blocked_human_path",
                        "polarity": "negative",
                        "confidence": 0.9,
                        "needs_clarification": False,
                    }
                ],
            )

            examples = build_session_feedback_examples(session)

        self.assertEqual(len(examples), 1)
        self.assertEqual(examples[0]["session_id"], "session_001")
        self.assertEqual(examples[0]["attributed_sentiment_score"], -1.0)
        self.assertEqual(examples[0]["trajectory_features"]["pick_tomato"], 1.0)
        self.assertEqual(examples[0]["trajectory_features"]["blocks_human_path"], 2.0)
        validate_feedback_examples(examples, probe_states=load_probe_states())


if __name__ == "__main__":
    unittest.main()
