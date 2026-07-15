from __future__ import annotations

import unittest
from unittest.mock import patch

from durf.baseline.h0_planner import SUBGOALS
from durf.group_a.play_with_baseline import h0_executor_action


class FakeModel:
    def predict(self, inputs, verbose=0):
        raise AssertionError("Exact-origin H0 must not call Keras predict")


class FakeBaseEnv:
    state = object()


class H0ExecutorTests(unittest.TestCase):
    def test_all_subgoals_use_rule_teacher_planner_action(self) -> None:
        model = FakeModel()
        planner = object()
        for index, expected_subgoal in enumerate(SUBGOALS):
            expected_action = index % 6
            with self.subTest(subgoal=expected_subgoal), patch(
                "durf.group_a.play_with_baseline.rule_teacher_decision",
                return_value=(expected_subgoal, expected_action),
            ):
                action, subgoal = h0_executor_action(
                    FakeBaseEnv(),
                    model,
                    planner,
                )
            self.assertEqual(action, expected_action)
            self.assertEqual(subgoal, expected_subgoal)

    def test_bundled_model_must_still_be_loaded(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "model was not loaded"):
            h0_executor_action(FakeBaseEnv(), None, object())


if __name__ == "__main__":
    unittest.main()
