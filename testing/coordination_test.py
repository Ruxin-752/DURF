"""Tests for explicit task-independent coordination decisions."""

from __future__ import annotations

import unittest

from durf.baseline.coordination import (
    CONTINUE_CURRENT_SUBGOAL,
    YIELD,
    CoordinationController,
    build_coordination_candidates,
    path_conflict_type,
)


class CoordinationTests(unittest.TestCase):
    def test_detects_human_entering_ai_tile(self):
        conflict = path_conflict_type(
            ai_pos=(1, 4),
            human_pos=(2, 4),
            ai_target=(1, 3),
            human_target=(1, 4),
            ai_is_moving=True,
            human_is_moving=True,
        )

        self.assertEqual(conflict, "human_entering_ai_tile")

    def test_default_prior_reproduces_yield_behavior(self):
        controller = CoordinationController()
        candidates = build_coordination_candidates(
            proposed_action=0,
            yield_action=1,
            stay_action=4,
            conflict_type="human_entering_ai_tile",
        )

        result = controller.resolve(
            timestep=10,
            episode=1,
            task_subgoal="GET_TOMATO",
            conflict_type="human_entering_ai_tile",
            ai_pos=(1, 4),
            human_pos=(2, 4),
            candidates=candidates,
            condition_features={},
        )

        self.assertIsNotNone(result)
        self.assertEqual(result.decision["selected"], YIELD)
        self.assertEqual(result.action, 1)

    def test_hu_can_reverse_default_yield_prior(self):
        controller = CoordinationController()
        candidates = build_coordination_candidates(
            proposed_action=0,
            yield_action=1,
            stay_action=4,
            conflict_type="human_entering_ai_tile",
        )

        result = controller.resolve(
            timestep=10,
            episode=1,
            task_subgoal="GET_TOMATO",
            conflict_type="human_entering_ai_tile",
            ai_pos=(1, 4),
            human_pos=(2, 4),
            candidates=candidates,
            condition_features={},
            hu_score=lambda option: (
                2.0 if option == CONTINUE_CURRENT_SUBGOAL else -2.0
            ),
            hu_lambda=1.0,
            apply_hu=True,
        )

        self.assertIsNotNone(result)
        self.assertEqual(
            result.decision["selected"],
            CONTINUE_CURRENT_SUBGOAL,
        )
        self.assertEqual(result.action, 0)

    def test_yield_expires_and_enters_cooldown(self):
        controller = CoordinationController(
            min_commit_steps=1,
            max_option_steps=2,
            yield_cooldown_steps=2,
        )

        def resolve(timestep):
            return controller.resolve(
                timestep=timestep,
                episode=1,
                task_subgoal="GET_TOMATO",
                conflict_type="human_entering_ai_tile",
                ai_pos=(1, 4),
                human_pos=(2, 4),
                candidates=build_coordination_candidates(
                    proposed_action=0,
                    yield_action=1,
                    stay_action=4,
                    conflict_type="human_entering_ai_tile",
                ),
                condition_features={},
            )

        self.assertEqual(resolve(10).decision["selected"], YIELD)
        self.assertEqual(resolve(11).decision["selected"], YIELD)
        self.assertEqual(
            resolve(12).decision["selected"],
            CONTINUE_CURRENT_SUBGOAL,
        )

    def test_no_conflict_clears_active_option(self):
        controller = CoordinationController()
        controller.resolve(
            timestep=10,
            episode=1,
            task_subgoal="GET_TOMATO",
            conflict_type="human_entering_ai_tile",
            ai_pos=(1, 4),
            human_pos=(2, 4),
            candidates=build_coordination_candidates(
                proposed_action=0,
                yield_action=1,
                stay_action=4,
                conflict_type="human_entering_ai_tile",
            ),
            condition_features={},
        )

        result = controller.resolve(
            timestep=11,
            episode=1,
            task_subgoal="GET_TOMATO",
            conflict_type=None,
            ai_pos=(1, 3),
            human_pos=(1, 4),
            candidates=[],
            condition_features={},
        )

        self.assertIsNone(result)
        self.assertIsNone(controller.active)


if __name__ == "__main__":
    unittest.main()
