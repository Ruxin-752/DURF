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
            hu_tolerance=1.0,
            apply_hu=True,
        )

        self.assertIsNotNone(result)
        self.assertEqual(
            result.decision["selected"],
            CONTINUE_CURRENT_SUBGOAL,
        )
        self.assertEqual(result.action, 0)

    def test_tolerance_zero_keeps_the_prior_and_records_the_shadow_score(self):
        controller = CoordinationController()
        candidates = build_coordination_candidates(
            proposed_action=0, yield_action=1, stay_action=4,
            conflict_type="human_entering_ai_tile",
        )
        result = controller.resolve(
            timestep=10, episode=1, task_subgoal="GET_TOMATO",
            conflict_type="human_entering_ai_tile", ai_pos=(1, 4), human_pos=(2, 4),
            candidates=candidates, condition_features={},
            hu_score=lambda option: 2.0 if option == CONTINUE_CURRENT_SUBGOAL else -2.0,
            hu_tolerance=0.0, apply_hu=True,
        )
        self.assertEqual(result.decision["selected"], YIELD)
        self.assertFalse(result.decision["preference_override"])
        self.assertEqual(result.decision["tolerance_unit"], "prior_points")
        # The prior is the score; Hu is recorded, never added in.
        by_option = {c["option"]: c for c in result.decision["candidates"]}
        self.assertEqual(by_option[YIELD]["final_score"], by_option[YIELD]["base_score"])
        self.assertEqual(by_option[CONTINUE_CURRENT_SUBGOAL]["hu_score"], 2.0)

    def test_an_option_hu_has_no_evidence_about_cannot_override_the_prior(self):
        controller = CoordinationController()
        candidates = build_coordination_candidates(
            proposed_action=0, yield_action=1, stay_action=4,
            conflict_type="human_entering_ai_tile",
        )
        result = controller.resolve(
            timestep=10, episode=1, task_subgoal="GET_TOMATO",
            conflict_type="human_entering_ai_tile", ai_pos=(1, 4), human_pos=(2, 4),
            candidates=candidates, condition_features={},
            hu_score=lambda option: 2.0 if option == CONTINUE_CURRENT_SUBGOAL else -2.0,
            hu_supported=lambda option: option != CONTINUE_CURRENT_SUBGOAL,
            hu_tolerance=1.0, apply_hu=True,
        )
        self.assertEqual(result.decision["selected"], YIELD)
        self.assertFalse(result.decision["preference_override"])

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

    def test_insisting_is_bounded_exactly_like_yielding(self):
        """A learned preference for CONTINUE in a head-on that never clears
        must not become a permanent deadlock: after its horizon CONTINUE goes
        on cooldown and YIELD gets its turn, then the roles swap back."""
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
                hu_score=lambda option: 1.0 if option == CONTINUE_CURRENT_SUBGOAL else -1.0,
                hu_tolerance=1.0,
                apply_hu=True,
            )

        picks = [resolve(t).decision["selected"] for t in range(10, 18)]
        self.assertEqual(picks[:2], [CONTINUE_CURRENT_SUBGOAL] * 2)   # its horizon
        self.assertEqual(picks[2:4], [YIELD] * 2)                     # continue on cooldown
        self.assertEqual(picks[4:6], [CONTINUE_CURRENT_SUBGOAL] * 2)  # yield on cooldown
        self.assertIn(YIELD, picks[6:])                               # and so on

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
