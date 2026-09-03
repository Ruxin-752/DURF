"""Guard rails for the YIELD split: WAIT / BACK_OFF / REROUTE.

YIELD is still a single Hu-scored coordination option (see coordination.py's
COORDINATION_OPTIONS docstring) -- these tests are about the rule-based
execution layer underneath it, which picks how to yield without Hu ever
seeing or choosing between WAIT/BACK_OFF/REROUTE directly.

Two invariants this project relies on:

1. Hu's coordination vocabulary never grows. WAIT, BACK_OFF and REROUTE are
   folded into the YIELD candidate's `reason` string (via `yield_mode`) purely
   for audit -- they must never become a third coordination option.
2. REROUTE must be provably real, not vestigial. It is only worth the code if
   there exists a scenario where blocking the human's current tile and
   destination genuinely changes the AI's planned first step to something
   different -- otherwise BACK_OFF/WAIT already cover every case.
"""

from __future__ import annotations

import unittest

from durf.baseline.coordination import (
    CONTINUE_CURRENT_SUBGOAL,
    YIELD,
    build_coordination_candidates,
    choose_reroute_action,
)


class YieldModeReasonTest(unittest.TestCase):
    """build_coordination_candidates folds yield_mode into `reason` only."""

    def test_no_yield_mode_keeps_the_original_reason(self) -> None:
        candidates = build_coordination_candidates(
            proposed_action=1,
            yield_action=2,
            stay_action=4,
            conflict_type="human_entering_ai_tile",
        )
        yield_candidate = next(c for c in candidates if c.option == YIELD)
        self.assertEqual(yield_candidate.reason, "yield_during_human_entering_ai_tile")

    def test_yield_mode_is_appended_not_substituted(self) -> None:
        for mode in ("reroute", "back_off", "wait"):
            candidates = build_coordination_candidates(
                proposed_action=1,
                yield_action=2,
                stay_action=4,
                conflict_type="ai_blocking_human_route",
                yield_mode=mode,
            )
            yield_candidate = next(c for c in candidates if c.option == YIELD)
            self.assertEqual(
                yield_candidate.reason,
                f"yield_during_ai_blocking_human_route_via_{mode}",
            )

    def test_coordination_option_vocabulary_is_unaffected(self) -> None:
        # However yield_mode is threaded through, Hu must still only ever
        # see two options.
        candidates = build_coordination_candidates(
            proposed_action=1,
            yield_action=2,
            stay_action=4,
            conflict_type="contested_destination",
            yield_mode="reroute",
        )
        self.assertEqual(
            {c.option for c in candidates},
            {CONTINUE_CURRENT_SUBGOAL, YIELD},
        )


class RerouteActionTest(unittest.TestCase):
    """choose_reroute_action against the real ring_tomato_onion_10x6 layout.

    ring_tomato_onion_10x6_h0_full_task is a genuine loop: two 1-tile-wide
    horizontal corridors (rows 1 and 4) connected by two 1-tile-wide vertical
    connectors (columns 1 and 8). That makes it possible to construct a real
    "the shortest route uses one connector; block it and the planner detours
    through the other one" case without inventing a synthetic layout.
    """

    @classmethod
    def setUpClass(cls) -> None:
        from durf.baseline.collect_rule_teacher_dataset import make_motion_planner

        cls.motion_planner = make_motion_planner(
            "ring_tomato_onion_10x6_h0_full_task", seed=0, horizon=400
        )

    def _player(self, position):
        from overcooked_ai_py.mdp.overcooked_mdp import PlayerState

        return PlayerState(position, (0, -1))

    def test_reroute_detours_through_the_other_connector(self) -> None:
        # From (3, 1), the shortest route to (4, 4) goes left through the
        # column-1 connector (action index 3, "west"). Blocking that
        # connector's two tiles (the human standing in it, about to finish
        # crossing it) must not just fail -- it must find the real detour
        # through the column-8 connector, whose first step is the opposite
        # direction, "east" (action index 2).
        player = self._player((3, 1))
        proposed_action = 3  # the route already in conflict: west, via col 1
        reroute = choose_reroute_action(
            self.motion_planner,
            player,
            target_positions=[(4, 4)],
            human_pos=(1, 2),
            human_target=(1, 3),
            proposed_action=proposed_action,
        )
        self.assertIsNotNone(reroute)
        self.assertNotEqual(reroute, proposed_action)
        self.assertEqual(reroute, 2)  # east: the column-8 detour

    def test_no_reroute_when_the_block_is_off_the_planned_route(self) -> None:
        # Blocking the column-8 connector cannot change a route that was
        # never going to use it -- the planner's first step is unaffected,
        # so there is nothing to call a "reroute": BACK_OFF/WAIT should
        # handle this conflict instead.
        player = self._player((3, 1))
        proposed_action = 3  # west, via col 1 -- col 8 is irrelevant to it
        reroute = choose_reroute_action(
            self.motion_planner,
            player,
            target_positions=[(4, 4)],
            human_pos=(8, 2),
            human_target=(8, 3),
            proposed_action=proposed_action,
        )
        self.assertIsNone(reroute)

    def test_no_target_positions_means_no_reroute(self) -> None:
        player = self._player((3, 1))
        reroute = choose_reroute_action(
            self.motion_planner,
            player,
            target_positions=[],
            human_pos=(1, 2),
            human_target=(1, 3),
            proposed_action=3,
        )
        self.assertIsNone(reroute)


if __name__ == "__main__":
    unittest.main()
