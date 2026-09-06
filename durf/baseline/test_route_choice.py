"""Guard rails for route choice when the partner is in the way.

The invariant `test_candidate_generation.py` states for candidate EXISTENCE --
"the partner never decides whether an option exists" -- has a twin one level
down, in route choice: the partner's transient position must not silently buy
an arbitrarily expensive detour either. Deciding whether to go around someone
or walk up to them and sort it out is a coordination judgement, and the
coordination layer can only make it if the encounter actually reaches it.

Before MAX_PARTNER_DETOUR_STEPS the rule teacher took any partner-avoiding
route at any price. In the ring layout that is a ~19-step trip around the loop
to dodge a partner two tiles ahead; the partner steps back the next tick, the
direct route wins again, and the pair mirrors each other indefinitely. That
livelock ran 692 of 800 steps in a recorded session, was invisible to both the
conflict detector (the two never contend for one tile) and the stall escape
(it only covers empty-handed WAIT), and held the H0 backbone to roughly a
quarter of its achievable score.
"""

from __future__ import annotations

import unittest

from durf.baseline.collect_rule_teacher_dataset import (
    MAX_PARTNER_DETOUR_STEPS,
    feature_candidate,
    make_motion_planner,
    route_first_action,
    route_to_feature,
)
from overcooked_ai_py.mdp.actions import Action
from overcooked_ai_py.mdp.overcooked_mdp import ObjectState, PlayerState

LAYOUT = "ring_tomato_onion_10x6_h0_full_task"
POT = (4, 0)
EAST = int(Action.ACTION_TO_INDEX[(1, 0)])
WEST = int(Action.ACTION_TO_INDEX[(-1, 0)])


class PartnerDetourBudgetTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.motion_planner = make_motion_planner(LAYOUT, seed=0, horizon=400)

    def _player(self, position, holding="dish"):
        held = ObjectState(holding, position) if holding else None
        return PlayerState(position, (0, -1), held_object=held)

    def test_a_long_detour_is_refused_and_the_encounter_is_flagged(self) -> None:
        # The recorded livelock state: AI one corridor tile from the partner,
        # both heading for the ready pot at the far end of a one-tile corridor.
        player = self._player((2, 1))
        _, avoiding_cost = route_to_feature(self.motion_planner, player, [POT], {(3, 1)})
        _, direct_cost = route_to_feature(self.motion_planner, player, [POT], set())
        self.assertGreater(avoiding_cost - direct_cost, MAX_PARTNER_DETOUR_STEPS)

        action, ignored_partner = route_first_action(
            self.motion_planner, player, [POT], {(3, 1)}
        )
        self.assertEqual(action, EAST)  # towards the pot, not around the ring
        self.assertTrue(ignored_partner)

    def test_the_mirrored_state_gives_the_same_direction(self) -> None:
        """The oscillation itself: one step later the partner has stepped back
        and the AI used to flip direction. Both states must now agree."""
        for ai_pos, human_pos in (((1, 1), (4, 1)), ((2, 1), (3, 1)), ((2, 1), (4, 1)), ((3, 1), (4, 1))):
            action, _ = route_first_action(
                self.motion_planner, self._player(ai_pos), [POT], {human_pos}
            )
            self.assertEqual(action, EAST, (ai_pos, human_pos))

    def test_a_detour_inside_the_budget_is_still_taken(self) -> None:
        """The budget bounds the price of politeness, it does not abolish it."""
        player = self._player((3, 4))
        blocked = {(2, 4)}
        _, avoiding_cost = route_to_feature(self.motion_planner, player, [POT], blocked)
        _, direct_cost = route_to_feature(self.motion_planner, player, [POT], set())
        self.assertLessEqual(avoiding_cost - direct_cost, MAX_PARTNER_DETOUR_STEPS)

        action, ignored_partner = route_first_action(
            self.motion_planner, player, [POT], blocked
        )
        self.assertEqual(action, EAST)  # away from the partner, the long way round
        self.assertFalse(ignored_partner)
        direct_action, _ = route_first_action(
            self.motion_planner, player, [POT], blocked, detour_budget=0
        )
        self.assertEqual(direct_action, WEST)  # the route the partner is standing on

    def test_an_unbounded_budget_restores_the_old_behaviour(self) -> None:
        action, ignored_partner = route_first_action(
            self.motion_planner, self._player((2, 1)), [POT], {(3, 1)}, detour_budget=None
        )
        self.assertEqual(action, WEST)  # the ~19-step trip around the ring
        self.assertFalse(ignored_partner)

    def test_candidate_records_the_partner_blocked_route_for_coordination(self) -> None:
        candidate = feature_candidate(
            subgoal="PICKUP_SOUP",
            task_score=95.0,
            reason="held_dish_and_soup_ready",
            motion_planner=self.motion_planner,
            player=self._player((2, 1)),
            feature_positions=[POT],
            blocked_positions={(3, 1)},
        )
        self.assertIsNotNone(candidate)
        self.assertEqual(candidate.action, EAST)
        self.assertTrue(candidate.metadata["route_blocked_by_partner"])

    def test_no_route_at_all_still_falls_back_to_the_direct_one(self) -> None:
        """The partner standing ON the goal tile leaves no avoiding route; the
        candidate must survive (existence is not the partner's call) and say so."""
        player = self._player((1, 1))
        avoiding_action, avoiding_cost = route_to_feature(
            self.motion_planner, player, [POT], {(4, 1)}
        )
        self.assertIsNone(avoiding_action)
        self.assertEqual(avoiding_cost, float("inf"))

        action, ignored_partner = route_first_action(
            self.motion_planner, player, [POT], {(4, 1)}
        )
        self.assertEqual(action, EAST)
        self.assertTrue(ignored_partner)


if __name__ == "__main__":
    unittest.main()
