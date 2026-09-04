"""Guard rails for the step-denominated task prior (task_cost.py).

The task-point scale is a priority encoding lifted from the rule teacher:
lumpy, unitless, and it made "how much may we give up for a preference?"
unanswerable (10 points sat between a free swap and a 4x throughput loss).
This estimator prices every candidate in the same unit -- estimated steps
until this agent's next delivery -- so a tolerance can mean "at most N extra
steps".  It is a greedy serial plan with no partner model: good enough to
widen a band, never trusted to overrule the backbone's own pick.
"""

from __future__ import annotations

import unittest

from durf.baseline.collect_rule_teacher_dataset import (
    generate_candidate_subgoals,
    make_motion_planner,
)
from durf.baseline.task_cost import (
    WAITING_SUBGOALS,
    attach_step_costs,
    build_feature_map,
    estimate_candidate_costs,
)
from overcooked_ai_py.mdp.overcooked_mdp import ObjectState, OvercookedState, PlayerState, SoupState

LAYOUT = "ring_tomato_onion_10x6_h0_full_task"
POT = (4, 0)


def _state(ai_pos, held, human_pos, soup=None):
    ai = PlayerState(ai_pos, (0, -1), held_object=ObjectState(held, ai_pos) if held else None)
    human = PlayerState(human_pos, (0, -1))
    return OvercookedState(players=[ai, human], objects={POT: soup} if soup else {}, order_list=None)


class StepEstimatorTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.motion_planner = make_motion_planner(LAYOUT, seed=0, horizon=400)
        cls.mdp = cls.motion_planner.mdp
        cls.features = build_feature_map(cls.mdp, cls.motion_planner)

    def _costs(self, state):
        candidates = generate_candidate_subgoals(state, self.motion_planner, 0)
        attach_step_costs(
            candidates, features=self.features, motion_planner=self.motion_planner,
            player=state.players[0], state=state, mdp=self.mdp,
        )
        self.assertTrue(all(c.step_cost is not None for c in candidates), [c.subgoal for c in candidates])
        return {c.subgoal: c.step_cost for c in candidates}, candidates

    def test_every_runtime_candidate_is_priced(self) -> None:
        cooking = SoupState.get_soup(POT, num_onions=1, num_tomatoes=2, cooking_tick=3)
        ready = SoupState.get_soup(POT, num_onions=1, num_tomatoes=2, cooking_tick=20, finished=True)
        for state in (
            _state((5, 1), None, (2, 4), cooking),
            _state((5, 1), "dish", (2, 4)),
            _state((5, 1), "tomato", (2, 4), SoupState.get_soup(POT, num_tomatoes=2, cooking_tick=3)),
            _state((5, 1), None, (2, 4)),
            _state((5, 1), "dish", (2, 4), ready),
            _state((5, 1), "tomato", (2, 4)),
        ):
            self._costs(state)

    def test_division_of_labour_gap_is_the_tomato_round_trip(self) -> None:
        cooking = SoupState.get_soup(POT, num_onions=1, num_tomatoes=2, cooking_tick=3)
        costs, _ = self._costs(_state((5, 1), None, (2, 4), cooking))
        self.assertLess(costs["GET_DISH"], costs["GET_TOMATO"])
        # No partner is modelled, so fetching a tomato first is priced as the
        # extra trip this agent itself would walk before it can plate.
        self.assertGreaterEqual(costs["GET_TOMATO"] - costs["GET_DISH"], 5.0)
        self.assertLessEqual(costs["GET_TOMATO"] - costs["GET_DISH"], 12.0)

    def test_waiting_is_priced_one_step_behind_the_best_action(self) -> None:
        cooking = SoupState.get_soup(POT, num_onions=1, num_tomatoes=2, cooking_tick=3)
        costs, _ = self._costs(_state((5, 1), None, (2, 4), cooking))
        self.assertEqual(costs["WAIT"], min(costs["GET_DISH"], costs["GET_TOMATO"]) + 1.0)

    def test_dropping_a_needed_ingredient_prices_the_refetch(self) -> None:
        # Holding a tomato the empty pot needs, far from the dispenser: putting
        # it down means walking back for another one, not "best action + 1".
        costs, _ = self._costs(_state((7, 1), "tomato", (2, 4)))
        self.assertIn("PUT_TOMATO_IN_POT", costs)
        self.assertGreater(costs["PUT_DOWN_OBJECT"] - costs["PUT_TOMATO_IN_POT"], 3.0)

    def test_keeping_the_plate_is_cheaper_than_dropping_it(self) -> None:
        """Where the estimator disagrees with the backbone, and says so: with
        nothing cooking the rule teacher drops the plate (PUT_DOWN_OBJECT 20
        > WAIT_NEAR_POT 10), but dropping it means fetching it again later.
        The band may expose this to a preference; tolerance 0 still drops."""
        costs, _ = self._costs(_state((5, 1), "dish", (2, 4)))
        self.assertLess(costs["WAIT_NEAR_POT"], costs["PUT_DOWN_OBJECT"])

    def test_waiting_subgoals_are_exactly_the_state_preserving_ones(self) -> None:
        self.assertEqual(WAITING_SUBGOALS, {"WAIT", "WAIT_NEAR_POT", "CONTINUE_CURRENT_SUBGOAL"})
        self.assertNotIn("PUT_DOWN_OBJECT", WAITING_SUBGOALS)

    def test_estimate_is_position_aware(self) -> None:
        near = _state((1, 4), None, (2, 4))   # next to the tomato dispenser
        far = _state((8, 3), None, (2, 4))
        near_costs = estimate_candidate_costs(["GET_TOMATO"], features=self.features, motion_planner=self.motion_planner, player=near.players[0], state=near, mdp=self.mdp)
        far_costs = estimate_candidate_costs(["GET_TOMATO"], features=self.features, motion_planner=self.motion_planner, player=far.players[0], state=far, mdp=self.mdp)
        self.assertNotEqual(near_costs["GET_TOMATO"], far_costs["GET_TOMATO"])


if __name__ == "__main__":
    unittest.main()
