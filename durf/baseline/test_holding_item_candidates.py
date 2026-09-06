"""Guard rails for the holding-item candidate loosening (mechanism_blueprint_v1.md
S3) and the retirement of recovery_action_override's two forcing triggers.

Two invariants this project relies on:

1. The task layer's own candidate pool, not a downstream override, is where
   Hu-visible alternatives live. AI_HELD_DISH_BEFORE_SOUP_READY and
   AI_HELD_UNNEEDED_INGREDIENT used to silently overrule whatever the task/Hu
   layer picked; their exact logic is now reproduced as scored candidates
   inside generate_candidate_subgoals, so Hu can see and rank them instead.
2. Zero behavior change at hu_task_tolerance=0: wherever the branch's own
   primary candidate is feasible, the new top-scored candidate must compute
   the IDENTICAL action the retired override used to force. Where the
   primary candidate is infeasible, the exact old single-candidate fallback
   must come back unchanged -- new low-score alternatives must never be
   offered in a state the frozen H0 backbone never actually visited before.

This was additionally verified empirically: a 40-episode replay sweep
(4 personas x 10 seeds x 400 steps = 16000 real AI decision points) compared
this module's task-layer choice against the override-adjusted final action
the pre-retirement runtime actually took, on the identical historical state
at every step -- 0 mismatches. See docs/mechanism_blueprint_v1.md S3.3.
"""

from __future__ import annotations

import unittest

from durf.baseline.collect_rule_teacher_dataset import (
    choose_task_candidate,
    generate_candidate_subgoals,
    make_motion_planner,
    put_down_candidates,
)
import durf.baseline.collect_rule_teacher_dataset as crtd
from overcooked_ai_py.mdp.overcooked_mdp import ObjectState, OvercookedState, PlayerState, SoupState

LAYOUT = "ring_tomato_onion_10x6_h0_full_task"
POT_POSITION = (4, 0)


def _state(ai_pos, ai_held, human_pos, pot_soup=None) -> OvercookedState:
    ai = PlayerState(ai_pos, (0, -1), held_object=ai_held)
    human = PlayerState(human_pos, (0, -1))
    objects = {}
    if pot_soup is not None:
        objects[POT_POSITION] = pot_soup
    return OvercookedState(players=[ai, human], objects=objects, order_list=None)


class HoldingUnneededIngredientTest(unittest.TestCase):
    """put_down_candidates: the retired AI_HELD_UNNEEDED_INGREDIENT trigger."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.motion_planner = make_motion_planner(LAYOUT, seed=0, horizon=400)

    def test_put_down_is_top_pick_with_wait_near_pot_and_wait_below_it(self) -> None:
        # Pot already has both tomatoes it needs and is cooking -- a held
        # tomato is unneeded, exactly the AI_HELD_UNNEEDED_INGREDIENT state.
        cooking = SoupState.get_soup(POT_POSITION, num_tomatoes=2, cooking_tick=3)
        tomato = ObjectState("tomato", (5, 1))
        state = _state((5, 1), tomato, (2, 1), pot_soup=cooking)
        candidates = generate_candidate_subgoals(state, self.motion_planner, 0)
        by_subgoal = {c.subgoal: c for c in candidates}
        self.assertIn("PUT_DOWN_OBJECT", by_subgoal)
        self.assertIn("WAIT_NEAR_POT", by_subgoal)
        self.assertIn("WAIT", by_subgoal)
        self.assertEqual(by_subgoal["PUT_DOWN_OBJECT"].task_score, 60.0)
        self.assertEqual(by_subgoal["WAIT_NEAR_POT"].task_score, 10.0)
        self.assertEqual(by_subgoal["WAIT"].task_score, 0.0)
        chosen = choose_task_candidate(candidates, task_tolerance=0.0)
        self.assertEqual(chosen.subgoal, "PUT_DOWN_OBJECT")

    def test_infeasible_put_down_falls_back_to_the_old_single_candidate(self) -> None:
        """No offering WAIT_NEAR_POT above plain WAIT in a state H0 never saw."""
        cooking = SoupState.get_soup(POT_POSITION, num_tomatoes=2, cooking_tick=3)
        tomato = ObjectState("tomato", (5, 1))
        state = _state((5, 1), tomato, (2, 1), pot_soup=cooking)

        orig_feature_candidate = crtd.feature_candidate

        def blocked(*, subgoal, **kwargs):
            if subgoal == "PUT_DOWN_OBJECT":
                return None
            return orig_feature_candidate(subgoal=subgoal, **kwargs)

        crtd.feature_candidate = blocked
        try:
            candidates = put_down_candidates(
                state,
                self.motion_planner.mdp,
                self.motion_planner,
                state.players[0],
                {tuple(state.players[1].position)},
                "tomato",
            )
        finally:
            crtd.feature_candidate = orig_feature_candidate

        self.assertEqual(len(candidates), 1)
        self.assertEqual(candidates[0].subgoal, "WAIT")
        self.assertEqual(candidates[0].reason, "holding_unneeded_ingredient")


class HoldingDishBeforeSoupReadyTest(unittest.TestCase):
    """generate_candidate_subgoals' dish/soup-not-ready branch: the retired
    AI_HELD_DISH_BEFORE_SOUP_READY trigger, reproduced as a state-dependent
    two-way split on whether a pot is actually cooking."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.motion_planner = make_motion_planner(LAYOUT, seed=0, horizon=400)

    def test_pot_cooking_prefers_waiting_near_it_over_putting_the_dish_down(self) -> None:
        cooking = SoupState.get_soup(POT_POSITION, num_onions=1, num_tomatoes=2, cooking_tick=3)
        dish = ObjectState("dish", (5, 1))
        state = _state((5, 1), dish, (2, 1), pot_soup=cooking)
        candidates = generate_candidate_subgoals(state, self.motion_planner, 0)
        by_subgoal = {c.subgoal: c for c in candidates}
        self.assertEqual(by_subgoal["WAIT_NEAR_POT"].task_score, 15.0)
        self.assertEqual(by_subgoal["WAIT"].task_score, 5.0)
        self.assertEqual(by_subgoal["PUT_DOWN_OBJECT"].task_score, 2.0)
        chosen = choose_task_candidate(candidates, task_tolerance=0.0)
        self.assertEqual(chosen.subgoal, "WAIT_NEAR_POT")

    def test_no_pot_cooking_prefers_putting_the_dish_down(self) -> None:
        dish = ObjectState("dish", (5, 1))
        state = _state((5, 1), dish, (2, 1), pot_soup=None)
        candidates = generate_candidate_subgoals(state, self.motion_planner, 0)
        by_subgoal = {c.subgoal: c for c in candidates}
        self.assertEqual(by_subgoal["PUT_DOWN_OBJECT"].task_score, 20.0)
        self.assertEqual(by_subgoal["WAIT_NEAR_POT"].task_score, 10.0)
        self.assertEqual(by_subgoal["WAIT"].task_score, 5.0)
        chosen = choose_task_candidate(candidates, task_tolerance=0.0)
        self.assertEqual(chosen.subgoal, "PUT_DOWN_OBJECT")

    def test_soup_ready_is_a_different_branch_entirely(self) -> None:
        ready = SoupState.get_soup(
            POT_POSITION, num_onions=1, num_tomatoes=2, cooking_tick=20, finished=True
        )
        dish = ObjectState("dish", (5, 1))
        state = _state((5, 1), dish, (2, 1), pot_soup=ready)
        candidates = generate_candidate_subgoals(state, self.motion_planner, 0)
        chosen = choose_task_candidate(candidates, task_tolerance=0.0)
        self.assertEqual(chosen.subgoal, "PICKUP_SOUP")

    def test_infeasible_put_down_when_no_pot_cooking_falls_back_to_plain_wait(self) -> None:
        """The override never chose WAIT_NEAR_POT in this sub-case (no pot is
        even cooking yet), so an infeasible PUT_DOWN_OBJECT here must fall
        back to the exact historical WAIT, not WAIT_NEAR_POT."""
        dish = ObjectState("dish", (5, 1))
        state = _state((5, 1), dish, (2, 1), pot_soup=None)

        orig_feature_candidate = crtd.feature_candidate

        def blocked(*, subgoal, **kwargs):
            if subgoal == "PUT_DOWN_OBJECT":
                return None
            return orig_feature_candidate(subgoal=subgoal, **kwargs)

        crtd.feature_candidate = blocked
        try:
            candidates = generate_candidate_subgoals(state, self.motion_planner, 0)
        finally:
            crtd.feature_candidate = orig_feature_candidate

        self.assertEqual(len(candidates), 1)
        self.assertEqual(candidates[0].subgoal, "WAIT")
        self.assertEqual(candidates[0].reason, "holding_dish_waiting_for_soup")


if __name__ == "__main__":
    unittest.main()
