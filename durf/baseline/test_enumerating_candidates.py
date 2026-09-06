"""Guard rails for the enumerating candidate generator (2026-09-05).

The generator used to be written as a POLICY: decide which situation we are in,
return that situation's action plus a couple of fallbacks.  We had been using it
as a MENU for the preference layer to reorder, and it was never rewritten for
that job, so three real choices were invisible:

  * the pot needs both a tomato and an onion on 63.7% of steps, but only the
    first was ever emitted, so "fetch the onion first instead" was not a losing
    option -- it was not an option;
  * the same collapse in the next-cycle prep branch;
  * ``GET_DISH`` was DELETED whenever the teammate already carried a dish, so
    no feedback could ever ask for a second one.

Measured effect at the tolerance actually used (1 step):

    decision points with >= 2 competing REAL actions
      mixed corpus (13 009 states: stochastic rollouts + recorded sessions)
        before:     0  (0.0%)
        after:  6 331  (48.7%)
      recorded-human states only (4 077 distinct)
        before:     0  (0.0%)
        after:  1 274  (31.2%)

Be honest about what that number is: ~90% of it is ONE pair, GET_ONION vs
GET_TOMATO ("which ingredient first").  The other combinations -- GET_DISH vs an
ingredient, and three-way -- are a few hundred cases in total.  The layout has
one pot and a two-ingredient recipe, so this is the choice it mostly offers.
"Real action" excludes WAIT / WAIT_NEAR_POT / CONTINUE_CURRENT_SUBGOAL and
PUT_DOWN_OBJECT (a one-shot world change, not work toward a delivery).

The invariant that makes those numbers safe to add is that H0 must be
untouched: at tolerance 0 the task optimum has to stay the same subgoal, or
"the preference changed the behaviour" stops being attributable.  Every
addition therefore scores strictly below the historical primary of its own
branch, and where the historical primary is infeasible the whole group is
dropped rather than letting an alternative inherit the top score.

Verified by shadow replay: the same 13 009 states (both player indices, 26 018
decisions) decided under the old and new trees gave 0 action mismatches and 0
subgoal mismatches, while 6 453 menus grew; an independent exhaustive synthetic
sweep (131 328 states: every open tile x orientation x pot content x held items)
also gave 0 mismatches and no ties at the optimum.
Whole-trajectory diffs cannot show this -- both players run this same function,
so the trajectories diverge for reasons unrelated to player 0's rule.
"""

from __future__ import annotations

import unittest

import durf.baseline.collect_rule_teacher_dataset as crtd
from durf.baseline.collect_rule_teacher_dataset import (
    TEAMMATE_ALREADY_HAS_DISH_SCORE,
    acceptable_candidates,
    choose_task_candidate,
    generate_candidate_subgoals,
    make_motion_planner,
    next_needed_ingredients,
)
from durf.baseline.task_cost import attach_step_costs, build_feature_map
from overcooked_ai_py.mdp.overcooked_mdp import (
    ObjectState,
    OvercookedState,
    PlayerState,
    SoupState,
)

LAYOUT = "ring_tomato_onion_10x6_h0_full_task"
POT = (4, 0)
AI = (5, 1)
HUMAN = (2, 1)


def _soup(names, cooking_tick=-1):
    return SoupState(
        POT, ingredients=[ObjectState(n, POT) for n in names], cooking_tick=cooking_tick
    )


def _state(*, ai_held=None, human_held=None, soup=None, ai=AI, human=HUMAN):
    objects = {POT: soup} if soup is not None else {}
    return OvercookedState(
        players=[
            PlayerState(ai, (0, -1), held_object=ai_held),
            PlayerState(human, (0, -1), held_object=human_held),
        ],
        objects=objects,
        order_list=None,
    )


class EnumeratingGeneratorTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.planner = make_motion_planner(LAYOUT, seed=0, horizon=400)
        cls.mdp = cls.planner.mdp

    def _menu(self, state):
        return {
            c.subgoal: c.task_score
            for c in generate_candidate_subgoals(state, self.planner, 0)
        }

    # ---------------------------------------------------- the second ingredient
    def test_pot_needing_two_kinds_offers_both(self) -> None:
        state = _state(soup=_soup(["tomato"]))
        self.assertEqual(next_needed_ingredients(state, self.mdp), ["tomato", "onion"])
        menu = self._menu(state)
        self.assertEqual(menu["GET_TOMATO"], 70.0)
        self.assertEqual(menu["GET_ONION"], 65.0)

    def test_the_alternative_never_outranks_the_historical_primary(self) -> None:
        state = _state(soup=_soup(["tomato"]))
        menu = self._menu(state)
        self.assertLess(menu["GET_ONION"], menu["GET_TOMATO"])
        self.assertEqual(
            choose_task_candidate(
                generate_candidate_subgoals(state, self.planner, 0), task_tolerance=0.0
            ).subgoal,
            "GET_TOMATO",
        )

    def test_a_pot_needing_one_kind_still_offers_only_that_one(self) -> None:
        """Two tomatoes in: only the onion is missing, so there is nothing to
        choose between and no phantom alternative may appear."""
        state = _state(soup=_soup(["tomato", "tomato"]))
        self.assertEqual(next_needed_ingredients(state, self.mdp), ["onion"])
        menu = self._menu(state)
        self.assertEqual(menu["GET_ONION"], 70.0)
        self.assertNotIn("GET_TOMATO", menu)

    def test_infeasible_primary_drops_the_whole_group(self) -> None:
        """If the historical primary has no route, the alternative must not
        inherit the top score -- that would be a state H0 never decided this
        way."""
        state = _state(soup=_soup(["tomato"]))
        original = crtd.feature_candidate

        def blocked(*, subgoal, task_score, **kwargs):
            if task_score == 70.0:
                return None
            return original(subgoal=subgoal, task_score=task_score, **kwargs)

        crtd.feature_candidate = blocked
        try:
            menu = self._menu(state)
        finally:
            crtd.feature_candidate = original
        self.assertNotIn("GET_ONION", menu)
        self.assertNotIn("GET_TOMATO", menu)

    # -------------------------------------------------- the teammate's dish
    def test_second_dish_is_scored_not_deleted(self) -> None:
        state = _state(
            soup=_soup(["tomato", "tomato", "onion"], cooking_tick=3),
            human_held=ObjectState("dish", HUMAN),
        )
        menu = self._menu(state)
        self.assertIn("GET_DISH", menu)
        self.assertEqual(menu["GET_DISH"], TEAMMATE_ALREADY_HAS_DISH_SCORE)

    def test_second_dish_scores_below_waiting_so_h0_can_never_pick_it(self) -> None:
        state = _state(
            soup=_soup(["tomato", "tomato", "onion"], cooking_tick=3),
            human_held=ObjectState("dish", HUMAN),
        )
        menu = self._menu(state)
        self.assertLess(menu["GET_DISH"], menu["WAIT"])
        self.assertNotEqual(
            choose_task_candidate(
                generate_candidate_subgoals(state, self.planner, 0), task_tolerance=0.0
            ).subgoal,
            "GET_DISH",
        )

    def test_without_a_teammate_dish_it_keeps_its_historical_score(self) -> None:
        state = _state(soup=_soup(["tomato", "tomato", "onion"], cooking_tick=3))
        self.assertEqual(self._menu(state)["GET_DISH"], 60.0)

    # ------------------------------------------------------ the prep branch
    # ------------------------------------------- partner carries the last one
    def test_partner_carrying_last_ingredient_puts_dish_on_the_menu(self) -> None:
        """Pot needs one onion, the human is carrying it: the pot cooks next.
        GET_DISH becomes an option (division of labour), well below GET_ONION
        so H0 still fetches the onion."""
        state = _state(
            soup=_soup(["tomato", "tomato"]),
            human_held=ObjectState("onion", HUMAN),
        )
        menu = self._menu(state)
        self.assertEqual(menu["GET_ONION"], 70.0)
        self.assertEqual(menu["GET_DISH"], 40.0)
        self.assertEqual(
            choose_task_candidate(
                generate_candidate_subgoals(state, self.planner, 0), task_tolerance=0.0
            ).subgoal,
            "GET_ONION",
        )

    def test_partner_carrying_a_non_last_ingredient_adds_nothing(self) -> None:
        state = _state(soup=_soup(["tomato"]), human_held=ObjectState("onion", HUMAN))
        self.assertNotIn("GET_DISH", self._menu(state))

    def test_partner_carrying_the_wrong_ingredient_adds_nothing(self) -> None:
        state = _state(
            soup=_soup(["tomato", "tomato"]), human_held=ObjectState("tomato", HUMAN)
        )
        self.assertNotIn("GET_DISH", self._menu(state))

    def test_prep_branch_offers_the_other_ingredient_just_below(self) -> None:
        state = _state(soup=_soup(["tomato", "tomato", "onion"], cooking_tick=3))
        menu = self._menu(state)
        self.assertEqual(menu["GET_TOMATO"], 50.0)
        self.assertEqual(menu["GET_ONION"], 47.0)

    # ------------------------------------------------------------ invariants
    def test_tolerance_zero_leaves_exactly_one_acceptable_option(self) -> None:
        """No ties at the optimum: if two candidates shared the top score, H0
        would become order-dependent and the 'preference off' control would
        stop being reproducible."""
        for state in (
            _state(soup=_soup(["tomato"])),
            _state(soup=_soup(["tomato", "tomato"])),
            _state(soup=_soup(["tomato", "tomato", "onion"], cooking_tick=3)),
            _state(
                soup=_soup(["tomato", "tomato", "onion"], cooking_tick=3),
                human_held=ObjectState("dish", HUMAN),
            ),
        ):
            band = acceptable_candidates(
                generate_candidate_subgoals(state, self.planner, 0), task_tolerance=0.0
            )
            self.assertEqual(band.size, 1, [c.subgoal for c in band.acceptable])

    def test_a_real_choice_appears_once_tolerance_is_opened(self) -> None:
        state = _state(soup=_soup(["tomato"]))
        candidates = generate_candidate_subgoals(state, self.planner, 0)
        attach_step_costs(
            candidates,
            features=build_feature_map(self.mdp, self.planner),
            motion_planner=self.planner,
            player=state.players[0],
            state=state,
            mdp=self.mdp,
        )
        band = acceptable_candidates(candidates, step_tolerance=1.0)
        self.assertIn("GET_TOMATO", [c.subgoal for c in band.acceptable])
        self.assertIn("GET_ONION", [c.subgoal for c in band.acceptable])


if __name__ == "__main__":
    unittest.main()
