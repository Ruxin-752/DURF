from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import patch

from durf.baseline.h0_planner import (
    SUBGOALS,
    dish_pickup_locations,
    empty_non_feature_counter_locations,
    execute_subgoal,
    first_action_to_feature,
    plan_h0_subgoal,
    subgoal_target_positions,
    useful_stash_counter_locations,
)
from durf.group_a.play_with_baseline import h0_executor_action


class FakeModel:
    def predict(self, inputs, verbose=0):
        raise AssertionError("Exact-origin H0 must not call Keras predict")


class FakeBaseEnv:
    state = object()


class PlannerHelperTests(unittest.TestCase):
    def test_placed_dish_is_a_pickup_target_before_dispenser(self) -> None:
        state = SimpleNamespace(
            objects={
                (2, 2): SimpleNamespace(name="dish"),
                (3, 3): SimpleNamespace(name="tomato"),
            }
        )
        mdp = SimpleNamespace(get_dish_dispenser_locations=lambda: [(9, 9)])
        self.assertEqual(dish_pickup_locations(state, mdp), [(2, 2), (9, 9)])

    def test_blocked_feature_moves_toward_a_staging_tile(self) -> None:
        valid = {
            (1, 4),
            (1, 3),
            (1, 2),
            (1, 1),
            (2, 1),
            (3, 1),
            (4, 1),
            (5, 1),
        }
        mdp = SimpleNamespace(get_valid_player_positions=lambda: list(valid))
        planner = SimpleNamespace(mdp=mdp, motion_goals_for_pos={})
        player = SimpleNamespace(position=(1, 4), orientation=(0, -1), pos_and_or=((1, 4), (0, -1)))
        action = first_action_to_feature(
            planner,
            player,
            [(4, 0)],
            blocked_positions={(4, 1)},
        )
        self.assertIsNotNone(action)
        self.assertNotEqual(action, 4)

    def test_stash_targets_only_empty_non_feature_counters(self) -> None:
        state = SimpleNamespace(objects={(2, 2): SimpleNamespace(name="onion")})
        mdp = SimpleNamespace(
            get_counter_locations=lambda: [(1, 1), (2, 2), (3, 3)],
            get_pot_locations=lambda: [(3, 3)],
            get_serving_locations=lambda: [],
            get_dish_dispenser_locations=lambda: [],
            get_tomato_dispenser_locations=lambda: [],
            get_onion_dispenser_locations=lambda: [],
        )
        self.assertEqual(
            empty_non_feature_counter_locations(state, mdp),
            [(1, 1)],
        )

    def test_stash_prefers_an_empty_counter_near_the_pot(self) -> None:
        state = SimpleNamespace(objects={})
        mdp = SimpleNamespace(
            get_counter_locations=lambda: [(1, 0), (3, 0), (5, 0), (8, 0)],
            get_pot_locations=lambda: [(4, 0)],
            get_serving_locations=lambda: [],
            get_dish_dispenser_locations=lambda: [],
            get_tomato_dispenser_locations=lambda: [],
            get_onion_dispenser_locations=lambda: [],
        )
        self.assertEqual(
            useful_stash_counter_locations(state, mdp),
            [(3, 0), (5, 0)],
        )

    def test_stash_executor_routes_to_empty_counter(self) -> None:
        ai = SimpleNamespace(
            position=(1, 2),
            orientation=(1, 0),
            pos_and_or=((1, 2), (1, 0)),
            held_object=SimpleNamespace(name="onion"),
        )
        human = SimpleNamespace(position=(9, 9))
        state = SimpleNamespace(players=[ai, human], objects={})
        mdp = SimpleNamespace(
            get_counter_locations=lambda: [(3, 1)],
            get_pot_locations=lambda: [],
            get_serving_locations=lambda: [],
            get_dish_dispenser_locations=lambda: [],
            get_tomato_dispenser_locations=lambda: [],
            get_onion_dispenser_locations=lambda: [],
            get_valid_player_positions=lambda: [(1, 2), (2, 2), (3, 2)],
        )
        planner = SimpleNamespace(mdp=mdp, motion_goals_for_pos={})
        self.assertNotEqual(
            execute_subgoal(state, planner, "STASH_HELD_OBJECT"),
            4,
        )

    def test_h0_stashes_ingredient_that_no_pot_can_accept(self) -> None:
        soup = SimpleNamespace(name="soup", is_ready=False, is_cooking=True)
        ai = SimpleNamespace(held_object=SimpleNamespace(name="onion"))
        state = SimpleNamespace(
            players=[ai],
            has_object=lambda position: True,
            get_object=lambda position: soup,
        )
        mdp = SimpleNamespace(
            start_all_orders=[["tomato", "tomato", "onion"]],
            get_pot_locations=lambda: [(4, 0)],
            get_pot_states=lambda state: {},
        )
        self.assertEqual(plan_h0_subgoal(state, mdp), "STASH_HELD_OBJECT")

    def test_prefetch_does_not_repick_a_staged_ingredient(self) -> None:
        cooking = SimpleNamespace(name="soup", is_ready=False, is_cooking=True)
        staged = SimpleNamespace(name="onion")
        objects = {(4, 0): cooking, (7, 0): staged}
        state = SimpleNamespace(
            objects=objects,
            has_object=lambda position: position in objects,
            get_object=lambda position: objects[position],
        )
        mdp = SimpleNamespace(
            start_all_orders=[["tomato", "tomato", "onion"]],
            get_pot_locations=lambda: [(4, 0)],
            get_pot_states=lambda _state: {"cooking": [(4, 0)]},
            get_ready_pots=lambda _pot_states: [],
            get_onion_dispenser_locations=lambda: [(9, 0)],
            get_tomato_dispenser_locations=lambda: [(0, 4)],
            get_dish_dispenser_locations=lambda: [(0, 1)],
        )

        self.assertEqual(
            subgoal_target_positions(state, mdp, "GET_ONION"),
            [(9, 0)],
        )

    def test_open_pot_reuses_a_staged_ingredient_before_dispenser(self) -> None:
        staged = SimpleNamespace(name="tomato")
        objects = {(7, 0): staged}
        state = SimpleNamespace(
            objects=objects,
            has_object=lambda position: position in objects,
            get_object=lambda position: objects[position],
        )
        mdp = SimpleNamespace(
            start_all_orders=[["tomato", "tomato", "onion"]],
            get_pot_locations=lambda: [(4, 0)],
            get_pot_states=lambda _state: {},
            get_ready_pots=lambda _pot_states: [],
            get_tomato_dispenser_locations=lambda: [(0, 4)],
            get_onion_dispenser_locations=lambda: [(9, 0)],
            get_dish_dispenser_locations=lambda: [(0, 1)],
        )

        self.assertEqual(
            subgoal_target_positions(state, mdp, "GET_TOMATO"),
            [(7, 0), (0, 4)],
        )


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
