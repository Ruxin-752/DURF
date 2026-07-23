"""Focused tests for Hu condition extraction and candidate event detectors."""

from __future__ import annotations

import unittest

from durf.feedback_attribution.condition_features import extract_condition_features
from durf.feedback_attribution.event_detectors import (
    detect_ai_missed_labor_division_opportunity,
    detect_ai_missed_useful_counter_object,
)
from durf.feedback_attribution.sample_builder import candidates_visible_at_feedback
from durf.hu.subgoal_reranker import LinearSubgoalReranker


TERRAIN = [
    "XXPXX",
    "D   X",
    "T   X",
    "X   X",
    "XXOXX",
]


def wrapped_object(
    name: str,
    position: tuple[int, int],
    **extra,
) -> dict:
    return {
        "position": list(position),
        "object": {
            "name": name,
            "position": list(position),
            **extra,
        },
    }


def make_step(
    total_step: int,
    *,
    ai_pos: tuple[int, int] = (3, 1),
    human_pos: tuple[int, int] = (1, 3),
    ai_held: str | None = None,
    human_held: str | None = None,
    objects: list[dict] | None = None,
    pot_states: dict | None = None,
    ai_subgoal: str = "WAIT",
) -> dict:
    def held(name: str | None, position: tuple[int, int]) -> dict | None:
        if name is None:
            return None
        return {"name": name, "position": list(position)}

    facts = {
        "ai_pos": list(ai_pos),
        "human_pos": list(human_pos),
        "ai_held_object": held(ai_held, ai_pos),
        "human_held_object": held(human_held, human_pos),
        "objects": objects or [],
        "pot_states": pot_states or {"empty": [[2, 0]]},
        "layout_features": {
            "layout_name": "ring_tomato_onion_test",
            "terrain": TERRAIN,
        },
    }
    return {
        "total_step": total_step,
        "ai_action_name": "stay",
        "human_action_name": "stay",
        "ai_subgoal": ai_subgoal,
        "state_facts": facts,
        "extra": {"state_before": facts},
    }


class ConditionFeatureTests(unittest.TestCase):
    def test_nested_pot_soup_and_last_ingredient_are_detected(self):
        step = make_step(
            1,
            human_held="onion",
            objects=[
                wrapped_object(
                    "soup",
                    (2, 0),
                    ingredients=["tomato", "tomato"],
                    is_cooking=False,
                    is_ready=False,
                )
            ],
            pot_states={"2_items": [[2, 0]]},
        )

        features = extract_condition_features(step)

        self.assertFalse(features["recipe_needs_tomato"])
        self.assertTrue(features["recipe_needs_onion"])
        self.assertTrue(features["pot_partially_filled"])
        self.assertTrue(features["human_holding_last_needed_ingredient"])
        self.assertEqual(features["human_inferred_subgoal"], "PUT_ONION_IN_POT")

    def test_counter_object_keeps_raw_context_and_boolean_model_features(self):
        step = make_step(
            1,
            ai_pos=(3, 1),
            objects=[wrapped_object("tomato", (3, 0))],
        )

        features = extract_condition_features(step)

        self.assertTrue(features["useful_counter_object_available"])
        self.assertTrue(features["useful_counter_object_closer_than_dispenser"])
        self.assertTrue(
            features["useful_counter_object_closer_to_pot_than_dispenser"]
        )
        self.assertEqual(features["useful_counter_object_type"], "tomato")
        self.assertEqual(features["useful_counter_object_position"], [3, 0])
        self.assertEqual(features["useful_counter_object_distance"], 0)

    def test_human_trying_to_pass_requires_ai_on_attempted_target(self):
        step = make_step(1)
        step["human_action_name"] = "north"

        features = extract_condition_features(step)

        self.assertFalse(features["human_trying_to_pass"])
        self.assertFalse(features["ai_on_human_path"])

    def test_old_hu_condition_dimension_remains_loadable(self):
        model = LinearSubgoalReranker(condition_keys=("pot_empty",))

        score = model.score("UNKNOWN_USER", {"pot_empty": True}, "WAIT")

        self.assertIsInstance(score, float)


class CandidateEventTests(unittest.TestCase):
    def test_detects_ignored_closer_counter_object(self):
        steps = [
            make_step(
                total_step,
                ai_pos=(3, 1),
                objects=[wrapped_object("tomato", (3, 0))],
                ai_subgoal="GET_TOMATO",
            )
            for total_step in (1, 2, 3)
        ]

        events = detect_ai_missed_useful_counter_object(steps)

        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["event_type"], "AI_missed_useful_counter_object")
        self.assertEqual(events[0]["evidence"]["object_type"], "tomato")

    def test_detects_complementary_dish_opportunity(self):
        objects = [
            wrapped_object(
                "soup",
                (2, 0),
                ingredients=["tomato", "tomato"],
                is_cooking=False,
                is_ready=False,
            )
        ]
        steps = [
            make_step(
                total_step,
                human_held="onion",
                objects=objects,
                pot_states={"2_items": [[2, 0]]},
                ai_subgoal="WAIT",
            )
            for total_step in (1, 2)
        ]

        events = detect_ai_missed_labor_division_opportunity(steps)

        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["event_type"], "AI_missed_labor_division_opportunity")
        self.assertEqual(
            events[0]["evidence"]["opportunity_kind"],
            "human_covers_last_ingredient",
        )
        self.assertEqual(events[0]["alternative_subgoals"], ["GET_DISH"])

    def test_feedback_candidates_do_not_include_future_evidence(self):
        objects = [
            wrapped_object(
                "soup",
                (2, 0),
                ingredients=["tomato", "tomato"],
                is_cooking=False,
                is_ready=False,
            )
        ]
        trajectory = [
            make_step(
                total_step,
                human_held="onion",
                objects=objects,
                pot_states={"2_items": [[2, 0]]},
                ai_subgoal="WAIT",
            )
            for total_step in (1, 2, 3)
        ]
        saved_future_event = {
            "event_type": "AI_missed_labor_division_opportunity",
            "start_timestep": 1,
            "end_timestep": 3,
        }

        visible = candidates_visible_at_feedback(
            trajectory_window=trajectory[:2],
            candidate_events=[saved_future_event],
            feedback_total_step=2,
            lookback_steps=25,
        )

        self.assertTrue(visible)
        self.assertTrue(all(event["end_timestep"] <= 2 for event in visible))
        self.assertEqual(visible[0]["end_timestep"], 2)


if __name__ == "__main__":
    unittest.main()
