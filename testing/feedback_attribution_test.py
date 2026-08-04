"""Focused tests for Hu condition extraction and candidate event detectors."""

from __future__ import annotations

import unittest

from durf.feedback_attribution.condition_features import extract_condition_features
from durf.feedback_attribution.event_detectors import (
    detect_coordination_decision_events,
    detect_ai_missed_labor_division_opportunity,
    detect_ai_missed_useful_counter_object,
)
from durf.feedback_attribution.hu_dataset_builder import (
    build_provenance_record,
    build_training_samples,
)
from durf.feedback_attribution.sample_builder import candidates_visible_at_feedback
from durf.feedback_attribution.subgoal_preferences import (
    infer_subgoal_preferences,
)
from durf.hu.subgoal_reranker import (
    COORDINATION_DECISION_LEVEL,
    TASK_DECISION_LEVEL,
    HierarchicalHu,
    LinearSubgoalReranker,
    PairwiseSample,
)
from durf.hu.train_subgoal_reranker import split_samples


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

    def test_hierarchical_hu_updates_decision_heads_independently(self):
        samples = [
            PairwiseSample(
                user_id="PILOT01",
                layout="test",
                condition_features={"pot_empty": True},
                preferred_subgoal="GET_TOMATO",
                rejected_subgoal="WAIT",
                decision_level=TASK_DECISION_LEVEL,
            ),
            PairwiseSample(
                user_id="PILOT01",
                layout="test",
                condition_features={
                    "human_trying_to_pass": True,
                    "ai_adjacent_to_current_subgoal_target": True,
                },
                preferred_subgoal="CONTINUE_CURRENT_SUBGOAL",
                rejected_subgoal="YIELD",
                decision_level=COORDINATION_DECISION_LEVEL,
            ),
        ]
        model = HierarchicalHu.from_samples(samples, seed=1)

        model.train(samples, epochs=30, learning_rate=0.05, seed=1)

        task_margin = model.score(
            TASK_DECISION_LEVEL,
            "PILOT01",
            {"pot_empty": True},
            "GET_TOMATO",
        ) - model.score(
            TASK_DECISION_LEVEL,
            "PILOT01",
            {"pot_empty": True},
            "WAIT",
        )
        coordination_margin = model.score(
            COORDINATION_DECISION_LEVEL,
            "PILOT01",
            {
                "human_trying_to_pass": True,
                "ai_adjacent_to_current_subgoal_target": True,
            },
            "CONTINUE_CURRENT_SUBGOAL",
        ) - model.score(
            COORDINATION_DECISION_LEVEL,
            "PILOT01",
            {
                "human_trying_to_pass": True,
                "ai_adjacent_to_current_subgoal_target": True,
            },
            "YIELD",
        )

        self.assertGreater(task_margin, 0.0)
        self.assertGreater(coordination_margin, 0.0)

    def test_old_single_head_model_loads_as_task_only_hu(self):
        old_model = LinearSubgoalReranker(condition_keys=("pot_empty",))

        hierarchical = HierarchicalHu.from_dict(old_model.to_dict())

        self.assertIsNotNone(hierarchical.task_head)
        self.assertIsNone(hierarchical.coordination_head)
        self.assertEqual(
            hierarchical.score(
                COORDINATION_DECISION_LEVEL,
                "PILOT01",
                {},
                "YIELD",
            ),
            0.0,
        )

    def test_train_validation_split_keeps_each_domain_in_training(self):
        samples = [
            PairwiseSample(
                user_id="PILOT01",
                layout="test",
                condition_features={},
                preferred_subgoal="GET_TOMATO",
                rejected_subgoal="WAIT",
                decision_level=TASK_DECISION_LEVEL,
            ),
            PairwiseSample(
                user_id="PILOT01",
                layout="test",
                condition_features={},
                preferred_subgoal="YIELD",
                rejected_subgoal="CONTINUE_CURRENT_SUBGOAL",
                decision_level=COORDINATION_DECISION_LEVEL,
            ),
        ]

        train, validation = split_samples(samples, validation_fraction=0.5, seed=1)

        self.assertEqual(len(train), 2)
        self.assertEqual(validation, [])


class ReviewedDatasetTests(unittest.TestCase):
    def test_human_review_overrides_automatic_label(self):
        attribution = {
            "feedback_event_id": "feedback:1",
            "target_event": "AI_missed_useful_ingredient_pickup",
            "target_time_window": [8, 10],
            "condition_features": {"pot_empty": True},
            "preferred_subgoals": ["GET_TOMATO"],
            "rejected_subgoals": ["WAIT"],
            "needs_clarification": True,
            "candidate_events": [],
        }
        review = {
            "decision": "revise",
            "approved_event": "AI_blocked_human_path",
            "approved_time_window": [9, 10],
            "approved_condition_overrides": {
                "human_trying_to_pass": True,
                "narrow_corridor": True,
            },
            "approved_preference": {
                "preferred_subgoals": ["YIELD"],
                "rejected_subgoals": ["CONTINUE_CURRENT_SUBGOAL"],
            },
            "use_for_hu_training": True,
        }

        provenance = build_provenance_record(
            attribution=attribution,
            feedback=None,
            trajectory=[],
            user_id="PILOT01",
            review_decision=review,
        )
        samples = build_training_samples([provenance])

        self.assertEqual(provenance["target_event"], "AI_blocked_human_path")
        self.assertEqual(provenance["decision_level"], "coordination")
        self.assertFalse(provenance["needs_clarification"])
        self.assertEqual(len(samples), 1)
        self.assertEqual(samples[0]["preferred_subgoal"], "YIELD")
        self.assertEqual(
            samples[0]["rejected_subgoal"],
            "CONTINUE_CURRENT_SUBGOAL",
        )
        self.assertEqual(samples[0]["label_source"], "human_review")

    def test_reviewed_record_not_approved_for_training_is_excluded(self):
        provenance = {
            "reviewed": True,
            "use_for_hu_training": False,
            "needs_clarification": False,
            "target_event": "AI_blocked_human_path",
            "event_actor": "ai",
            "preferred_subgoals": ["YIELD"],
            "rejected_subgoals": ["CONTINUE_CURRENT_SUBGOAL"],
        }

        self.assertEqual(build_training_samples([provenance]), [])


class CandidateEventTests(unittest.TestCase):
    def test_explicit_coordination_decision_becomes_neutral_event(self):
        step = make_step(10, ai_subgoal="GET_TOMATO")
        step["coordination_decision"] = {
            "decision_id": "coord:e1:t10:n1",
            "decision_level": "coordination",
            "conflict_type": "human_entering_ai_tile",
            "task_subgoal": "GET_TOMATO",
            "condition_at_decision": {
                "human_trying_to_pass": True,
                "ai_adjacent_to_current_subgoal_target": True,
            },
            "candidate_set": ["CONTINUE_CURRENT_SUBGOAL", "YIELD"],
            "candidates": [],
            "selected": "YIELD",
        }

        events = detect_coordination_decision_events([step])

        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["event_type"], "AI_yielded_to_human")
        self.assertEqual(events[0]["event_valence"], "neutral_context")
        self.assertEqual(
            events[0]["evidence"]["decision_id"],
            "coord:e1:t10:n1",
        )

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


class SubgoalPreferenceTests(unittest.TestCase):
    def test_ambiguous_pick_drop_event_does_not_invent_pair(self):
        preferred, rejected = infer_subgoal_preferences(
            target_event="AI_pick_drop_loop",
            observed_subgoal="WAIT",
            event_valence="negative_problem",
        )

        self.assertEqual(preferred, [])
        self.assertEqual(rejected, [])

    def test_negative_event_can_compare_known_alternative_to_observed_subgoal(self):
        preferred, rejected = infer_subgoal_preferences(
            target_event="AI_missed_labor_division_opportunity",
            observed_subgoal="WAIT",
            alternative_subgoals=["GET_DISH"],
            event_valence="missed_opportunity",
        )

        self.assertEqual(preferred, ["GET_DISH"])
        self.assertEqual(rejected, ["WAIT"])

    def test_partial_llm_output_is_not_completed_by_static_defaults(self):
        preferred, rejected = infer_subgoal_preferences(
            target_event="AI_blocked_human_path",
            preferred_subgoals=["YIELD"],
            rejected_subgoals=[],
            observed_subgoal="CONTINUE_CURRENT_SUBGOAL",
            event_valence="negative_problem",
        )

        self.assertEqual(preferred, ["YIELD"])
        self.assertEqual(rejected, [])


if __name__ == "__main__":
    unittest.main()
