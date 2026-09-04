"""Focused tests for Hu condition extraction and candidate event detectors."""

from __future__ import annotations

import unittest

from durf.feedback_attribution.condition_features import (
    decision_condition_features,
    extract_condition_features,
)
from durf.feedback_attribution.event_detectors import (
    detect_ai_failed_to_prepare_ingredient_while_waiting,
    detect_ai_failed_to_yield_or_clear_path,
    detect_ai_held_unneeded_object_too_long,
    detect_ai_ignored_ready_or_nearly_ready_pot,
    detect_ai_missed_labor_division_opportunity,
    detect_ai_missed_plate_pickup_opportunity,
    detect_ai_missed_useful_counter_object,
    detect_ai_pick_drop_loop,
    detect_ai_put_object_on_unhelpful_counter,
    detect_ai_successfully_put_ingredient_into_pot,
    detect_coordination_decision_events,
)
from durf.feedback_attribution.hu_dataset_builder import (
    build_provenance_record,
    build_training_samples,
)
from durf.feedback_attribution.probe_state_detector import (
    DEFAULT_PROBES,
    build_probe_hit,
)
from durf.feedback_attribution.sample_builder import (
    build_preview_attribution,
    candidates_visible_at_feedback,
)
from durf.feedback_attribution.subgoal_preferences import (
    TASK_HU_SUBGOALS as ATTRIBUTION_TASK_HU_SUBGOALS,
    infer_explicit_preference_from_text,
    infer_subgoal_preferences,
    resolve_useful_ingredient,
)
from durf.hu.subgoal_reranker import (
    COORDINATION_DECISION_LEVEL,
    COORDINATION_SUBGOALS,
    TASK_DECISION_LEVEL,
    TASK_HU_SUBGOALS,
    HierarchicalHu,
    LinearSubgoalReranker,
    PairwiseSample,
)
from durf.hu.train_subgoal_reranker import (
    check_participant_protocol,
    dedupe_against_train,
    dedupe_samples,
    evaluate_by_user,
    split_samples,
)


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


def facts(
    *,
    ai_pos: tuple[int, int] = (3, 1),
    human_pos: tuple[int, int] = (1, 3),
    ai_held: str | None = None,
    human_held: str | None = None,
    objects: list[dict] | None = None,
    pot_states: dict | None = None,
) -> dict:
    """State snapshot for detectors that need distinct before/after facts."""

    def held(name: str | None, position: tuple[int, int]) -> dict | None:
        if name is None:
            return None
        return {"name": name, "position": list(position)}

    return {
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


def before_after_step(
    total_step: int,
    *,
    before: dict,
    after: dict,
    ai_action: str = "stay",
    human_action: str = "stay",
    ai_subgoal: str = "WAIT",
) -> dict:
    return {
        "total_step": total_step,
        "ai_action_name": ai_action,
        "human_action_name": human_action,
        "ai_subgoal": ai_subgoal,
        "state_facts": after,
        "extra": {"state_before": before},
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

    def test_decision_condition_uses_the_pre_action_state(self):
        """A label about the decision at step t must carry the condition the
        decision layer saw (state_before), not the post-action snapshot.
        The put-down step is the canonical trap: after the action the AI is
        empty-handed, but PUT_DOWN_OBJECT was chosen while holding a tomato."""
        before = facts(ai_held="tomato", pot_states={"cooking": [[2, 0]]})
        after = facts(ai_held=None, pot_states={"cooking": [[2, 0]]})
        step = before_after_step(7, before=before, after=after, ai_action="interact", ai_subgoal="PUT_DOWN_OBJECT")

        outcome_view = extract_condition_features(step)
        decision_view = decision_condition_features(step)

        self.assertTrue(outcome_view["ai_empty_handed"])
        self.assertFalse(decision_view["ai_empty_handed"])
        self.assertTrue(decision_view["ai_has_tomato"])
        self.assertEqual(decision_view["condition_state_source"], "state_before")

        # The runtime-recorded features are what Hu actually scored with, so
        # they win over anything recomputed here.
        step["ai_condition_features"] = {"ai_adjacent_to_current_subgoal_target": True, "ai_current_subgoal": "PUT_DOWN_OBJECT"}
        decision_view = decision_condition_features(step)
        self.assertTrue(decision_view["ai_adjacent_to_current_subgoal_target"])
        self.assertEqual(decision_view["ai_current_subgoal"], "PUT_DOWN_OBJECT")

        # Legacy sessions without a pre-action snapshot are flagged, not silently shifted.
        legacy = {**step, "extra": {}}
        self.assertEqual(decision_condition_features(legacy)["condition_state_source"], "state_after_fallback")

    def test_unlabelled_subgoals_carry_no_opinion(self):
        """Weights start at exactly zero and only labelled subgoals move, so a
        subgoal Hu has never seen scores 0 and reports no support. Random
        init used to leave every unlabelled subgoal with a permanent ~0.05
        opinion -- enough to win a zero-margin argmax inside the band."""
        model = LinearSubgoalReranker(subgoals=("A", "B", "C"), condition_keys=("pot_empty",), seed=7)
        self.assertEqual(float(abs(model.condition_weights).sum()), 0.0)
        self.assertEqual(model.has_support("C"), False)

        samples = [
            PairwiseSample(user_id="u", layout=None, condition_features={"pot_empty": True},
                           preferred_subgoal="A", rejected_subgoal="B",
                           decision_level=TASK_DECISION_LEVEL)
            for _ in range(4)
        ]
        model.train(samples, epochs=20)
        self.assertTrue(model.has_support("A"))
        self.assertTrue(model.has_support("B"))
        self.assertFalse(model.has_support("C"))
        self.assertEqual(model.score("u", {"pot_empty": True}, "C"), 0.0)
        self.assertGreater(model.score("u", {"pot_empty": True}, "A"), model.score("u", {"pot_empty": True}, "B"))

        reloaded = LinearSubgoalReranker.from_dict(model.to_dict())
        self.assertEqual(reloaded.has_support("C"), False)
        self.assertTrue(reloaded.has_support("A"))

        legacy = model.to_dict(); legacy.pop("subgoal_support")
        self.assertIsNone(LinearSubgoalReranker.from_dict(legacy).has_support("A"))

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

    def test_coordination_single_sided_preferred_completes_rejected_side(self):
        provenance = {
            "reviewed": False,
            "needs_clarification": False,
            "target_event": "AI_maintained_current_subgoal_during_conflict",
            "event_actor": "ai",
            "decision_level": "coordination",
            "preferred_subgoals": ["CONTINUE_CURRENT_SUBGOAL"],
            "rejected_subgoals": [],
            "user_id": "PILOT01",
            "feedback_event_id": "feedback:single",
            "layout": "ring_tomato_onion_10x6_h0_full_task",
            "condition_features": {"ai_adjacent_to_current_subgoal_target": True},
        }

        samples = build_training_samples([provenance])
        self.assertEqual(len(samples), 1)
        self.assertEqual(samples[0]["preferred_subgoal"], "CONTINUE_CURRENT_SUBGOAL")
        self.assertEqual(samples[0]["rejected_subgoal"], "YIELD")

    def test_coordination_single_sided_rejected_completes_preferred_side(self):
        provenance = {
            "reviewed": False,
            "needs_clarification": False,
            "target_event": "AI_successfully_yielded",
            "event_actor": "ai",
            "decision_level": "coordination",
            "preferred_subgoals": [],
            "rejected_subgoals": ["CONTINUE_CURRENT_SUBGOAL"],
            "user_id": "PILOT01",
            "feedback_event_id": "feedback:single-rej",
            "layout": "ring_tomato_onion_10x6_h0_full_task",
            "condition_features": {"human_trying_to_pass": True},
        }

        samples = build_training_samples([provenance])
        self.assertEqual(len(samples), 1)
        self.assertEqual(samples[0]["preferred_subgoal"], "YIELD")
        self.assertEqual(samples[0]["rejected_subgoal"], "CONTINUE_CURRENT_SUBGOAL")

    def test_task_single_sided_labels_still_skipped(self):
        # Task is not a binary domain: one-sided labels remain unpaired and
        # must not be silently completed by complementing.
        provenance = {
            "reviewed": False,
            "needs_clarification": False,
            "target_event": "AI_ignored_ready_or_nearly_ready_pot",
            "event_actor": "ai",
            "decision_level": "task",
            "preferred_subgoals": ["GET_DISH"],
            "rejected_subgoals": [],
            "user_id": "PILOT01",
            "feedback_event_id": "feedback:task-single",
            "layout": "ring_tomato_onion_10x6_h0_full_task",
            "condition_features": {"pot_cooking_or_ready": True},
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
        self.assertEqual(events[0]["event_type"], "AI_successfully_yielded")
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

    def test_put_ingredient_requires_adjacency_and_flags_human_ambiguity(self):
        soup_before = [
            wrapped_object("soup", (2, 0), ingredients=["tomato"])
        ]
        soup_after = [
            wrapped_object("soup", (2, 0), ingredients=["tomato", "tomato"])
        ]

        adjacent = before_after_step(
            1,
            before=facts(ai_pos=(2, 1), ai_held="tomato", objects=soup_before),
            after=facts(ai_pos=(2, 1), ai_held=None, objects=soup_after),
            ai_action="interact",
        )
        events = detect_ai_successfully_put_ingredient_into_pot([adjacent])

        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["confidence"], 0.85)
        self.assertFalse(events[0]["evidence"]["human_acted_same_step"])

        far = before_after_step(
            2,
            before=facts(ai_pos=(4, 2), ai_held="tomato", objects=soup_before),
            after=facts(ai_pos=(4, 2), ai_held=None, objects=soup_after),
            ai_action="interact",
        )
        self.assertEqual(detect_ai_successfully_put_ingredient_into_pot([far]), [])

        ambiguous = before_after_step(
            3,
            before=facts(
                ai_pos=(2, 1),
                ai_held="tomato",
                human_held="onion",
                objects=soup_before,
            ),
            after=facts(
                ai_pos=(2, 1),
                ai_held=None,
                human_held=None,
                objects=soup_after,
            ),
            ai_action="interact",
            human_action="interact",
        )
        events = detect_ai_successfully_put_ingredient_into_pot([ambiguous])

        self.assertEqual(len(events), 1)
        self.assertTrue(events[0]["evidence"]["human_acted_same_step"])
        self.assertEqual(events[0]["confidence"], 0.7)

    def test_held_unneeded_fires_on_sliding_window_not_contiguous_streak(self):
        objects = [
            wrapped_object(
                "soup",
                (2, 0),
                ingredients=["tomato", "tomato"],
                is_cooking=False,
                is_ready=False,
            )
        ]
        steps = []
        for total_step in range(1, 16):
            step = make_step(
                total_step,
                objects=objects,
                pot_states={"2_items": [[2, 0]]},
                ai_held=None,
            )
            if total_step in {1, 5, 9, 12, 15}:
                step["state_facts"]["ai_held_object"] = {
                    "name": "tomato",
                    "position": [3, 1],
                }
            steps.append(step)

        events = detect_ai_held_unneeded_object_too_long(steps)

        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["event_type"], "AI_held_unneeded_object_too_long")
        self.assertEqual(events[0]["evidence"]["duration_steps"], 5)

    def test_ignored_pot_fires_during_cooking_when_ai_is_idle(self):
        steps = [
            make_step(
                total_step,
                ai_pos=(3, 1),
                ai_held=None,
                pot_states={"cooking": [[2, 0]]},
                ai_subgoal="WAIT",
            )
            for total_step in (1, 2, 3)
        ]

        events = detect_ai_ignored_ready_or_nearly_ready_pot(steps)

        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["event_type"], "AI_ignored_ready_or_nearly_ready_pot")
        self.assertEqual(events[0]["evidence"]["detected_pot_states"], ["cooking"])
        self.assertEqual(events[0]["related_subgoal"], "WAIT")

    def test_ignored_pot_does_not_fire_while_ai_preps_during_cooking(self):
        """Fetching the next batch's ingredient IS the productive use of a
        cooking wait -- it is what AI_failed_to_prepare_ingredient_while_waiting
        asks for, so the two detectors must not contradict each other."""
        steps = [
            make_step(
                total_step,
                ai_held=None,
                pot_states={"cooking": [[2, 0]]},
                ai_subgoal="GET_TOMATO",
            )
            for total_step in (1, 2, 3, 4)
        ]
        self.assertEqual(detect_ai_ignored_ready_or_nearly_ready_pot(steps), [])

    def test_ignored_pot_does_not_fire_during_cooking_when_human_has_dish(self):
        steps = [
            make_step(
                total_step,
                ai_held=None,
                human_held="dish",
                pot_states={"cooking": [[2, 0]]},
                ai_subgoal="WAIT",
            )
            for total_step in (1, 2, 3)
        ]
        self.assertEqual(detect_ai_ignored_ready_or_nearly_ready_pot(steps), [])

    def test_ignored_pot_fires_on_ready_pot_unless_ai_is_attending(self):
        ignoring = [
            make_step(t, ai_held=None, pot_states={"ready": [[2, 0]]}, ai_subgoal="GET_TOMATO")
            for t in (1, 2, 3)
        ]
        events = detect_ai_ignored_ready_or_nearly_ready_pot(ignoring)
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["related_subgoal"], "GET_TOMATO")

        for attending in ("GET_DISH", "PICKUP_SOUP", "WAIT_NEAR_POT"):
            steps = [
                make_step(t, ai_held=None, pot_states={"ready": [[2, 0]]}, ai_subgoal=attending)
                for t in (1, 2, 3, 4, 5)
            ]
            self.assertEqual(
                detect_ai_ignored_ready_or_nearly_ready_pot(steps), [], attending
            )

    def test_prep_wait_does_not_fire_while_ai_walks_to_dispenser(self):
        idle = [
            make_step(t, ai_held=None, human_held="dish", pot_states={"cooking": [[2, 0]]}, ai_subgoal="WAIT")
            for t in (1, 2, 3, 4)
        ]
        self.assertEqual(len(detect_ai_failed_to_prepare_ingredient_while_waiting(idle)), 1)

        walking = [
            make_step(t, ai_held=None, human_held="dish", pot_states={"cooking": [[2, 0]]}, ai_subgoal="GET_ONION")
            for t in (1, 2, 3, 4, 5, 6)
        ]
        self.assertEqual(detect_ai_failed_to_prepare_ingredient_while_waiting(walking), [])

    def test_prep_wait_requires_a_fetch_option_on_the_table(self):
        """When the next batch is already staged the runtime offers no
        GET_TOMATO/GET_ONION; demanding one then names a phantom option."""
        nothing_to_fetch = [
            {
                **make_step(t, ai_held=None, human_held="dish", pot_states={"cooking": [[2, 0]]}, ai_subgoal="WAIT"),
                "ai_subgoal_candidates": [{"subgoal": "WAIT"}],
            }
            for t in (1, 2, 3, 4, 5)
        ]
        self.assertEqual(detect_ai_failed_to_prepare_ingredient_while_waiting(nothing_to_fetch), [])

        fetch_available = [
            {**step, "ai_subgoal_candidates": [{"subgoal": "GET_TOMATO"}, {"subgoal": "WAIT"}]}
            for step in nothing_to_fetch
        ]
        self.assertEqual(len(detect_ai_failed_to_prepare_ingredient_while_waiting(fetch_available)), 1)

    def test_missed_plate_fires_after_three_frames(self):
        steps = [
            make_step(
                total_step,
                ai_pos=(3, 1),
                ai_held=None,
                human_held=None,
                pot_states={"cooking": [[2, 0]]},
                ai_subgoal="WAIT",
            )
            for total_step in (1, 2, 3)
        ]

        events = detect_ai_missed_plate_pickup_opportunity(steps)

        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["event_type"], "AI_missed_plate_pickup_opportunity")

    def test_put_on_unhelpful_counter_is_neutral_context(self):
        step = before_after_step(
            1,
            before=facts(ai_pos=(3, 1), ai_held="tomato"),
            after=facts(
                ai_pos=(3, 1),
                ai_held=None,
                objects=[wrapped_object("tomato", (4, 4))],
            ),
            ai_action="interact",
        )

        events = detect_ai_put_object_on_unhelpful_counter([step])

        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["event_valence"], "neutral_context")
        self.assertTrue(events[0]["evidence"]["drop_may_be_staging_or_yield"])

    @staticmethod
    def pick_drop_steps(positions: list[tuple[int, int]]) -> list[dict]:
        """Alternating pickups (None -> tomato) and drops (tomato -> None)."""
        steps = []
        for total, tile in enumerate(positions, start=1):
            if total % 2 == 1:
                before = facts(ai_pos=tile, ai_held=None)
                after = facts(ai_pos=tile, ai_held="tomato")
            else:
                before = facts(ai_pos=tile, ai_held="tomato")
                after = facts(ai_pos=tile, ai_held=None)
            steps.append(
                before_after_step(
                    total,
                    before=before,
                    after=after,
                    ai_action="interact",
                )
            )
        return steps

    def test_pick_drop_loop_fires_only_on_same_tile_repeats(self):
        loop_steps = self.pick_drop_steps([(3, 1)] * 6)

        events = detect_ai_pick_drop_loop(loop_steps)

        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["event_type"], "AI_pick_drop_loop")
        self.assertEqual(events[0]["confidence"], 0.85)
        self.assertEqual(events[0]["evidence"]["loop_tile"], [3, 1])
        self.assertEqual(events[0]["evidence"]["pick_drop_count"], 6)

    def test_pick_drop_moving_between_tiles_is_not_a_loop(self):
        moving_steps = self.pick_drop_steps(
            [(3, 1), (4, 1), (4, 1), (5, 1), (5, 1), (3, 1)]
        )

        events = detect_ai_pick_drop_loop(moving_steps)

        self.assertEqual(events, [])

    def test_duplicate_dish_requires_human_inferred_pickup_soup(self):
        steps = [
            make_step(
                total_step,
                human_held="dish",
                pot_states={"cooking": [[2, 0]]},
                ai_subgoal="GET_DISH",
            )
            for total_step in (1, 2)
        ]

        events = detect_ai_missed_labor_division_opportunity(steps)

        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["evidence"]["opportunity_kind"], "duplicate_dish_task")

    def test_duplicate_dish_not_reported_before_human_commits_to_pot(self):
        steps = [
            make_step(
                total_step,
                human_held="dish",
                pot_states={"empty": [[2, 0]]},
                ai_subgoal="GET_DISH",
            )
            for total_step in (1, 2)
        ]

        events = detect_ai_missed_labor_division_opportunity(steps)

        self.assertEqual(events, [])

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

    def test_useful_ingredient_is_not_a_hu_dimension(self):
        """A label on a name the runtime never offers can never change a
        decision, so it must not be a Hu dimension -- it is resolved instead."""
        self.assertNotIn("GET_USEFUL_INGREDIENT", ATTRIBUTION_TASK_HU_SUBGOALS)
        self.assertNotIn("GET_USEFUL_INGREDIENT", TASK_HU_SUBGOALS)

    def test_useful_ingredient_resolves_to_the_decisions_real_candidates(self):
        candidates = [{"subgoal": "GET_TOMATO"}, {"subgoal": "WAIT"}]
        resolved, unresolved = resolve_useful_ingredient(
            ["GET_USEFUL_INGREDIENT"], candidates
        )
        self.assertEqual(resolved, ["GET_TOMATO"])
        self.assertEqual(unresolved, [])

        # Nothing to fetch was on the table: the preference is not expressible
        # at that decision and must be dropped, not trained on a phantom.
        resolved, unresolved = resolve_useful_ingredient(
            ["GET_USEFUL_INGREDIENT"], [{"subgoal": "WAIT"}]
        )
        self.assertEqual(resolved, [])
        self.assertEqual(unresolved, ["GET_USEFUL_INGREDIENT"])

        # Concrete names pass through untouched.
        resolved, unresolved = resolve_useful_ingredient(["GET_DISH"], [])
        self.assertEqual((resolved, unresolved), (["GET_DISH"], []))

    def test_builder_resolves_useful_ingredient_against_trajectory_candidates(self):
        trajectory = [
            {
                **make_step(5, ai_subgoal="WAIT", human_held="dish", pot_states={"cooking": [[2, 0]]}),
                "ai_subgoal_candidates": [
                    {"subgoal": "GET_ONION", "task_score": 50.0},
                    {"subgoal": "WAIT", "task_score": 0.0},
                ],
            }
        ]
        attribution = {
            "feedback_event_id": "feedback:9",
            "target_event": "AI_failed_to_prepare_ingredient_while_waiting",
            "target_time_window": [5, 8],
            "candidate_events": [
                {
                    "event_type": "AI_failed_to_prepare_ingredient_while_waiting",
                    "start_timestep": 5,
                    "end_timestep": 8,
                    "actor": "ai",
                    "event_valence": "missed_opportunity",
                    "related_subgoal": "WAIT",
                }
            ],
        }
        provenance = build_provenance_record(
            attribution=attribution, feedback=None, trajectory=trajectory, user_id="SIM"
        )
        self.assertEqual(provenance["attributed_preferred_subgoals"], ["GET_USEFUL_INGREDIENT"])
        self.assertEqual(provenance["preferred_subgoals"], ["GET_ONION"])
        self.assertEqual(provenance["rejected_subgoals"], ["WAIT"])
        self.assertEqual(provenance["unresolved_subgoals"], [])
        samples = build_training_samples([provenance])
        self.assertEqual(
            [(s["preferred_subgoal"], s["rejected_subgoal"]) for s in samples],
            [("GET_ONION", "WAIT")],
        )

        # Same feedback, but the runtime had nothing to fetch on the table.
        trajectory[0]["ai_subgoal_candidates"] = [{"subgoal": "WAIT", "task_score": 0.0}]
        provenance = build_provenance_record(
            attribution=attribution, feedback=None, trajectory=trajectory, user_id="SIM"
        )
        self.assertEqual(provenance["preferred_subgoals"], [])
        self.assertEqual(provenance["unresolved_subgoals"], ["GET_USEFUL_INGREDIENT"])
        self.assertEqual(build_training_samples([provenance]), [])

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


def coordination_step(
    *,
    total_step: int,
    selected: str,
    candidates: list[dict],
    conflict_type: str = "human_entering_ai_tile",
    ai_subgoal: str = "GET_TOMATO",
    task_candidates: list[dict] | None = None,
) -> dict:
    return {
        "total_step": total_step,
        "ai_action_name": "stay",
        "human_action_name": "north",
        "ai_subgoal": ai_subgoal,
        "state_facts": {
            "ai_pos": [1, 4],
            "human_pos": [2, 4],
            "ai_held_object": None,
            "human_held_object": None,
            "objects": [],
            "pot_states": {"empty": [[2, 0]]},
            "layout_features": {
                "layout_name": "test",
                "terrain": ["XXXXX", "X   X", "XXXXX"],
            },
        },
        "extra": {"state_before": {"ai_pos": [1, 4], "human_pos": [2, 4]}},
        "ai_subgoal_candidates": task_candidates or [],
        "coordination_decision": {
            "record_type": "runtime_decision",
            "decision_level": "coordination",
            "conflict_type": conflict_type,
            "task_subgoal": ai_subgoal,
            "selected": selected,
            "candidates": candidates,
        },
    }


class ProbeDetectionTests(unittest.TestCase):
    def test_coordination_probe_reads_coordination_candidate_pool(self):
        yield_probe = next(
            probe
            for probe in DEFAULT_PROBES
            if probe.name == "human_path_conflict_ai_should_yield"
        )
        step = coordination_step(
            total_step=10,
            selected="YIELD",
            candidates=[
                {
                    "option": "YIELD",
                    "action": 4,
                    "base_score": 1.0,
                    "hu_score": 0.5,
                    "final_score": 1.5,
                    "reason": "yield_during_human_entering_ai_tile",
                    "feasible": True,
                },
                {
                    "option": "CONTINUE_CURRENT_SUBGOAL",
                    "action": 0,
                    "base_score": 0.0,
                    "hu_score": 0.1,
                    "final_score": 0.1,
                    "reason": "continue_task_during_human_entering_ai_tile",
                    "feasible": True,
                },
            ],
            # The task pool deliberately contains no YIELD; the old buggy path
            # read it and reported the coordination probe as unevaluable.
            task_candidates=[
                {"subgoal": "GET_TOMATO", "final_score": 70.0},
                {"subgoal": "WAIT", "final_score": 0.0},
            ],
        )

        hit = build_probe_hit(step, yield_probe)

        self.assertEqual(hit["chosen_subgoal"], "YIELD")
        self.assertFalse(hit["evaluation"]["evaluation_unavailable"])
        self.assertEqual(hit["evaluation"]["best_preferred_rank"], 1)
        self.assertEqual(hit["evaluation"]["best_rejected_rank"], 2)
        self.assertTrue(hit["evaluation"]["chosen_is_preferred"])
        self.assertEqual(
            [row["subgoal"] for row in hit["candidate_ranking"]],
            ["YIELD", "CONTINUE_CURRENT_SUBGOAL"],
        )

    def test_coordination_probe_without_decision_is_marked_unavailable(self):
        yield_probe = next(
            probe
            for probe in DEFAULT_PROBES
            if probe.name == "human_path_conflict_ai_should_yield"
        )
        step = coordination_step(
            total_step=11,
            selected="YIELD",
            candidates=[],
        )
        step.pop("coordination_decision")

        hit = build_probe_hit(step, yield_probe)

        self.assertTrue(hit["evaluation"]["evaluation_unavailable"])
        self.assertFalse(hit["evaluation"]["preferred_available"])
        self.assertEqual(hit["evaluation"]["best_preferred_rank"], None)

    def test_task_probe_keeps_reading_task_candidate_pool(self):
        task_probe = next(
            probe
            for probe in DEFAULT_PROBES
            if probe.name == "pot_ready_ai_should_get_dish"
        )
        step = make_step(12, ai_subgoal="GET_TOMATO")
        step["ai_subgoal_candidates"] = [
            {"subgoal": "GET_DISH", "final_score": 80.0},
            {"subgoal": "GET_TOMATO", "final_score": 70.0},
            {"subgoal": "WAIT", "final_score": 0.0},
        ]

        hit = build_probe_hit(step, task_probe)

        self.assertFalse(hit["evaluation"]["evaluation_unavailable"])
        self.assertEqual(hit["evaluation"]["best_preferred_rank"], 1)
        self.assertEqual(hit["evaluation"]["best_rejected_rank"], 2)


    def test_detects_detour_requiring_ai_standing(self):
        before = facts(ai_pos=(3, 3), human_pos=(1, 3))
        after = facts(ai_pos=(3, 3), human_pos=(2, 3))
        step = {
            "total_step": 5,
            "ai_action_name": "stay",
            "human_action_name": "east",
            "ai_subgoal": "WAIT",
            "state_facts": after,
            "extra": {"state_before": before},
        }

        events = detect_ai_failed_to_yield_or_clear_path([step])

        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["event_type"], "AI_failed_to_yield_or_clear_path")
        self.assertTrue(events[0]["evidence"]["ai_blocks_next_tile"])
        self.assertEqual(events[0]["evidence"]["human_attempted_target"], [2, 3])

    def test_direct_block_is_not_a_detour_event(self):
        # Human's destination is exactly the AI tile; that is the existing
        # AI_blocked_human_path case, not the detour case.
        before = facts(ai_pos=(2, 3), human_pos=(1, 3))
        after = facts(ai_pos=(2, 3), human_pos=(1, 3))
        step = {
            "total_step": 5,
            "ai_action_name": "stay",
            "human_action_name": "east",
            "ai_subgoal": "WAIT",
            "state_facts": after,
            "extra": {"state_before": before},
        }

        events = detect_ai_failed_to_yield_or_clear_path([step])

        self.assertEqual(events, [])

    def test_detour_event_not_emitted_when_human_moves_away_from_ai(self):
        # Human moves east but the AI is not on the tile after the destination.
        before = facts(ai_pos=(4, 3), human_pos=(1, 3))
        after = facts(ai_pos=(4, 3), human_pos=(2, 3))
        step = {
            "total_step": 5,
            "ai_action_name": "stay",
            "human_action_name": "east",
            "ai_subgoal": "WAIT",
            "state_facts": after,
            "extra": {"state_before": before},
        }

        events = detect_ai_failed_to_yield_or_clear_path([step])

        self.assertEqual(events, [])

    def test_probe_vocabulary_aligned_with_runtime_candidate_pools(self):
        # Runtime-constructible task candidates today (collect_rule_teacher_dataset
        # SUBGOALS plus PUT_DOWN_OBJECT from the holding-unneeded branch).
        runtime_task_candidates = {
            "GET_TOMATO",
            "PUT_TOMATO_IN_POT",
            "GET_ONION",
            "PUT_ONION_IN_POT",
            "GET_DISH",
            "PICKUP_SOUP",
            "SERVE_SOUP",
            "WAIT",
            "PUT_DOWN_OBJECT",
        }
        for probe in DEFAULT_PROBES:
            for subgoal in (*probe.preferred_subgoals, *probe.rejected_subgoals):
                self.assertIn(
                    subgoal,
                    runtime_task_candidates
                    if probe.domain == "task"
                    else set(COORDINATION_SUBGOALS),
                    f"{probe.name} references non-runtime subgoal {subgoal}",
                )

    def test_ready_pot_probe_prefers_only_empty_hand_reachable_dish(self):
        pot_probe = next(
            probe
            for probe in DEFAULT_PROBES
            if probe.name == "pot_ready_ai_should_get_dish"
        )
        # ai_empty_handed=True implies PICKUP_SOUP is unreachable (the AI has
        # no dish yet), so it must not be a preferred label.
        self.assertNotIn("PICKUP_SOUP", pot_probe.preferred_subgoals)
        self.assertIn("GET_DISH", pot_probe.preferred_subgoals)

    def test_probe_rejected_within_runtime_vocabulary(self):
        # held-unneeded probes must not reject GET_TOMATO/GET_ONION, which the
        # runtime pool cannot contain while the AI is holding the object.
        onion_probe = next(
            probe
            for probe in DEFAULT_PROBES
            if probe.name == "ai_holding_unneeded_onion_should_put_down"
        )
        self.assertNotIn("GET_ONION", onion_probe.rejected_subgoals)
        self.assertNotIn("GET_TOMATO", onion_probe.rejected_subgoals)


class TrainSplitTests(unittest.TestCase):
    def _sample(
        self,
        feedback_id: str | None,
        preferred: str = "GET_DISH",
        rejected: str = "WAIT",
        user_id: str = "PILOT01",
    ) -> PairwiseSample:
        return PairwiseSample(
            user_id=user_id,
            layout="ring_tomato_onion_test",
            condition_features={"pot_cooking_or_ready": True},
            preferred_subgoal=preferred,
            rejected_subgoal=rejected,
            decision_level=TASK_DECISION_LEVEL,
            source_feedback_id=feedback_id,
        )

    def test_dedupe_against_train_skips_reused_feedback(self):
        train = [
            self._sample("chat_messages.csv:49"),
            self._sample("chat_messages.csv:50"),
        ]
        test = [
            self._sample("chat_messages.csv:49", preferred="GET_TOMATO"),
            self._sample("chat_messages.csv:99"),
        ]

        result = dedupe_against_train(test, train)

        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].source_feedback_id, "chat_messages.csv:99")

    def test_dedupe_keeps_all_when_no_train_ids(self):
        train = [
            self._sample(None),
        ]
        test = [
            self._sample("chat_messages.csv:49"),
            self._sample("chat_messages.csv:50"),
        ]

        result = dedupe_against_train(test, train)

        self.assertEqual(len(result), 2)

    def test_dedupe_samples_keeps_sibling_pairs_from_same_feedback(self):
        # One feedback expands into several pairs; all are legitimate.
        samples = [
            self._sample(
                "chat_messages.csv:49:2026-07-22T13:18:02.836+00:00",
                preferred="PUT_TOMATO_IN_POT",
                rejected="WAIT",
            ),
            self._sample(
                "chat_messages.csv:49:2026-07-22T13:18:02.836+00:00",
                preferred="PUT_TOMATO_IN_POT",
                rejected="GET_TOMATO",
            ),
        ]

        result = dedupe_samples(samples)

        self.assertEqual(len(result), 2)

    def test_dedupe_samples_drops_exact_duplicate_pair(self):
        samples = [
            self._sample("chat_messages.csv:49", preferred="GET_DISH"),
            self._sample("chat_messages.csv:49", preferred="GET_DISH"),
            self._sample("chat_messages.csv:50", preferred="GET_DISH"),
        ]

        result = dedupe_samples(samples)

        self.assertEqual(len(result), 2)

    def test_protocol_rejects_unseen_test_user(self):
        train = [
            self._sample("chat_messages.csv:1", user_id="PILOT01"),
        ]
        test = [
            self._sample("chat_messages.csv:9", user_id="PILOT01"),
            self._sample("chat_messages.csv:10", user_id="PILOT99"),
        ]

        with self.assertRaises(ValueError) as ctx:
            check_participant_protocol(train, test)

        self.assertIn("PILOT99", str(ctx.exception))

    def test_protocol_warns_when_test_predates_train(self):
        train = [
            self._sample(
                "chat_messages.csv:5:2026-07-22T14:00:00+00:00",
                user_id="PILOT01",
            ),
        ]
        test = [
            self._sample(
                "chat_messages.csv:2:2026-07-22T10:00:00+00:00",
                user_id="PILOT01",
            ),
        ]

        checks = check_participant_protocol(train, test)

        self.assertTrue(checks["protocol_ok"])
        self.assertEqual(len(checks["temporal_warnings"]), 1)

    def test_evaluate_by_user_reports_each_user(self):
        train = [
            self._sample("chat_messages.csv:1", user_id="PILOT01"),
            self._sample("chat_messages.csv:1", user_id="PILOT02"),
        ]
        test = [
            self._sample("chat_messages.csv:9", user_id="PILOT01"),
            self._sample("chat_messages.csv:10", user_id="PILOT02"),
        ]
        model = HierarchicalHu.from_samples(train)

        metrics = evaluate_by_user(model, test)

        self.assertEqual(sorted(metrics), ["PILOT01", "PILOT02"])
        self.assertEqual(
            metrics["PILOT01"]["task"]["samples"],
            1,
        )
        self.assertEqual(
            metrics["PILOT02"]["task"]["samples"],
            1,
        )


if __name__ == "__main__":
    unittest.main()


class SimTaskPreferenceTests(unittest.TestCase):
    """The sim persona's task-domain PREFERENCE feedback (docs/hu_feedback_data_flow.md S6).

    The corrective templates produce nothing on this backbone, so the only way
    a sim participant can express a task preference is at decision points
    where two candidates are genuinely close. These tests pin the three parts
    that must agree: when an opportunity exists, what the persona says, and
    what that sentence is parsed into.
    """

    def test_opportunity_requires_two_close_feasible_candidates(self):
        from durf.group_a.sim_session import _preference_opportunity

        cooking_dish_vs_prep = [
            {"subgoal": "GET_DISH", "task_score": 60.0},
            {"subgoal": "GET_TOMATO", "task_score": 50.0},
            {"subgoal": "WAIT", "task_score": 0.0},
        ]
        self.assertEqual(
            _preference_opportunity(cooking_dish_vs_prep), "division_of_labour"
        )

        # Soup is ready: the dish is 20 points ahead, no longer a toss-up.
        ready = [
            {"subgoal": "GET_DISH", "task_score": 90.0},
            {"subgoal": "GET_TOMATO", "task_score": 70.0},
        ]
        self.assertIsNone(_preference_opportunity(ready))

        no_pot_cooking = [
            {"subgoal": "PUT_DOWN_OBJECT", "task_score": 20.0},
            {"subgoal": "WAIT_NEAR_POT", "task_score": 10.0},
            {"subgoal": "WAIT", "task_score": 5.0},
        ]
        self.assertEqual(_preference_opportunity(no_pot_cooking), "hold_plate")

        # Holding an unneeded ingredient: dropping it is 50 points better, so
        # "hold on to it" would be a task regression, not a preference.
        unneeded = [
            {"subgoal": "PUT_DOWN_OBJECT", "task_score": 60.0},
            {"subgoal": "WAIT_NEAR_POT", "task_score": 10.0},
        ]
        self.assertIsNone(_preference_opportunity(unneeded))
        self.assertIsNone(_preference_opportunity([{"subgoal": "WAIT", "task_score": 0.0}]))

    def test_every_template_parses_to_the_pair_it_is_meant_to_express(self):
        from durf.group_a import sim_session as ss

        expected = [
            (ss._TASK_PREF_PREP_FIRST, ["GET_USEFUL_INGREDIENT"], ["GET_DISH"]),
            (ss._TASK_PREF_DISH_FIRST, ["GET_DISH"], ["GET_USEFUL_INGREDIENT"]),
            (ss._TASK_PREF_HOLD_PLATE, ["WAIT_NEAR_POT"], ["PUT_DOWN_OBJECT"]),
        ]
        for templates, preferred, rejected in expected:
            for text in templates:
                got_preferred, got_rejected, source = infer_explicit_preference_from_text(text)
                self.assertEqual((got_preferred, got_rejected), (preferred, rejected), text)
                self.assertEqual(source, "explicit_text_fallback", text)

    def test_personas_disagree_on_the_division_of_labour_and_repeat_is_capped(self):
        from durf.group_a import sim_session as ss

        candidates = [
            {"subgoal": "GET_DISH", "task_score": 60.0},
            {"subgoal": "GET_TOMATO", "task_score": 50.0},
        ]
        said = {}
        for persona in ("cooperative", "polite", "selfish", "lenient"):
            human = ss.SimHuman(ss.SimHumanConfig(persona=persona, seed=1))
            text = human._voice_task_preference(candidates)
            self.assertIsNotNone(text)
            preferred, rejected, _ = infer_explicit_preference_from_text(text)
            said[persona] = (preferred[0], rejected[0])
        self.assertEqual(said["cooperative"], ("GET_USEFUL_INGREDIENT", "GET_DISH"))
        self.assertEqual(said["polite"], ("GET_USEFUL_INGREDIENT", "GET_DISH"))
        self.assertEqual(said["selfish"], ("GET_DISH", "GET_USEFUL_INGREDIENT"))
        self.assertEqual(said["lenient"], ("GET_DISH", "GET_USEFUL_INGREDIENT"))

        # A standing preference is stated a couple of times per round, not on
        # every one of the ~200 steps the opportunity is open.
        human = ss.SimHuman(ss.SimHumanConfig(persona="cooperative", seed=1))
        voiced = [human._voice_task_preference(candidates) for _ in range(6)]
        self.assertEqual(sum(text is not None for text in voiced), ss.PREFERENCE_MAX_PER_EPISODE)
        human.reset_episode()
        self.assertIsNotNone(human._voice_task_preference(candidates))

    def test_explicit_task_preference_is_not_filed_as_the_nearby_coordination_event(self):
        """Conflicts are frequent; a task preference typed while one was being
        resolved must not become a YIELD/CONTINUE label."""
        from durf.feedback_attribution.schemas import candidate_event

        trajectory = [
            {
                **make_step(20, ai_subgoal="GET_DISH", pot_states={"cooking": [[2, 0]]}),
                "ai_subgoal_candidates": [
                    {"subgoal": "GET_DISH", "task_score": 60.0},
                    {"subgoal": "GET_TOMATO", "task_score": 50.0},
                    {"subgoal": "WAIT", "task_score": 0.0},
                ],
            }
        ]
        coordination_event = candidate_event(
            event_type="AI_successfully_yielded",
            start_timestep=20,
            end_timestep=20,
            evidence={"decision_level": "coordination"},
            severity=0.2,
            confidence=0.9,
            related_subgoal="YIELD",
        )
        feedback = {
            "record_type": "feedback_event",
            "total_step": 20,
            "episode": 1,
            "episode_step": 20,
            "feedback_text": "i'll get the plate, you start the next ingredient",
            "feedback_value": None,
        }

        attribution = build_preview_attribution(
            feedback=feedback,
            trajectory=trajectory,
            candidate_events=[coordination_event],
        )

        self.assertEqual(attribution["preferred_subgoals"], ["GET_USEFUL_INGREDIENT"])
        self.assertEqual(attribution["rejected_subgoals"], ["GET_DISH"])
        self.assertEqual(attribution["decision_level"], "task")
        self.assertEqual(attribution["preference_source"], "explicit_text_fallback")
        self.assertIsNone(attribution["target_event"])
        # The keyword selector did not match this sentence to the conflict at
        # all, so there was nothing to displace.
        self.assertIsNone(attribution["preference_overridden_event"])

        # ... and the pair that reaches training is resolved against the real
        # candidate set at that decision, so it names a runtime option.
        provenance = build_provenance_record(
            attribution=attribution, feedback=feedback, trajectory=trajectory, user_id="SIM"
        )
        samples = build_training_samples([provenance])
        self.assertEqual(
            [(s["preferred_subgoal"], s["rejected_subgoal"], s["decision_level"]) for s in samples],
            [("GET_TOMATO", "GET_DISH", "task")],
        )
        self.assertFalse(provenance["condition_features"]["ai_empty_handed"] is None)

    def test_a_two_sided_statement_outranks_a_keyword_matched_event(self):
        """Event selection is keyword matching. When it lands on an event whose
        default names only one side, or on the wrong decision domain, the
        sentence -- which names both sides explicitly -- is the better label."""
        from durf.feedback_attribution.schemas import candidate_event

        trajectory = [
            {
                **make_step(20, ai_subgoal="GET_TOMATO", pot_states={"cooking": [[2, 0]]}),
                "ai_subgoal_candidates": [
                    {"subgoal": "GET_DISH", "task_score": 60.0},
                    {"subgoal": "GET_ONION", "task_score": 50.0},
                ],
            }
        ]

        def attribute(text, event):
            return build_preview_attribution(
                feedback={
                    "record_type": "feedback_event",
                    "total_step": 20,
                    "episode": 1,
                    "episode_step": 20,
                    "feedback_text": text,
                    "feedback_value": None,
                },
                trajectory=trajectory,
                candidate_events=[event],
            )

        # (a) The event agrees on the domain AND names both sides, using the
        # subgoal the AI was actually running -- more specific than the
        # sentence's generic "ingredient". It is kept.
        same_domain = candidate_event(
            event_type="AI_missed_plate_pickup_opportunity",
            start_timestep=20,
            end_timestep=20,
            evidence={},
            severity=0.3,
            confidence=0.7,
            event_valence="missed_opportunity",
            related_subgoal="GET_TOMATO",
        )
        result = attribute("grab the dish now, i'll handle the next ingredient", same_domain)
        self.assertEqual(result["preferred_subgoals"], ["GET_DISH"])
        self.assertEqual(result["rejected_subgoals"], ["GET_TOMATO"])
        self.assertIsNone(result["preference_overridden_event"])

        # (b) different domain: a complete coordination pair vs a task sentence.
        conflict = candidate_event(
            event_type="AI_blocked_human_path",
            start_timestep=20,
            end_timestep=20,
            evidence={},
            severity=0.5,
            confidence=0.8,
            event_valence="negative_problem",
            related_subgoal="CONTINUE_CURRENT_SUBGOAL",
        )
        result = attribute(
            "you keep blocking me. anyway, you get the plate, i'll take care of the ingredients",
            conflict,
        )
        self.assertEqual(result["decision_level"], "task")
        self.assertEqual(result["preferred_subgoals"], ["GET_DISH"])
        self.assertEqual(result["preference_overridden_event"], "AI_blocked_human_path")

        # (c) event and sentence agree on domain and the event is complete:
        # the event stays the label, nothing is displaced.
        result = attribute("step aside, i can't pass", conflict)
        self.assertEqual(result["target_event"], "AI_blocked_human_path")
        self.assertIsNone(result["preference_overridden_event"])
