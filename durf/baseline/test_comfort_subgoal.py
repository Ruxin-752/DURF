"""Path A consistency tests: execution reuse + comfort agent wiring.

These run headless on the real ring layout (overcooked + motion planner only).
The central guarantee is that the newly factored ``execute_subgoal`` reproduces
the frozen ``rule_teacher_decision`` execution exactly, so the comfort agent
only changes subgoal *selection*, never H0's motion behavior.
"""

from __future__ import annotations

import json
import unittest
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from overcooked_ai_py.mdp.actions import Action
from overcooked_ai_py.mdp.overcooked_mdp import (
    ObjectState,
    OvercookedState,
    PlayerState,
    SoupState,
)

from durf.baseline import evaluate_comfort_multiseed
from durf.baseline.comfort_reward import (
    _facing_target_resource,
    _pot_snapshot,
    context_from_state,
)
from durf.baseline.comfort_subgoal_agent import (
    DEFAULT_MODEL_PATH,
    HUMAN_FEEDBACK_MODEL_PATH,
    PAPER_CV_ENSEMBLE_PATH,
    STAY,
    ComfortSubgoalAgent,
    load_comfort_weights,
)
from durf.baseline.h0_planner import (
    execute_subgoal,
    make_motion_planner,
    rule_teacher_decision,
    yield_path_action,
)
from durf.baseline.runtime import RING_TOMATO_ONION_H0_LAYOUT, make_direct_multi_env

from src.feature_schema import load_features
from src.subgoal_featurizer import SubgoalContext, featurize_subgoal
from src.subgoal_planner import enumerate_feasible_subgoals, plan_subgoal

LAYOUT = RING_TOMATO_ONION_H0_LAYOUT
HORIZON = 120


def _states_from_rollout(seed: int = 42, steps: int = HORIZON):
    """Yield real (state, motion_planner) pairs by driving two rule agents."""

    motion_planner = make_motion_planner(LAYOUT, seed, steps)
    env = make_direct_multi_env(LAYOUT, seed, horizon=steps)
    env.multi_reset()
    for _ in range(steps):
        state = env.base_env.state
        yield state, motion_planner
        _, a0 = rule_teacher_decision(state, motion_planner, player_index=0)
        _, a1 = rule_teacher_decision(state, motion_planner, player_index=1)
        (_, _), _, done, _ = env.multi_step(int(a0), int(a1))
        if done:
            break


class ExecuteSubgoalConsistencyTests(unittest.TestCase):
    def test_matches_rule_teacher_on_its_own_subgoal(self) -> None:
        """execute_subgoal == rule_teacher_decision action for the same subgoal."""

        checked = 0
        for state, motion_planner in _states_from_rollout():
            for player_index in (0, 1):
                subgoal, expected_action = rule_teacher_decision(
                    state, motion_planner, player_index=player_index
                )
                got = execute_subgoal(
                    state, motion_planner, subgoal, player_index=player_index
                )
                self.assertEqual(
                    int(got),
                    int(expected_action),
                    msg=f"subgoal={subgoal} player={player_index}",
                )
                checked += 1
        self.assertGreater(checked, 0)

    def test_wait_and_unknown_resolve_to_stay(self) -> None:
        state, motion_planner = next(iter(_states_from_rollout(steps=1)))
        self.assertEqual(execute_subgoal(state, motion_planner, "WAIT", 0), 4)

    def test_yield_path_requires_explicit_facing_and_one_step_escape(self) -> None:
        ai = SimpleNamespace(position=(1, 3), orientation=(0, 1))
        human = SimpleNamespace(position=(1, 4), orientation=(0, -1))
        state = SimpleNamespace(players=[ai, human])
        mdp = SimpleNamespace(
            get_valid_player_positions=lambda: [(1, 2), (1, 3), (1, 4)]
        )
        self.assertEqual(yield_path_action(state, mdp, player_index=0), 0)

        human.orientation = (1, 0)
        self.assertIsNone(yield_path_action(state, mdp, player_index=0))
        human.orientation = (0, -1)
        blocked_mdp = SimpleNamespace(
            get_valid_player_positions=lambda: [(1, 3), (1, 4)]
        )
        self.assertIsNone(yield_path_action(state, blocked_mdp, player_index=0))


class ComfortAgentWiringTests(unittest.TestCase):
    def test_latest_session_corridor_block_is_reward_ranked_yield(self) -> None:
        """Step-185 geometry: WAIT blocks; one north step clears the path."""

        motion_planner = make_motion_planner(LAYOUT, 42, 800)
        held_tomato = ObjectState("tomato", (1, 4))
        state = OvercookedState(
            players=[
                PlayerState((1, 3), (0, 1)),
                PlayerState((1, 4), (0, -1), held_tomato),
            ],
            objects={
                (7, 0): ObjectState("onion", (7, 0)),
                (4, 0): SoupState.get_soup((4, 0), num_onions=1),
            },
        )
        context = context_from_state(state, motion_planner.mdp, ai_index=0)
        self.assertEqual(context.candidate_path_effects["WAIT"], "blocks")
        self.assertEqual(context.candidate_path_effects["YIELD_PATH"], "clears")
        yield_phi = featurize_subgoal(context, "YIELD_PATH")
        self.assertTrue(
            {
                "clears_human_path",
                "clears_human_shortest_path",
                "avoids_human_shortest_path",
                "respects_human_intent",
                "time_cost",
                "distance_cost",
            }.issubset(yield_phi)
        )
        self.assertFalse(
            {"pick_tomato", "pick_onion", "pick_dish", "supports_serving"}
            & set(yield_phi)
        )

        weights = {feature: 0.0 for feature in load_features()}
        weights.update(
            {
                "clears_human_path": 1.0,
                "blocks_human_path": -1.0,
                "human_wait_cost": -1.0,
                "frustrates_human": -1.0,
            }
        )
        agent = ComfortSubgoalAgent(
            motion_planner,
            weights=weights,
            feedback_mode="frozen",
            max_consecutive_wait=0,
        )
        action, subgoal = agent.act(state)
        self.assertEqual(subgoal, "YIELD_PATH")
        self.assertEqual(action, 0)  # north, away from the adjacent human
        successor, _ = motion_planner.mdp.get_state_transition(
            state,
            (Action.INDEX_TO_ACTION[action], (0, -1)),
        )
        self.assertEqual(successor.player_positions, ((1, 2), (1, 3)))
        self.assertEqual(agent.last_decision["score_formula"], "w_dot_phi")
        self.assertEqual(agent.last_decision["decision_source"], "reward_argmax")
        self.assertFalse(agent.last_decision["policy_overrides_enabled"])

    def test_empty_human_heading_uses_only_clear_straight_line_target(self) -> None:
        player = SimpleNamespace(
            position=(1, 1), orientation=(1, 0), held_object=None
        )
        mdp = SimpleNamespace(
            get_tomato_dispenser_locations=lambda: [(4, 1)],
            get_onion_dispenser_locations=lambda: [],
            get_dish_dispenser_locations=lambda: [],
            get_serving_locations=lambda: [],
            get_pot_states=lambda state: {},
            get_ready_pots=lambda pot_states: [],
            get_valid_player_positions=lambda: [(1, 1), (2, 1), (3, 1)],
        )
        self.assertEqual(
            _facing_target_resource(SimpleNamespace(), mdp, player), "tomato"
        )
        player.orientation = (0, -1)
        self.assertIsNone(
            _facing_target_resource(SimpleNamespace(), mdp, player)
        )

    def test_pot_snapshot_prefers_partial_pot_accepting_held_ingredient(self) -> None:
        def soup(*ingredients):
            return SimpleNamespace(
                name="soup",
                ingredients=list(ingredients),
            )

        objects = {
            (1, 0): soup("tomato", "tomato", "onion"),
            (4, 0): soup("onion"),
        }
        state = SimpleNamespace(
            has_object=lambda pos: pos in objects,
            get_object=lambda pos: objects[pos],
        )
        mdp = SimpleNamespace(
            start_all_orders=[["tomato", "tomato", "onion"]],
            get_pot_states=lambda state: {},
            get_ready_pots=lambda pot_states: [(1, 0)],
            get_cooking_pots=lambda pot_states: [],
            get_pot_locations=lambda: [(1, 0), (4, 0)],
        )
        self.assertEqual(
            _pot_snapshot(state, mdp, preferred_ingredient="tomato"),
            (["onion"], "partial"),
        )

    def test_live_context_keeps_ready_and_partial_pots(self) -> None:
        def soup(*ingredients):
            return SimpleNamespace(name="soup", ingredients=list(ingredients))

        objects = {
            (1, 0): soup("tomato", "tomato", "onion"),
            (4, 0): soup("onion"),
        }
        state = SimpleNamespace(
            players=[
                SimpleNamespace(
                    position=(2, 2),
                    orientation=(1, 0),
                    held_object=SimpleNamespace(name="tomato"),
                ),
                SimpleNamespace(
                    position=(3, 2), orientation=(0, -1), held_object=None
                ),
            ],
            has_object=lambda pos: pos in objects,
            get_object=lambda pos: objects[pos],
        )
        mdp = SimpleNamespace(
            start_all_orders=[["tomato", "tomato", "onion"]],
            get_pot_states=lambda state: {},
            get_ready_pots=lambda pot_states: [(1, 0)],
            get_cooking_pots=lambda pot_states: [],
            get_pot_locations=lambda: [(1, 0), (4, 0)],
            get_tomato_dispenser_locations=lambda: [],
            get_onion_dispenser_locations=lambda: [],
            get_dish_dispenser_locations=lambda: [],
            get_serving_locations=lambda: [],
            get_valid_player_positions=lambda: [(2, 2), (3, 2)],
        )
        context = context_from_state(state, mdp, ai_index=0)
        self.assertEqual(len(context.pot_snapshots), 2)
        self.assertTrue(context.soup_ready)
        self.assertTrue(context.has_open_recipe_work)
        self.assertEqual(
            set(enumerate_feasible_subgoals(context)),
            {"PUT_TOMATO_IN_POT", "WAIT"},
        )

    def test_live_context_counts_only_objects_on_non_feature_counters(self) -> None:
        soup = SimpleNamespace(
            name="soup", ingredients=["tomato", "tomato", "onion"]
        )
        objects = {
            (1, 0): soup,
            (2, 0): SimpleNamespace(name="onion"),
            (3, 0): SimpleNamespace(name="dish"),
            # A deliberately loose test MDP reports this dispenser as a
            # counter too; fixed features must still never count as staging.
            (4, 0): SimpleNamespace(name="tomato"),
        }
        state = SimpleNamespace(
            players=[
                SimpleNamespace(
                    position=(2, 2), orientation=(1, 0), held_object=None
                ),
                SimpleNamespace(
                    position=(3, 2), orientation=(0, -1), held_object=None
                ),
            ],
            objects=objects,
            has_object=lambda pos: pos in objects,
            get_object=lambda pos: objects[pos],
        )
        mdp = SimpleNamespace(
            start_all_orders=[["tomato", "tomato", "onion"]],
            get_pot_states=lambda state: {},
            get_ready_pots=lambda pot_states: [],
            get_cooking_pots=lambda pot_states: [(1, 0)],
            get_pot_locations=lambda: [(1, 0)],
            get_counter_locations=lambda: [(2, 0), (3, 0), (4, 0)],
            get_tomato_dispenser_locations=lambda: [(4, 0)],
            get_onion_dispenser_locations=lambda: [],
            get_dish_dispenser_locations=lambda: [],
            get_serving_locations=lambda: [],
            get_valid_player_positions=lambda: [(2, 2), (3, 2)],
        )

        with patch(
            "durf.baseline.comfort_reward._candidate_geometry",
            return_value=({}, {}),
        ):
            context = context_from_state(state, mdp, ai_index=0)

        self.assertEqual(context.staged_inventory, {"onion": 1, "dish": 1})
        self.assertEqual(
            enumerate_feasible_subgoals(context),
            ["GET_TOMATO", "WAIT"],
        )

    def test_open_second_pot_denies_passive_cooking_wait_exemption(self) -> None:
        context = SubgoalContext(
            recipe=["tomato", "tomato", "onion"],
            pot_ingredients=["tomato", "tomato", "onion"],
            pot_status="cooking",
            pot_snapshots=[
                {
                    "ingredients": ["tomato", "tomato", "onion"],
                    "status": "cooking",
                },
                {"ingredients": ["onion"], "status": "partial"},
            ],
        )
        fake_agent = SimpleNamespace(
            _agent_blocks_human_target_access=lambda state, context: False
        )
        self.assertFalse(
            ComfortSubgoalAgent._passive_cooking_wait_is_valid(
                fake_agent, SimpleNamespace(), context, "WAIT"
            )
        )

    def test_route2_defaults_to_neutral_prior_and_best_available_model(self) -> None:
        agent = ComfortSubgoalAgent(
            make_motion_planner(LAYOUT, 42, HORIZON),
            feedback_mode="route2",
        )
        self.assertEqual(agent.model_path, DEFAULT_MODEL_PATH)
        if PAPER_CV_ENSEMBLE_PATH.exists():
            self.assertEqual(
                PAPER_CV_ENSEMBLE_PATH.parent.name,
                "paper_aligned_v5_seed137_selected",
            )
            self.assertEqual(agent.model_path, PAPER_CV_ENSEMBLE_PATH)
        else:
            self.assertEqual(agent.model_path, HUMAN_FEEDBACK_MODEL_PATH)
        self.assertEqual(set(agent.weights), set(load_features()))
        self.assertTrue(all(value == 0.0 for value in agent.weights.values()))

    def test_act_returns_valid_action_and_feasible_subgoal(self) -> None:
        agent = ComfortSubgoalAgent(
            make_motion_planner(LAYOUT, 42, HORIZON), lambda_pref=1.0, ai_index=0
        )
        for state, _ in _states_from_rollout(steps=40):
            action, subgoal = agent.act(state)
            self.assertIn(int(action), range(6))
            context = context_from_state(state, agent.mdp, ai_index=0)
            self.assertIn(subgoal, enumerate_feasible_subgoals(context))

    def test_chosen_subgoal_matches_plan_subgoal(self) -> None:
        weights = load_comfort_weights()
        motion_planner = make_motion_planner(LAYOUT, 42, HORIZON)
        agent = ComfortSubgoalAgent(
            motion_planner,
            weights=weights,
            lambda_pref=1.0,
            ai_index=0,
            subgoal_commitment_steps=0,
            switch_margin=0.0,
        )
        for state, _ in _states_from_rollout(steps=40):
            _, subgoal = agent.act(state)
            decision = agent.last_decision
            expected = plan_subgoal(
                weights,
                decision["context"],
                lambda_pref=1.0,
                feasible_subgoals=decision["feasible_subgoals"],
                tie_fallback=decision["h0_tie_fallback"],
            )["chosen_subgoal"]
            self.assertEqual(subgoal, expected)
            self.assertEqual(decision["score_formula"], "w_dot_phi")

    def test_latest_session_step24_clears_block_and_fetches_second_tomato(self) -> None:
        """Regression for the observed 44-step WAIT at the single pot access."""

        actions = [
            (3, 4), (3, 4), (1, 4), (1, 4), (1, 3), (2, 3),
            (2, 3), (2, 3), (1, 3), (5, 4), (2, 1), (2, 1),
            (2, 1), (2, 0), (0, 1), (0, 3), (0, 5), (3, 0),
            (3, 0), (3, 0), (3, 2), (0, 2), (5, 4),
        ]
        make_motion_planner(LAYOUT, 42, 800)  # registers/validates the layout
        env = make_direct_multi_env(LAYOUT, 42, horizon=800)
        try:
            env.multi_reset()
            for ai_action, human_action in actions:
                env.multi_step(ai_action, human_action)
            state = env.base_env.state
            context = context_from_state(state, env.base_env.mdp, ai_index=0)
            self.assertEqual([p.position for p in state.players], [(4, 1), (3, 1)])
            self.assertEqual(
                [getattr(p.held_object, "name", None) for p in state.players],
                [None, "tomato"],
            )
            self.assertEqual(context.pot_ingredients, ["onion"])
            self.assertEqual(context.candidate_path_effects["WAIT"], "blocks")
            self.assertEqual(context.candidate_path_effects["GET_TOMATO"], "clears")

            decision = plan_subgoal(
                load_comfort_weights(),
                context,
                tie_fallback="GET_TOMATO",
            )
            ranking = {row["subgoal"]: row for row in decision["ranking"]}
            self.assertEqual(decision["chosen_subgoal"], "GET_TOMATO")
            self.assertGreater(
                ranking["GET_TOMATO"]["total_score"],
                ranking["WAIT"]["total_score"],
            )
            self.assertNotIn(
                "duplicate_human_task", ranking["GET_TOMATO"]["features"]
            )
            self.assertIn("blocks_human_path", ranking["WAIT"]["features"])
            self.assertNotIn("clears_human_path", ranking["WAIT"]["features"])
        finally:
            env.close()

    def test_latest_session_human_final_ingredient_exposes_dish_work(self) -> None:
        """Regression for the step-29..48 only-WAIT candidate collapse."""

        motion_planner = make_motion_planner(LAYOUT, 42, 800)
        held_tomato = ObjectState("tomato", (4, 1))
        state = OvercookedState(
            players=[
                PlayerState((6, 1), (-1, 0)),
                PlayerState((4, 1), (0, -1), held_tomato),
            ],
            objects={
                (4, 0): SoupState.get_soup(
                    (4, 0), num_tomatoes=1, num_onions=1
                )
            },
        )
        context = context_from_state(state, motion_planner.mdp, ai_index=0)
        self.assertEqual(context.open_missing_ingredients, ["tomato"])
        self.assertEqual(context.human_holding, "tomato")
        self.assertEqual(
            enumerate_feasible_subgoals(context), ["GET_DISH", "WAIT"]
        )

        weights = {feature: 0.0 for feature in load_features()}
        weights["pick_dish"] = 1.0
        agent = ComfortSubgoalAgent(
            motion_planner,
            weights=weights,
            feedback_mode="frozen",
            ai_index=0,
        )
        decision = agent.plan(state)
        self.assertEqual(decision["preconstraint_feasible_subgoals"], ["GET_DISH", "WAIT"])
        self.assertEqual(decision["feasible_subgoals"], ["GET_DISH", "WAIT"])
        self.assertEqual(decision["chosen_subgoal"], "GET_DISH")
        self.assertNotEqual(decision["candidate_next_actions"]["GET_DISH"], STAY)
        self.assertEqual(decision["score_formula"], "w_dot_phi")

    def test_route1_live_feedback_updates_posterior_and_returns_trace(self) -> None:
        state, motion_planner = next(iter(_states_from_rollout(steps=1)))
        agent = ComfortSubgoalAgent(
            motion_planner,
            feedback_mode="route1-literal",
            route1_prior="zero",
            lambda_pref=1.0,
            ai_index=0,
        )
        agent.plan(state)
        trace = agent.update_from_feedback(
            "You are blocking me.",
            oracle_feedback={
                "expected_feedback_type": "descriptive",
                "target_features": {"blocks_human_path": 1},
                "attributed_sentiment_score": -1.0,
            },
            interpretation="oracle",
        )

        self.assertEqual(trace["status"], "updated")
        self.assertEqual(trace["mode"], "route1-literal")
        self.assertEqual(trace["feedback_type"], "descriptive")
        self.assertLess(agent.weights["blocks_human_path"], 0.0)
        self.assertIn("before_subgoal", trace)
        self.assertIn("after_subgoal", trace)
        learned = dict(agent.weights)
        agent.reset_context()
        self.assertEqual(agent.weights, learned)
        self.assertIsNone(agent.last_decision)
        self.assertEqual(agent.trajectory_history, [])

    def test_real_inferred_feedback_trace_matches_reward_driven_decision(self) -> None:
        """The trace reports whatever decision follows from updated weights."""

        state, motion_planner = next(iter(_states_from_rollout(steps=1)))
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint = Path(tmp) / "route1_state.json"
            agent = ComfortSubgoalAgent(
                motion_planner,
                feedback_mode="route1-literal",
                route1_prior="zero",
                human_feedback_precision=4.0,
                learner_state_path=checkpoint,
                lambda_pref=1.0,
                ai_index=0,
            )
            before = agent.plan(state)
            self.assertEqual(before["chosen_subgoal"], "GET_TOMATO")

            trace = agent.update_from_feedback(
                "Go ahead and pick up the onion, that's perfect.",
                source="human_live",
                interpretation="inferred",
                feedback_form_prediction={
                    "feedback_type": "mixed",
                    "phrases": [
                        {
                            "text": "Go ahead and pick up the onion",
                            "feedback_type": "imperative",
                            "confidence": 0.99,
                            "probabilities": {"imperative": 0.99},
                            "classifier": "test_ui_feedback_form",
                            "confidence_threshold": 0.55,
                            "abstained": False,
                        },
                        {
                            "text": "that's perfect",
                            "feedback_type": "evaluative",
                            "confidence": 0.99,
                            "probabilities": {"evaluative": 0.99},
                            "classifier": "test_ui_feedback_form",
                            "confidence_threshold": 0.55,
                            "abstained": False,
                        },
                    ],
                },
            )
            after = agent.plan(state)

            self.assertEqual(trace["status"], "updated", msg=trace)
            self.assertEqual(trace["source"], "human_live")
            self.assertEqual(trace["source_precision_multiplier"], 4.0)
            self.assertTrue(trace["reference_types"])
            self.assertEqual(trace["after_subgoal"], after["chosen_subgoal"])
            expected_switch = before["chosen_subgoal"] != after["chosen_subgoal"]
            self.assertEqual(trace["policy_switch"], expected_switch)
            self.assertEqual(
                trace["accepted_but_no_policy_switch"],
                not expected_switch,
            )
            self.assertTrue(checkpoint.exists())

            resumed = ComfortSubgoalAgent(
                motion_planner,
                feedback_mode="route1-literal",
                route1_prior="zero",
                human_feedback_precision=4.0,
                learner_state_path=checkpoint,
                resume_learner_state=True,
                lambda_pref=1.0,
                ai_index=0,
            )
            self.assertEqual(
                resumed.plan(state)["chosen_subgoal"],
                after["chosen_subgoal"],
            )

    def test_route2_live_feedback_persists_and_resumes_weights(self) -> None:
        state, motion_planner = next(iter(_states_from_rollout(steps=1)))
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint = Path(tmp) / "route2_state.json"
            agent = ComfortSubgoalAgent(
                motion_planner,
                feedback_mode="route2",
                learner_state_path=checkpoint,
                ai_index=0,
            )
            agent.plan(state)
            before = dict(agent.weights)
            trace = agent.update_from_feedback(
                "Keep moving and prepare an onion for the next round.",
                feedback_event_id="live-event-resume-1",
            )

            self.assertEqual(trace["status"], "updated")
            self.assertEqual(trace["mode"], "route2")
            self.assertTrue(checkpoint.exists())
            saved_state = json.loads(checkpoint.read_text(encoding="utf-8"))
            self.assertEqual(saved_state["version"], 3)
            self.assertEqual(saved_state["score_formula"], "w_dot_phi")
            self.assertEqual(len(saved_state["model_sha256"]), 64)
            self.assertEqual(
                saved_state["route2_posterior"]["update_rule"],
                "paper_independent_gaussian",
            )
            self.assertEqual(
                saved_state["route2_posterior"]["processed_feedback_event_ids"],
                ["live-event-resume-1"],
            )
            self.assertGreater(
                min(saved_state["route2_posterior"]["precision"].values()),
                1.0 / 25.0,
            )
            self.assertIn("checkpoint_sha256", trace)
            self.assertNotEqual(agent.weights, before)

            resumed = ComfortSubgoalAgent(
                motion_planner,
                feedback_mode="route2",
                learner_state_path=checkpoint,
                resume_learner_state=True,
                ai_index=0,
            )
            self.assertEqual(resumed.weights, agent.weights)
            before_precision = dict(resumed._route2_precision)
            duplicate = resumed.update_from_feedback(
                "Keep moving and prepare an onion for the next round.",
                feedback_event_id="live-event-resume-1",
            )
            self.assertEqual(duplicate["status"], "ignored")
            self.assertEqual(
                duplicate["trace_state"], "ignored_duplicate_feedback_event"
            )
            self.assertEqual(resumed._route2_precision, before_precision)

    def test_route2_resume_rejects_observation_precision_mismatch(self) -> None:
        motion_planner = make_motion_planner(LAYOUT, 42, HORIZON)
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint = Path(tmp) / "route2_precision_state.json"
            ComfortSubgoalAgent(
                motion_planner,
                feedback_mode="route2",
                route2_observation_precision=2.0,
                learner_state_path=checkpoint,
            ).save_learner_state()

            with self.assertRaisesRegex(ValueError, "observation precision"):
                ComfortSubgoalAgent(
                    motion_planner,
                    feedback_mode="route2",
                    route2_observation_precision=3.0,
                    learner_state_path=checkpoint,
                    resume_learner_state=True,
                )

    def test_route2_resume_rejects_gaussian_ema_or_ema_alpha_mismatch(self) -> None:
        motion_planner = make_motion_planner(LAYOUT, 42, HORIZON)
        with tempfile.TemporaryDirectory() as tmp:
            gaussian_checkpoint = Path(tmp) / "route2_gaussian_state.json"
            ComfortSubgoalAgent(
                motion_planner,
                feedback_mode="route2",
                learner_state_path=gaussian_checkpoint,
            ).save_learner_state()
            with self.assertRaisesRegex(ValueError, "update rule"):
                ComfortSubgoalAgent(
                    motion_planner,
                    feedback_mode="route2",
                    online_blend=0.25,
                    learner_state_path=gaussian_checkpoint,
                    resume_learner_state=True,
                )

            ema_checkpoint = Path(tmp) / "route2_ema_state.json"
            ComfortSubgoalAgent(
                motion_planner,
                feedback_mode="route2",
                online_blend=0.25,
                learner_state_path=ema_checkpoint,
            ).save_learner_state()
            with self.assertRaisesRegex(ValueError, "update rule"):
                ComfortSubgoalAgent(
                    motion_planner,
                    feedback_mode="route2",
                    learner_state_path=ema_checkpoint,
                    resume_learner_state=True,
                )
            with self.assertRaisesRegex(ValueError, "online blend"):
                ComfortSubgoalAgent(
                    motion_planner,
                    feedback_mode="route2",
                    online_blend=0.50,
                    learner_state_path=ema_checkpoint,
                    resume_learner_state=True,
                )

    def test_route2_resume_rejects_old_state_missing_update_identity(self) -> None:
        motion_planner = make_motion_planner(LAYOUT, 42, HORIZON)
        state = ComfortSubgoalAgent(
            motion_planner,
            feedback_mode="route2",
        ).learner_state_dict()
        with tempfile.TemporaryDirectory() as tmp:
            for field in ("observation_precision", "update_rule", "online_blend"):
                with self.subTest(field=field):
                    legacy = json.loads(json.dumps(state))
                    legacy["version"] = 2
                    del legacy["route2_posterior"][field]
                    checkpoint = Path(tmp) / f"route2_missing_{field}.json"
                    checkpoint.write_text(json.dumps(legacy), encoding="utf-8")
                    with self.assertRaisesRegex(
                        ValueError, "lacks exact update-configuration identity"
                    ):
                        ComfortSubgoalAgent(
                            motion_planner,
                            feedback_mode="route2",
                            learner_state_path=checkpoint,
                            resume_learner_state=True,
                        )

    def test_route2_resume_rejects_different_base_weights(self) -> None:
        motion_planner = make_motion_planner(LAYOUT, 42, HORIZON)
        features = load_features()
        custom_base = {feature: 0.25 for feature in features}
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint = Path(tmp) / "route2_custom_base_state.json"
            ComfortSubgoalAgent(
                motion_planner,
                weights=custom_base,
                feedback_mode="route2",
                learner_state_path=checkpoint,
            ).save_learner_state()

            with self.assertRaisesRegex(ValueError, "base weights"):
                ComfortSubgoalAgent(
                    motion_planner,
                    feedback_mode="route2",
                    learner_state_path=checkpoint,
                    resume_learner_state=True,
                )

    def test_route2_resume_rejects_nonfinite_mean_or_precision(self) -> None:
        motion_planner = make_motion_planner(LAYOUT, 42, HORIZON)
        state = ComfortSubgoalAgent(
            motion_planner,
            feedback_mode="route2",
        ).learner_state_dict()
        feature = load_features()[0]
        corruptions = (
            ("mean_nan", "mean", float("nan"), "non-finite posterior mean"),
            ("mean_inf", "mean", float("inf"), "non-finite posterior mean"),
            ("precision_nan", "precision", float("nan"), "invalid posterior precision"),
            ("precision_inf", "precision", float("inf"), "invalid posterior precision"),
        )
        with tempfile.TemporaryDirectory() as tmp:
            for name, field, value, error in corruptions:
                with self.subTest(corruption=name):
                    corrupted = json.loads(json.dumps(state))
                    corrupted["route2_posterior"][field][feature] = value
                    checkpoint = Path(tmp) / f"route2_{name}.json"
                    checkpoint.write_text(json.dumps(corrupted), encoding="utf-8")
                    with self.assertRaisesRegex(ValueError, error):
                        ComfortSubgoalAgent(
                            motion_planner,
                            feedback_mode="route2",
                            learner_state_path=checkpoint,
                            resume_learner_state=True,
                        )

    def test_route2_resume_rejects_missing_or_empty_precision(self) -> None:
        motion_planner = make_motion_planner(LAYOUT, 42, HORIZON)
        state = ComfortSubgoalAgent(
            motion_planner,
            feedback_mode="route2",
        ).learner_state_dict()
        with tempfile.TemporaryDirectory() as tmp:
            for name, value in (("missing", None), ("empty", {})):
                with self.subTest(precision=name):
                    corrupted = json.loads(json.dumps(state))
                    if value is None:
                        del corrupted["route2_posterior"]["precision"]
                    else:
                        corrupted["route2_posterior"]["precision"] = value
                    checkpoint = Path(tmp) / f"route2_precision_{name}.json"
                    checkpoint.write_text(json.dumps(corrupted), encoding="utf-8")
                    with self.assertRaisesRegex(
                        ValueError, "invalid posterior precision"
                    ):
                        ComfortSubgoalAgent(
                            motion_planner,
                            feedback_mode="route2",
                            learner_state_path=checkpoint,
                            resume_learner_state=True,
                        )

    def test_route2_rejected_resume_is_atomic(self) -> None:
        motion_planner = make_motion_planner(LAYOUT, 42, HORIZON)
        agent = ComfortSubgoalAgent(motion_planner, feedback_mode="route2")
        checkpoint_state = agent.learner_state_dict()
        features = load_features()
        checkpoint_state["route2_posterior"]["mean"] = {
            feature: 0.125 for feature in features
        }
        checkpoint_state["route2_posterior"]["precision"] = {
            feature: 0.5 for feature in features
        }
        # This fails only after the candidate mean and precision have both been
        # parsed, exercising the no-partial-commit guarantee.
        checkpoint_state["route2_posterior"]["processed_feedback_event_ids"] = [
            123
        ]

        agent.weights = {feature: -0.25 for feature in features}
        agent._route2_precision = {feature: 0.75 for feature in features}
        agent._processed_feedback_event_ids = {"keep-processed"}
        agent._seen_feedback_event_ids = {"keep-processed", "keep-seen"}
        before_weights = dict(agent.weights)
        before_precision = dict(agent._route2_precision)
        before_processed = set(agent._processed_feedback_event_ids)
        before_seen = set(agent._seen_feedback_event_ids)

        with tempfile.TemporaryDirectory() as tmp:
            checkpoint = Path(tmp) / "route2_atomic_rejection.json"
            checkpoint.write_text(json.dumps(checkpoint_state), encoding="utf-8")
            with self.assertRaisesRegex(
                ValueError, "invalid processed feedback event IDs"
            ):
                agent.load_learner_state(checkpoint)

        self.assertEqual(agent.weights, before_weights)
        self.assertEqual(agent._route2_precision, before_precision)
        self.assertEqual(agent._processed_feedback_event_ids, before_processed)
        self.assertEqual(agent._seen_feedback_event_ids, before_seen)

    def test_route2_reference_gate_rejects_other_abstention_and_low_confidence(self) -> None:
        cases = (
            (
                {
                    "phrase": "Hello there",
                    "reference_type": "other",
                    "confidence": 0.91,
                    "threshold": 0.40,
                    "abstained": False,
                    "probabilities": {"other": 0.91},
                },
                "ignored",
                "reference_type_other",
            ),
            (
                {
                    "phrase": "Maybe something",
                    "reference_type": "feature",
                    "confidence": 0.80,
                    "threshold": 0.40,
                    "abstained": True,
                    "probabilities": {"feature": 0.80},
                },
                "rejected_low_confidence",
                "classifier_abstained",
            ),
            (
                {
                    "phrase": "Uncertain preference",
                    "reference_type": "feature",
                    "confidence": 0.30,
                    "threshold": 0.50,
                    "abstained": False,
                    "probabilities": {"feature": 0.30},
                },
                "rejected_low_confidence",
                "below_class_confidence_threshold",
            ),
        )
        for prediction, expected_status, expected_reason in cases:
            with self.subTest(reason=expected_reason):
                agent = ComfortSubgoalAgent(
                    SimpleNamespace(mdp=object()),
                    feedback_mode="route2",
                    require_reference_gate=True,
                )
                before_weights = dict(agent.weights)
                before_precision = dict(agent._route2_precision)
                with (
                    patch(
                        "src.phrase_reference_classifier.classify_utterance",
                        return_value=[prediction],
                    ),
                    patch.object(
                        agent,
                        "_ensure_route2",
                        side_effect=AssertionError("neural predictor must remain gated"),
                    ),
                ):
                    trace = agent.update_from_feedback(prediction["phrase"])

                self.assertEqual(trace["status"], expected_status)
                self.assertEqual(trace["rejection_reason"], expected_reason)
                self.assertFalse(trace["posterior_changed"])
                self.assertEqual(agent.weights, before_weights)
                self.assertEqual(agent._route2_precision, before_precision)
                audited = trace["reference_gate"]["predictions"][0]
                self.assertEqual(audited["reference_type"], prediction["reference_type"])
                self.assertEqual(audited["confidence"], prediction["confidence"])
                self.assertEqual(audited["threshold"], prediction["threshold"])
                self.assertIsNone(
                    trace["reference_gate"]["human_language_accuracy_claim"]
                )

    def test_route2_default_reference_classifier_is_audit_only(self) -> None:
        features = load_features()
        reward_prediction = {feature: 0.25 for feature in features}
        cases = (
            {
                "phrase": "Hello there",
                "reference_type": "other",
                "confidence": 0.91,
                "threshold": 0.40,
                "abstained": False,
                "probabilities": {"other": 0.91},
            },
            {
                "phrase": "Uncertain game feedback",
                "reference_type": "feature",
                "confidence": 0.30,
                "threshold": 0.50,
                "abstained": True,
                "probabilities": {"feature": 0.30},
            },
        )
        for index, phrase_prediction in enumerate(cases):
            with self.subTest(reference_type=phrase_prediction["reference_type"]):
                agent = ComfortSubgoalAgent(
                    SimpleNamespace(mdp=object()), feedback_mode="route2"
                )
                original_text = phrase_prediction["phrase"] + "."
                with (
                    patch(
                        "src.phrase_reference_classifier.classify_utterance",
                        return_value=[phrase_prediction],
                    ),
                    patch.object(
                        agent,
                        "_ensure_route2",
                        return_value=(object(), {"<unk>": 0}, features, False),
                    ),
                    patch(
                        "src.neural_inference.predict_reward_vector",
                        return_value=reward_prediction,
                    ) as neural_prediction,
                ):
                    trace = agent.update_from_feedback(
                        original_text, feedback_event_id=f"audit-only-{index}"
                    )

                self.assertEqual(trace["status"], "updated")
                self.assertEqual(trace["model_input_text"], original_text)
                self.assertFalse(trace["reference_gate"]["accepted"])
                self.assertFalse(trace["reference_gate"]["enforced"])
                self.assertEqual(
                    trace["reference_gate"]["decision"], "audit_only"
                )
                self.assertIsNone(
                    trace["reference_gate"]["human_language_accuracy_claim"]
                )
                self.assertEqual(neural_prediction.call_args.args[3], original_text)
                self.assertTrue(
                    all(
                        abs(value - 2.04) < 1e-12
                        for value in agent._route2_precision.values()
                    )
                )

    def test_route2_event_id_is_idempotent_but_repeated_text_with_new_id_updates(self) -> None:
        features = load_features()
        prediction = {feature: 1.0 for feature in features}
        accepted_reference = {
            "phrase": "Stop blocking me",
            "reference_type": "feature",
            "confidence": 0.90,
            "threshold": 0.50,
            "abstained": False,
            "probabilities": {"feature": 0.90},
            "top2_margin": 0.80,
            "margin_threshold": 0.05,
        }
        agent = ComfortSubgoalAgent(
            SimpleNamespace(mdp=object()),
            feedback_mode="route2",
        )
        with (
            patch(
                "src.phrase_reference_classifier.classify_utterance",
                return_value=[accepted_reference],
            ) as classifier,
            patch.object(
                agent,
                "_ensure_route2",
                return_value=(object(), {"<unk>": 0}, features, False),
            ),
            patch(
                "src.neural_inference.predict_reward_vector",
                return_value=prediction,
            ) as neural_prediction,
        ):
            first = agent.update_from_feedback(
                "Stop blocking me.", feedback_event_id="event-1"
            )
            precision_after_first = dict(agent._route2_precision)
            duplicate = agent.update_from_feedback(
                "Stop blocking me.", feedback_event_id="event-1"
            )
            precision_after_duplicate = dict(agent._route2_precision)
            repeated = agent.update_from_feedback(
                "Stop blocking me.", feedback_event_id="event-2"
            )

        self.assertEqual(first["status"], "updated")
        self.assertEqual(first["observation_precision"], 2.0)
        self.assertTrue(first["reference_gate"]["accepted"])
        self.assertEqual(duplicate["status"], "ignored")
        self.assertEqual(
            duplicate["rejection_reason"], "duplicate_feedback_event_id"
        )
        self.assertEqual(precision_after_duplicate, precision_after_first)
        self.assertEqual(repeated["status"], "updated")
        self.assertTrue(
            all(abs(value - 4.04) < 1e-12 for value in agent._route2_precision.values())
        )
        self.assertEqual(classifier.call_count, 2)
        self.assertEqual(neural_prediction.call_count, 2)

    def test_route2_phrase_gate_filters_fragments_and_uses_grounded_text(self) -> None:
        features = load_features()
        reward_prediction = {feature: 0.5 for feature in features}
        phrase_predictions = [
            {
                "phrase": "You should keep moving",
                "reference_type": "action_behavioral",
                "confidence": 0.88,
                "threshold": 0.49,
                "abstained": False,
                "probabilities": {"action_behavioral": 0.88},
            },
            {
                "phrase": "come next to me",
                "reference_type": "other",
                "confidence": 0.75,
                "threshold": 0.41,
                "abstained": False,
                "probabilities": {"other": 0.75},
            },
            {
                "phrase": "avoid blocking my path",
                "reference_type": "feature",
                "confidence": 0.84,
                "threshold": 0.38,
                "abstained": False,
                "probabilities": {"feature": 0.84},
            },
            {
                "phrase": "perhaps over there",
                "reference_type": "action_spatial",
                "confidence": 0.20,
                "threshold": 0.42,
                "abstained": True,
                "probabilities": {"action_spatial": 0.20},
            },
        ]
        agent = ComfortSubgoalAgent(
            SimpleNamespace(mdp=object()),
            feedback_mode="route2",
            require_reference_gate=True,
        )
        with (
            patch(
                "src.phrase_reference_classifier.classify_utterance",
                return_value=phrase_predictions,
            ),
            patch.object(
                agent,
                "_ensure_route2",
                return_value=(object(), {"<unk>": 0}, features, False),
            ),
            patch(
                "src.neural_inference.predict_reward_vector",
                return_value=reward_prediction,
            ) as neural_prediction,
        ):
            trace = agent.update_from_feedback(
                "You should keep moving; come next to me; avoid blocking my path; "
                "perhaps over there.",
                feedback_event_id="mixed-phrase-event",
            )

        grounded_text = "You should keep moving avoid blocking my path"
        self.assertEqual(trace["status"], "updated")
        self.assertEqual(trace["model_input_text"], grounded_text)
        self.assertEqual(
            trace["reference_gate"]["accepted_phrases"],
            ["You should keep moving", "avoid blocking my path"],
        )
        self.assertEqual(
            [
                row["gate_accepted"]
                for row in trace["reference_gate"]["predictions"]
            ],
            [True, False, True, False],
        )
        self.assertEqual(
            neural_prediction.call_args.args[3],
            grounded_text,
        )
        self.assertTrue(
            all(abs(value - 2.04) < 1e-12 for value in agent._route2_precision.values())
        )

    def test_route2_rejects_legacy_state_without_model_identity(self) -> None:
        motion_planner = make_motion_planner(LAYOUT, 42, HORIZON)
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint = Path(tmp) / "legacy_route2_state.json"
            checkpoint.write_text(
                json.dumps(
                    {
                        "version": 1,
                        "feedback_mode": "route2",
                        "weights": {feature: 0.0 for feature in load_features()},
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "predates paper-aligned"):
                ComfortSubgoalAgent(
                    motion_planner,
                    feedback_mode="route2",
                    learner_state_path=checkpoint,
                    resume_learner_state=True,
                )

    def test_route2_language_changes_only_weights_then_switches_prefetch(self) -> None:
        features = load_features()
        context = {
            "recipe": ["tomato", "tomato", "onion"],
            "pot_ingredients": ["tomato", "tomato", "onion"],
            "pot_status": "cooking",
            "agent_holding": None,
            "human_holding": None,
            "human_intent": None,
        }
        feasible = ["GET_DISH", "GET_TOMATO", "GET_ONION", "WAIT"]
        before_weights = {feature: 0.0 for feature in features}
        before_weights.update(
            {"pick_dish": -1.0, "pick_tomato": -1.0, "pick_onion": -1.0}
        )
        before = plan_subgoal(
            before_weights,
            context,
            feasible_subgoals=feasible,
            tie_fallback="WAIT",
        )
        self.assertEqual(before["chosen_subgoal"], "WAIT")
        before_phi = {
            subgoal: featurize_subgoal(context, subgoal) for subgoal in feasible
        }

        agent = ComfortSubgoalAgent(
            SimpleNamespace(mdp=object()),
            weights=before_weights,
            feedback_mode="route2",
            online_blend=1.0,
        )
        before["context"] = context
        before["h0_tie_fallback"] = "WAIT"
        agent.last_decision = before
        prediction = {feature: 0.0 for feature in features}
        prediction.update(
            {"pick_dish": -1.0, "pick_tomato": -1.0, "pick_onion": 2.0}
        )

        with (
            patch(
                "src.phrase_reference_classifier.classify_utterance",
                return_value=[
                    {
                        "phrase": "prepare an onion for the next round",
                        "reference_type": "feature",
                        "confidence": 0.90,
                        "threshold": 0.50,
                        "abstained": False,
                        "probabilities": {"feature": 0.90},
                    }
                ],
            ),
            patch.object(
                agent,
                "_ensure_route2",
                return_value=(object(), {"<unk>": 0}, features, True),
            ),
            patch("src.neural_inference.predict_reward_vector", return_value=prediction),
        ):
            trace = agent.update_from_feedback(
                "While the soup cooks, prepare an onion for the next round."
            )

        after = plan_subgoal(
            agent.weights,
            context,
            feasible_subgoals=feasible,
            tie_fallback="WAIT",
        )
        after_phi = {
            subgoal: featurize_subgoal(context, subgoal) for subgoal in feasible
        }
        self.assertEqual(before_phi, after_phi)
        self.assertEqual(trace["before_subgoal"], "WAIT")
        self.assertEqual(trace["after_subgoal"], "GET_ONION")
        self.assertTrue(trace["policy_switch"])
        self.assertEqual(after["chosen_subgoal"], "GET_ONION")
        self.assertEqual(after["decision_source"], "reward_argmax")
        self.assertEqual(after["score_formula"], "w_dot_phi")

    def test_route2_defaults_to_paper_gaussian_reward_update(self) -> None:
        features = load_features()
        prediction = {feature: 1.0 for feature in features}
        agent = ComfortSubgoalAgent(
            SimpleNamespace(mdp=object()),
            feedback_mode="route2",
        )
        with (
            patch.object(
                agent,
                "_ensure_route2",
                return_value=(object(), {"<unk>": 0}, features, False),
            ),
            patch("src.neural_inference.predict_reward_vector", return_value=prediction),
        ):
            trace = agent.update_from_feedback("That reward profile is right.")

        expected = 2.0 / (2.0 + 1.0 / 25.0)
        self.assertEqual(trace["update_rule"], "paper_independent_gaussian")
        self.assertIsNone(trace["blend"])
        self.assertEqual(trace["observation_precision"], 2.0)
        self.assertEqual(
            trace["observation_precision_policy"], "paper_fixed_precision"
        )
        self.assertAlmostEqual(agent.weights[features[0]], expected)
        self.assertAlmostEqual(agent._route2_precision[features[0]], 2.04)

    def test_route2_human_precision_calibration_is_explicit_opt_in(self) -> None:
        features = load_features()
        prediction = {feature: 1.0 for feature in features}
        agent = ComfortSubgoalAgent(
            SimpleNamespace(mdp=object()),
            feedback_mode="route2",
            route2_human_observation_precision=0.02,
        )
        with (
            patch.object(
                agent,
                "_ensure_route2",
                return_value=(object(), {"<unk>": 0}, features, False),
            ),
            patch("src.neural_inference.predict_reward_vector", return_value=prediction),
        ):
            live_trace = agent.update_from_feedback(
                "Please do not block me.", source="human_live"
            )

        expected = 0.02 / (0.02 + 1.0 / 25.0)
        self.assertEqual(live_trace["update_rule"], "paper_independent_gaussian")
        self.assertEqual(live_trace["observation_precision"], 0.02)
        self.assertEqual(
            live_trace["observation_precision_policy"],
            "explicit_human_calibration",
        )
        self.assertAlmostEqual(agent.weights[features[0]], expected)
        self.assertAlmostEqual(agent._route2_precision[features[0]], 0.06)

    def test_frozen_mode_records_but_does_not_change_weights(self) -> None:
        state, motion_planner = next(iter(_states_from_rollout(steps=1)))
        agent = ComfortSubgoalAgent(
            motion_planner,
            feedback_mode="frozen",
            ai_index=0,
        )
        agent.plan(state)
        before = dict(agent.weights)
        trace = agent.update_from_feedback("Stop blocking me.")

        self.assertEqual(trace["status"], "ignored")
        self.assertEqual(agent.weights, before)

    def test_evaluative_feedback_uses_recent_decision_window(self) -> None:
        state, motion_planner = next(iter(_states_from_rollout(steps=1)))
        agent = ComfortSubgoalAgent(
            motion_planner,
            feedback_mode="route1-literal",
            route1_lookback=25,
            ai_index=0,
        )
        agent.plan(state)
        agent.record_transition(
            state_before={
                "ai_held_object": None,
                "human_held_object": None,
                "ai_pos": [1, 1],
                "human_pos": [2, 1],
            },
            state_after={
                "ai_held_object": {"name": "onion"},
                "human_held_object": None,
                "ai_pos": [1, 1],
                "human_pos": [2, 1],
                "pot_states": {},
            },
            ai_action_name="interact",
            human_action_name="stay",
            environment_reward=0.0,
        )
        with patch(
            "src.feedback_observations.extract_sentiment",
            return_value={"sentiment": "positive", "sentiment_score": 1.0},
        ):
            trace = agent.update_from_feedback("Good job.")

        self.assertEqual(trace["status"], "updated")
        self.assertEqual(trace["feedback_type"], "evaluative")
        self.assertEqual(trace["trajectory_window"], 1)
        self.assertEqual(trace["trajectory_source"], "executed_steps")
        self.assertEqual(
            trace["observations"][0]["grounding_source"],
            "trajectory_features",
        )
        self.assertEqual(
            trace["observations"][0]["temporal_credit_mode"],
            "trajectory_window",
        )
        self.assertEqual(trace["recent_event_count"], 1)
        self.assertIn(
            "ingredient_onion",
            trace["observations"][0]["target_features"],
        )

    def test_behavioral_feedback_uses_repeated_live_events(self) -> None:
        state, motion_planner = next(iter(_states_from_rollout(steps=1)))
        agent = ComfortSubgoalAgent(
            motion_planner,
            feedback_mode="route1-literal",
            route1_lookback=25,
            ai_index=0,
        )
        agent.plan(state)
        for _ in range(2):
            agent.record_transition(
                state_before={
                    "ai_held_object": None,
                    "human_held_object": None,
                    "ai_pos": [2, 1],
                    "human_pos": [1, 1],
                },
                state_after={
                    "ai_held_object": None,
                    "human_held_object": None,
                    "ai_pos": [2, 1],
                    "human_pos": [1, 1],
                    "pot_states": {},
                },
                ai_action_name="stay",
                human_action_name="east",
                environment_reward=0.0,
            )
        prediction = {
            "phrase": "You keep blocking my way.",
            "reference_type": "action_behavioral",
            "confidence": 0.99,
            "probabilities": {"action_behavioral": 0.99},
            "classifier": "test",
            "abstained": False,
            "top2_margin": 0.98,
        }
        with (
            patch(
                "src.feedback_observations.predict_reference_type",
                return_value=prediction,
            ),
            patch(
                "src.feedback_observations.modified_vader_observation",
                return_value=-1.0,
            ),
        ):
            trace = agent.update_from_feedback(
                "You keep blocking my way.",
                feedback_form_prediction={
                    "feedback_type": "evaluative",
                    "confidence": 0.99,
                    "probabilities": {"evaluative": 0.99},
                    "classifier": "test_ui_feedback_form",
                    "confidence_threshold": 0.55,
                    "abstained": False,
                },
            )

        self.assertEqual(trace["status"], "updated")
        self.assertEqual(trace["recent_event_count"], 2)
        observation = trace["observations"][0]
        self.assertEqual(observation["temporal_credit_mode"], "behavior_repetition")
        self.assertEqual(observation["target_event_steps"], [1, 2])
        self.assertIn("blocks_human_path", observation["target_features"])

    def test_explicit_empty_feasible_set_does_not_reintroduce_wait(self) -> None:
        context = {
            "recipe": ["tomato", "tomato", "onion"],
            "pot_ingredients": [],
            "pot_status": "empty",
        }
        with self.assertRaisesRegex(ValueError, "at least one feasible subgoal"):
            plan_subgoal({}, context, feasible_subgoals=[])

    def test_commitment_and_switch_margin_prevent_oscillation(self) -> None:
        agent = ComfortSubgoalAgent.__new__(ComfortSubgoalAgent)
        agent.committed_subgoal = "GET_TOMATO"
        agent.commitment_age = 1
        agent.subgoal_commitment_steps = 2
        agent.switch_margin = 0.2
        decision = {
            "chosen_subgoal": "GET_ONION",
            "feasible_subgoals": ["GET_TOMATO", "GET_ONION"],
            "ranking": [
                {"subgoal": "GET_ONION", "total_score": 1.1},
                {"subgoal": "GET_TOMATO", "total_score": 1.0},
            ],
        }
        held = agent._stabilize_decision(decision)
        self.assertEqual(held["chosen_subgoal"], "GET_TOMATO")
        self.assertEqual(held["stability_reason"], "subgoal_commitment")

        agent.commitment_age = 2
        held = agent._stabilize_decision(
            {
                "chosen_subgoal": "GET_ONION",
                "feasible_subgoals": ["GET_TOMATO", "GET_ONION"],
                "ranking": [
                    {"subgoal": "GET_ONION", "total_score": 1.1},
                    {"subgoal": "GET_TOMATO", "total_score": 1.0},
                ],
            }
        )
        self.assertEqual(held["chosen_subgoal"], "GET_TOMATO")
        self.assertEqual(held["stability_reason"], "switch_hysteresis")

    def test_intentional_wait_is_not_overridden_by_deadlock_guards(self) -> None:
        motion_planner = SimpleNamespace(mdp=object())
        features = load_features()
        agent = ComfortSubgoalAgent(
            motion_planner,
            weights={feature: 0.0 for feature in features},
            feedback_mode="frozen",
            deadlock_patience=2,
        )
        player = SimpleNamespace(
            position=(1, 1), orientation=(0, 1), held_object=None
        )
        state = SimpleNamespace(players=[player, player], objects={})
        context = {
            "recipe": ["tomato", "tomato", "onion"],
            "pot_ingredients": ["tomato", "tomato", "onion"],
            "pot_status": "cooking",
            "agent_holding": None,
        }
        feasible = ["GET_DISH", "GET_TOMATO", "GET_ONION", "WAIT"]
        with (
            patch(
                "durf.baseline.comfort_subgoal_agent.context_from_state",
                return_value=context,
            ),
            patch(
                "durf.baseline.comfort_subgoal_agent.enumerate_feasible_subgoals",
                return_value=feasible,
            ),
            patch(
                "durf.baseline.comfort_subgoal_agent.plan_h0_subgoal",
                return_value="WAIT",
            ),
            patch(
                "durf.baseline.comfort_subgoal_agent.execute_subgoal",
                side_effect=lambda _state, _planner, subgoal, **_kwargs: (
                    4 if subgoal == "WAIT" else 0
                ),
            ),
        ):
            choices = [agent.act(state)[1] for _ in range(5)]

        self.assertEqual(choices, ["WAIT"] * 5)
        self.assertTrue(agent.last_decision["deadlock_detected"])
        self.assertFalse(agent.last_decision["policy_overrides_enabled"])
        self.assertEqual(agent.last_decision["decision_source"], "h0_tie_fallback")
        self.assertFalse(agent.last_decision["liveness_constraint_applied"])
        self.assertEqual(
            agent.last_decision["liveness_reason"],
            "passive_cooking_wait_exempt",
        )
        self.assertTrue(agent.last_decision["passive_cooking_wait_exempt"])

    def test_prefetched_dish_may_wait_for_cooking_soup_when_not_blocking(self) -> None:
        motion_planner = SimpleNamespace(mdp=object())
        weights = {feature: 0.0 for feature in load_features()}
        weights["delays_serving"] = 1.0
        agent = ComfortSubgoalAgent(
            motion_planner,
            weights=weights,
            feedback_mode="frozen",
            max_consecutive_wait=2,
        )
        ai = SimpleNamespace(
            position=(1, 1),
            orientation=(0, 1),
            held_object=SimpleNamespace(name="dish"),
        )
        human = SimpleNamespace(
            position=(3, 1), orientation=(0, 1), held_object=None
        )
        state = SimpleNamespace(players=[ai, human], objects={})
        context = SubgoalContext(
            recipe=["tomato", "tomato", "onion"],
            pot_ingredients=["tomato", "tomato", "onion"],
            pot_status="cooking",
            agent_holding="dish",
        )
        with (
            patch(
                "durf.baseline.comfort_subgoal_agent.context_from_state",
                return_value=context,
            ),
            patch(
                "durf.baseline.comfort_subgoal_agent.enumerate_feasible_subgoals",
                return_value=["STASH_HELD_OBJECT", "WAIT"],
            ),
            patch(
                "durf.baseline.comfort_subgoal_agent.plan_h0_subgoal",
                return_value="STASH_HELD_OBJECT",
            ),
            patch(
                "durf.baseline.comfort_subgoal_agent.execute_subgoal",
                side_effect=lambda _state, _planner, subgoal, **_kwargs: (
                    STAY if subgoal == "WAIT" else 0
                ),
            ),
        ):
            choices = [agent.act(state)[1] for _ in range(5)]

        self.assertEqual(choices, ["WAIT"] * 5)
        self.assertTrue(agent.last_decision["passive_cooking_wait_exempt"])
        self.assertFalse(agent.last_decision["liveness_constraint_applied"])

    def test_prefetched_dish_wait_is_not_exempt_when_it_blocks_access(self) -> None:
        context = SubgoalContext(
            recipe=["tomato", "tomato", "onion"],
            pot_ingredients=["tomato", "tomato", "onion"],
            pot_status="cooking",
            agent_holding="dish",
            candidate_path_effects={"WAIT": "blocks"},
        )
        agent = ComfortSubgoalAgent.__new__(ComfortSubgoalAgent)
        agent.ai_index = 0
        agent.mdp = object()
        self.assertFalse(
            agent._passive_cooking_wait_is_valid(
                SimpleNamespace(players=[]),
                context,
                "STASH_HELD_OBJECT",
            )
        )

    def test_problem_session_stalled_wait_becomes_infeasible(self) -> None:
        """Regression for the 2026-08-12 pot-access 44-WAIT failure."""

        motion_planner = SimpleNamespace(mdp=object())
        features = load_features()
        weights = {feature: 0.0 for feature in features}
        # Force the preference ordering observed in that session while keeping
        # this liveness test independent of later feature-quality fixes.
        weights["time_cost"] = 1.0
        weights["delays_serving"] = 1.0
        agent = ComfortSubgoalAgent(
            motion_planner,
            weights=weights,
            feedback_mode="frozen",
            max_consecutive_wait=3,
        )
        ai = SimpleNamespace(
            position=(4, 1), orientation=(0, -1), held_object=None
        )
        human = SimpleNamespace(
            position=(3, 1),
            orientation=(1, 0),
            held_object=SimpleNamespace(name="tomato"),
        )
        state = SimpleNamespace(players=[ai, human], objects={})
        context = {
            "recipe": ["tomato", "tomato", "onion"],
            "pot_ingredients": ["onion"],
            "pot_status": "partial",
            "agent_holding": None,
            "human_holding": "tomato",
        }
        feasible = ["GET_TOMATO", "WAIT"]
        with (
            patch(
                "durf.baseline.comfort_subgoal_agent.context_from_state",
                return_value=context,
            ),
            patch(
                "durf.baseline.comfort_subgoal_agent.enumerate_feasible_subgoals",
                return_value=feasible,
            ),
            patch(
                "durf.baseline.comfort_subgoal_agent.plan_h0_subgoal",
                return_value="GET_TOMATO",
            ),
            patch(
                "durf.baseline.comfort_subgoal_agent.execute_subgoal",
                side_effect=lambda _state, _planner, subgoal, **_kwargs: (
                    4 if subgoal == "WAIT" else 0
                ),
            ),
        ):
            choices = [agent.act(state)[1] for _ in range(4)]

        self.assertEqual(choices, ["WAIT", "WAIT", "WAIT", "GET_TOMATO"])
        self.assertTrue(agent.last_decision["liveness_constraint_applied"])
        self.assertEqual(
            agent.last_decision["liveness_reason"], "stalled_wait_infeasible"
        )
        self.assertEqual(agent.last_decision["liveness_removed_subgoals"], ["WAIT"])
        self.assertEqual(
            agent.last_decision["preconstraint_feasible_subgoals"], feasible
        )
        self.assertEqual(agent.last_decision["feasible_subgoals"], ["GET_TOMATO"])
        self.assertEqual(agent.last_decision["score_formula"], "w_dot_phi")
        self.assertFalse(agent.last_decision["policy_overrides_enabled"])

    def test_unexecutable_high_reward_subgoal_is_filtered_before_ranking(self) -> None:
        """A high w-dot-phi GET action cannot win when it only executes STAY."""

        motion_planner = SimpleNamespace(mdp=object())
        features = load_features()
        weights = {feature: 0.0 for feature in features}
        weights["pick_tomato"] = 10.0
        weights["pick_onion"] = 1.0
        agent = ComfortSubgoalAgent(
            motion_planner,
            weights=weights,
            feedback_mode="frozen",
        )
        player = SimpleNamespace(
            position=(1, 1), orientation=(0, 1), held_object=None
        )
        state = SimpleNamespace(players=[player, player], objects={})
        context = {
            "recipe": ["tomato", "tomato", "onion"],
            "pot_ingredients": [],
            "pot_status": "empty",
            "agent_holding": None,
        }
        feasible = ["GET_TOMATO", "GET_ONION", "WAIT"]

        def next_action(_state, _planner, subgoal, **_kwargs):
            return {
                "GET_TOMATO": STAY,
                "GET_ONION": 0,
                "WAIT": STAY,
            }[subgoal]

        with (
            patch(
                "durf.baseline.comfort_subgoal_agent.context_from_state",
                return_value=context,
            ),
            patch(
                "durf.baseline.comfort_subgoal_agent.enumerate_feasible_subgoals",
                return_value=feasible,
            ),
            patch(
                "durf.baseline.comfort_subgoal_agent.plan_h0_subgoal",
                return_value="GET_TOMATO",
            ),
            patch(
                "durf.baseline.comfort_subgoal_agent.execute_subgoal",
                side_effect=next_action,
            ),
        ):
            action, choice = agent.act(state)

        self.assertEqual(choice, "GET_ONION")
        self.assertEqual(action, 0)
        self.assertEqual(
            agent.last_decision["feasible_subgoals"], ["GET_ONION", "WAIT"]
        )
        self.assertTrue(
            agent.last_decision["motion_feasibility_filter_applied"]
        )
        self.assertEqual(
            agent.last_decision["motion_unexecutable_subgoals"], ["GET_TOMATO"]
        )
        self.assertEqual(
            agent.last_decision["candidate_next_actions"],
            {"GET_TOMATO": STAY, "GET_ONION": 0, "WAIT": STAY},
        )
        self.assertEqual(agent.last_decision["score_formula"], "w_dot_phi")
        self.assertNotIn(
            "GET_TOMATO",
            [row["subgoal"] for row in agent.last_decision["ranking"]],
        )

    def test_two_tile_motion_cycle_temporarily_removes_stalled_goal(self) -> None:
        motion_planner = SimpleNamespace(mdp=object())
        weights = {feature: 0.0 for feature in load_features()}
        weights["pick_tomato"] = 10.0
        agent = ComfortSubgoalAgent(
            motion_planner,
            weights=weights,
            feedback_mode="frozen",
            deadlock_patience=2,
        )

        def state_at(position, orientation):
            ai = SimpleNamespace(
                position=position,
                orientation=orientation,
                held_object=None,
            )
            human = SimpleNamespace(
                position=(9, 9), orientation=(0, 1), held_object=None
            )
            return SimpleNamespace(players=[ai, human], objects={})

        states = [
            state_at((3, 1), (-1, 0)),
            state_at((4, 1), (1, 0)),
            state_at((3, 1), (-1, 0)),
            state_at((4, 1), (1, 0)),
        ]
        context = {
            "recipe": ["tomato", "tomato", "onion"],
            "pot_ingredients": ["onion", "tomato"],
            "pot_status": "partial",
        }
        with (
            patch(
                "durf.baseline.comfort_subgoal_agent.context_from_state",
                return_value=context,
            ),
            patch(
                "durf.baseline.comfort_subgoal_agent.enumerate_feasible_subgoals",
                return_value=["GET_TOMATO", "WAIT"],
            ),
            patch(
                "durf.baseline.comfort_subgoal_agent.plan_h0_subgoal",
                return_value="GET_TOMATO",
            ),
            patch(
                "durf.baseline.comfort_subgoal_agent.execute_subgoal",
                side_effect=lambda _state, _planner, subgoal, **_kwargs: (
                    STAY if subgoal == "WAIT" else 0
                ),
            ),
        ):
            choices = [agent.act(state)[1] for state in states]

        self.assertEqual(choices, ["GET_TOMATO"] * 3 + ["WAIT"])
        self.assertTrue(agent.last_decision["stalled_motion_constraint_applied"])
        self.assertEqual(
            agent.last_decision["stalled_motion_removed_subgoal"],
            "GET_TOMATO",
        )
        self.assertEqual(agent.last_decision["feasible_subgoals"], ["WAIT"])
        self.assertEqual(agent.last_decision["motion_cycle_steps"], 2)
        self.assertEqual(agent.last_decision["score_formula"], "w_dot_phi")

    def test_cross_subgoal_motion_cycle_commits_to_one_reward_argmax(self) -> None:
        """Regression for the live GET_DISH/GET_TOMATO hesitation loop."""

        motion_planner = SimpleNamespace(mdp=object())
        weights = {feature: 0.0 for feature in load_features()}
        weights["clears_human_path"] = 2.0
        weights["blocks_human_path"] = -2.0
        agent = ComfortSubgoalAgent(
            motion_planner,
            weights=weights,
            feedback_mode="frozen",
            deadlock_patience=2,
        )

        def state_at(position, orientation):
            ai = SimpleNamespace(
                position=position,
                orientation=orientation,
                held_object=None,
            )
            human = SimpleNamespace(
                position=(9, 9), orientation=(0, 1), held_object=None
            )
            return SimpleNamespace(players=[ai, human], objects={})

        state_a = state_at((1, 2), (0, -1))
        state_b = state_at((1, 3), (0, 1))

        def context_for(state, *_args, **_kwargs):
            tomato_wins = tuple(state.players[0].position) == (1, 2)
            return {
                "recipe": ["tomato", "tomato", "onion"],
                "pot_ingredients": ["onion", "tomato"],
                "pot_status": "partial",
                "human_intent": "get tomato",
                "candidate_path_effects": {
                    "GET_TOMATO": "clears" if tomato_wins else "blocks",
                    "GET_DISH": "blocks" if tomato_wins else "clears",
                },
            }

        with (
            patch(
                "durf.baseline.comfort_subgoal_agent.context_from_state",
                side_effect=context_for,
            ),
            patch(
                "durf.baseline.comfort_subgoal_agent.enumerate_feasible_subgoals",
                return_value=["GET_TOMATO", "GET_DISH", "WAIT"],
            ),
            patch(
                "durf.baseline.comfort_subgoal_agent.plan_h0_subgoal",
                return_value="GET_TOMATO",
            ),
            patch(
                "durf.baseline.comfort_subgoal_agent.execute_subgoal",
                side_effect=lambda _state, _planner, subgoal, **_kwargs: (
                    STAY
                    if subgoal == "WAIT"
                    else (0 if subgoal == "GET_TOMATO" else 1)
                ),
            ),
        ):
            choices = [
                agent.act(state)[1]
                for state in [state_a, state_b, state_a, state_b, state_a]
            ]

        # GET_TOMATO is the first w·phi winner. At the next position the raw
        # winner flips to GET_DISH, but the already selected option is pursued
        # rather than reversing immediately. The artificial A-B-A-B states
        # eventually prove that option itself stalled, so the existing generic
        # deadlock constraint releases it and selects GET_DISH.
        self.assertEqual(
            choices,
            ["GET_TOMATO", "GET_TOMATO", "GET_TOMATO", "GET_DISH", "GET_DISH"],
        )
        held = agent.decision_history[1]
        self.assertTrue(held["option_commitment_triggered"])
        self.assertTrue(held["oscillation_commitment_applied"])
        self.assertEqual(held["option_uncommitted_chosen_subgoal"], "GET_DISH")
        self.assertEqual(held["chosen_subgoal"], "GET_TOMATO")
        self.assertEqual(held["stability_reason"], "subgoal_option_commitment")
        self.assertEqual(held["score_formula"], "w_dot_phi")
        self.assertFalse(held["policy_overrides_enabled"])
        escaped = agent.decision_history[3]
        self.assertTrue(escaped["stalled_motion_constraint_applied"])
        self.assertEqual(escaped["chosen_subgoal"], "GET_DISH")

    def test_known_joint_collision_exposes_and_selects_generic_yield(self) -> None:
        """Regression for live steps 322-325: both players targeted one tile."""

        valid = {(1, 1), (1, 2), (1, 3), (1, 4), (2, 1), (2, 2), (2, 3)}
        mdp = SimpleNamespace(get_valid_player_positions=lambda: list(valid))
        motion_planner = SimpleNamespace(mdp=mdp)
        weights = {feature: 0.0 for feature in load_features()}
        weights["pick_tomato"] = 10.0
        weights["clears_human_path"] = 6.0
        weights["clears_human_shortest_path"] = 2.0
        weights["avoids_human_shortest_path"] = 2.0
        agent = ComfortSubgoalAgent(
            motion_planner,
            weights=weights,
            feedback_mode="frozen",
        )
        state = SimpleNamespace(
            players=[
                SimpleNamespace(position=(1, 2), orientation=(0, 1), held_object=None),
                SimpleNamespace(position=(1, 4), orientation=(0, -1), held_object=None),
            ],
            objects={},
        )
        context = {
            "recipe": ["tomato", "tomato", "onion"],
            "pot_ingredients": ["onion", "tomato"],
            "pot_status": "partial",
            "candidate_path_effects": {},
        }
        south = int(Action.ACTION_TO_INDEX[(0, 1)])
        north = int(Action.ACTION_TO_INDEX[(0, -1)])
        with (
            patch(
                "durf.baseline.comfort_subgoal_agent.context_from_state",
                return_value=context,
            ),
            patch(
                "durf.baseline.comfort_subgoal_agent.enumerate_feasible_subgoals",
                return_value=["GET_TOMATO", "WAIT"],
            ),
            patch(
                "durf.baseline.comfort_subgoal_agent.plan_h0_subgoal",
                return_value="GET_TOMATO",
            ),
            patch(
                "durf.baseline.comfort_subgoal_agent.execute_subgoal",
                side_effect=lambda _state, _planner, subgoal, **_kwargs: (
                    south if subgoal == "GET_TOMATO" else STAY
                ),
            ),
        ):
            action, choice = agent.act(state, human_action=north)

        self.assertEqual(choice, "YIELD_PATH")
        self.assertNotIn(action, {south, STAY})
        self.assertEqual(agent.last_decision["joint_motion_conflicts"], ["GET_TOMATO"])
        self.assertTrue(agent.last_decision["joint_motion_filter_applied"])
        self.assertIn("YIELD_PATH", agent.last_decision["feasible_subgoals"])
        self.assertEqual(agent.last_decision["score_formula"], "w_dot_phi")
        self.assertFalse(agent.last_decision["policy_overrides_enabled"])

    def test_all_unexecutable_productive_subgoals_fall_back_to_wait(self) -> None:
        motion_planner = SimpleNamespace(mdp=object())
        features = load_features()
        weights = {feature: 0.0 for feature in features}
        weights["pick_tomato"] = 10.0
        weights["pick_onion"] = 9.0
        agent = ComfortSubgoalAgent(
            motion_planner,
            weights=weights,
            feedback_mode="frozen",
        )
        player = SimpleNamespace(
            position=(1, 1), orientation=(0, 1), held_object=None
        )
        state = SimpleNamespace(players=[player, player], objects={})
        context = {
            "recipe": ["tomato", "tomato", "onion"],
            "pot_ingredients": [],
            "pot_status": "empty",
            "agent_holding": None,
        }
        feasible = ["GET_TOMATO", "GET_ONION", "WAIT"]
        with (
            patch(
                "durf.baseline.comfort_subgoal_agent.context_from_state",
                return_value=context,
            ),
            patch(
                "durf.baseline.comfort_subgoal_agent.enumerate_feasible_subgoals",
                return_value=feasible,
            ),
            patch(
                "durf.baseline.comfort_subgoal_agent.plan_h0_subgoal",
                return_value="GET_TOMATO",
            ),
            patch(
                "durf.baseline.comfort_subgoal_agent.execute_subgoal",
                return_value=STAY,
            ),
        ):
            action, choice = agent.act(state)

        self.assertEqual(choice, "WAIT")
        self.assertEqual(action, STAY)
        self.assertEqual(agent.last_decision["feasible_subgoals"], ["WAIT"])
        self.assertEqual(agent.last_decision["executable_productive_subgoals"], [])
        self.assertEqual(
            agent.last_decision["motion_unexecutable_subgoals"],
            ["GET_TOMATO", "GET_ONION"],
        )
        self.assertEqual(
            [row["subgoal"] for row in agent.last_decision["ranking"]], ["WAIT"]
        )

    def test_passive_cooking_wait_exemption_is_denied_when_ai_blocks_pot(self) -> None:
        mdp = SimpleNamespace(
            get_pot_locations=lambda: [(4, 0)],
            get_serving_locations=lambda: [(9, 3)],
            get_valid_player_positions=lambda: [(4, 1), (3, 1)],
        )
        motion_planner = SimpleNamespace(mdp=mdp)
        features = load_features()
        weights = {feature: 0.0 for feature in features}
        weights["pick_dish"] = -1.0
        agent = ComfortSubgoalAgent(
            motion_planner,
            weights=weights,
            feedback_mode="frozen",
            max_consecutive_wait=2,
        )
        ai = SimpleNamespace(
            position=(4, 1), orientation=(0, -1), held_object=None
        )
        human = SimpleNamespace(
            position=(3, 1),
            orientation=(1, 0),
            held_object=SimpleNamespace(name="dish"),
        )
        state = SimpleNamespace(players=[ai, human], objects={})
        context = {
            "recipe": ["tomato", "tomato", "onion"],
            "pot_ingredients": ["tomato", "tomato", "onion"],
            "pot_status": "cooking",
            "agent_holding": None,
            "human_holding": "dish",
        }
        with (
            patch(
                "durf.baseline.comfort_subgoal_agent.context_from_state",
                return_value=context,
            ),
            patch(
                "durf.baseline.comfort_subgoal_agent.enumerate_feasible_subgoals",
                return_value=["GET_DISH", "WAIT"],
            ),
            patch(
                "durf.baseline.comfort_subgoal_agent.plan_h0_subgoal",
                return_value="WAIT",
            ),
            patch(
                "durf.baseline.comfort_subgoal_agent.execute_subgoal",
                side_effect=lambda _state, _planner, subgoal, **_kwargs: (
                    4 if subgoal == "WAIT" else 0
                ),
            ),
        ):
            choices = [agent.act(state)[1] for _ in range(3)]

        self.assertEqual(choices, ["WAIT", "WAIT", "GET_DISH"])
        self.assertTrue(agent.last_decision["liveness_constraint_applied"])
        self.assertFalse(agent.last_decision["passive_cooking_wait_exempt"])
        self.assertEqual(
            agent.last_decision["liveness_reason"], "stalled_wait_infeasible"
        )

    def test_stalled_wait_is_kept_when_no_productive_action_is_executable(self) -> None:
        motion_planner = SimpleNamespace(mdp=object())
        features = load_features()
        weights = {feature: 0.0 for feature in features}
        weights["time_cost"] = 1.0
        weights["delays_serving"] = 1.0
        agent = ComfortSubgoalAgent(
            motion_planner,
            weights=weights,
            feedback_mode="frozen",
            max_consecutive_wait=2,
        )
        player = SimpleNamespace(
            position=(1, 1), orientation=(0, 1), held_object=None
        )
        state = SimpleNamespace(players=[player, player], objects={})
        context = {
            "recipe": ["tomato", "tomato", "onion"],
            "pot_ingredients": ["onion"],
            "pot_status": "partial",
            "agent_holding": None,
            "human_holding": "tomato",
        }
        with (
            patch(
                "durf.baseline.comfort_subgoal_agent.context_from_state",
                return_value=context,
            ),
            patch(
                "durf.baseline.comfort_subgoal_agent.enumerate_feasible_subgoals",
                return_value=["GET_TOMATO", "WAIT"],
            ),
            patch(
                "durf.baseline.comfort_subgoal_agent.plan_h0_subgoal",
                return_value="GET_TOMATO",
            ),
            patch(
                "durf.baseline.comfort_subgoal_agent.execute_subgoal",
                return_value=4,
            ),
        ):
            choices = [agent.act(state)[1] for _ in range(4)]

        self.assertEqual(choices, ["WAIT"] * 4)
        self.assertFalse(agent.last_decision["liveness_constraint_applied"])
        self.assertEqual(
            agent.last_decision["liveness_reason"],
            "no_executable_productive_alternative",
        )

    def test_updated_feedback_traces_accepted_without_policy_switch(self) -> None:
        trace = {
            "status": "updated",
            "before_subgoal": "GET_TOMATO",
            "after_subgoal": "GET_TOMATO",
        }
        ComfortSubgoalAgent._annotate_policy_switch(trace)
        self.assertFalse(trace["policy_switch"])
        self.assertTrue(trace["accepted_but_no_policy_switch"])


class MultiSeedSelectionTests(unittest.TestCase):
    @staticmethod
    def _row(
        *,
        soup: float,
        comfort: float,
        discomfort: float,
        waits: int = 0,
    ) -> dict:
        return {
            "steps": 100,
            "soup_reward": soup,
            "comfort_per_step": comfort,
            "discomfort_rate": discomfort,
            "longest_wait_streak": min(waits, 3),
            "subgoal_counts": {"WAIT": waits},
        }

    def test_fixed_paper_scale_reports_dev_and_untouched_test(self) -> None:
        baseline = {
            0: self._row(soup=100, comfort=0.0, discomfort=0.5),
            1: self._row(soup=100, comfort=0.0, discomfort=0.5),
            2: self._row(soup=100, comfort=0.0, discomfort=0.5),
        }
        comfort = {
            0: self._row(soup=100, comfort=0.10, discomfort=0.4),
            1: self._row(soup=100, comfort=0.10, discomfort=0.4),
            2: self._row(soup=100, comfort=0.10, discomfort=0.4),
        }

        def fake_rollout(kind, _layout, seed, _horizon, lambda_pref):
            if kind == "h0_rule":
                return baseline[seed]
            self.assertEqual(lambda_pref, 1.0)
            return comfort[seed]

        with patch.object(evaluate_comfort_multiseed, "rollout", side_effect=fake_rollout):
            result = evaluate_comfort_multiseed.evaluate(
                layout="test",
                horizon=100,
                dev_seeds=[0, 1],
                test_seeds=[2],
                lambdas=[1.0],
            )
        self.assertEqual(result["selected_lambda"], 1.0)
        self.assertEqual(result["untouched_test"]["comfort"]["n"], 1)

    def test_manual_comfort_scale_is_rejected_by_evaluator(self) -> None:
        with self.assertRaisesRegex(ValueError, "fixes lambda_pref=1.0"):
            evaluate_comfort_multiseed.evaluate(
                layout="test",
                horizon=10,
                dev_seeds=[0],
                test_seeds=[1],
                lambdas=[0.5, 1.0],
            )

    def test_multiseed_evaluator_rejects_dev_test_overlap(self) -> None:
        with self.assertRaisesRegex(ValueError, "seed overlap"):
            evaluate_comfort_multiseed.evaluate(
                layout="test",
                horizon=10,
                dev_seeds=[0, 1],
                test_seeds=[1, 2],
                lambdas=[1.0],
            )


if __name__ == "__main__":
    unittest.main()
