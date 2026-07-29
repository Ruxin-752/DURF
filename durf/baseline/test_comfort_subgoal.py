"""Path A consistency tests: execution reuse + comfort agent wiring.

These run headless on the real ring layout (overcooked + motion planner only).
The central guarantee is that the newly factored ``execute_subgoal`` reproduces
the frozen ``rule_teacher_decision`` execution exactly, so the comfort agent
only changes subgoal *selection*, never H0's motion behavior.
"""

from __future__ import annotations

import unittest
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from durf.baseline import evaluate_comfort_multiseed
from durf.baseline.comfort_reward import context_from_state
from durf.baseline.comfort_subgoal_agent import ComfortSubgoalAgent, load_comfort_weights
from durf.baseline.h0_planner import (
    execute_subgoal,
    make_motion_planner,
    rule_teacher_decision,
)
from durf.baseline.runtime import RING_TOMATO_ONION_H0_LAYOUT, make_direct_multi_env

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


class ComfortAgentWiringTests(unittest.TestCase):
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
            context = context_from_state(state, agent.mdp, ai_index=0)
            expected = plan_subgoal(weights, context, lambda_pref=1.0)["chosen_subgoal"]
            _, subgoal = agent.act(state)
            self.assertEqual(subgoal, expected)

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

    def test_real_inferred_feedback_traces_accepted_without_switch(self) -> None:
        """Accepted feedback is distinguishable from an actual policy switch."""

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
            )
            after = agent.plan(state)

            self.assertEqual(trace["status"], "updated", msg=trace)
            self.assertEqual(trace["source"], "human_live")
            self.assertEqual(trace["source_precision_multiplier"], 4.0)
            self.assertIn("action_behavioral", trace["reference_types"])
            self.assertEqual(trace["after_subgoal"], "GET_TOMATO")
            self.assertTrue(trace["accepted_but_no_policy_switch"])
            self.assertFalse(trace["policy_switch"])
            self.assertEqual(after["chosen_subgoal"], "GET_TOMATO")
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
            self.assertEqual(resumed.plan(state)["chosen_subgoal"], "GET_TOMATO")

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
        self.assertIn(
            "ingredient_onion",
            trace["observations"][0]["target_features"],
        )

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

    def test_actual_state_deadlock_uses_deterministic_nonwait_fallback(self) -> None:
        motion_planner = SimpleNamespace(mdp=object())
        agent = ComfortSubgoalAgent(
            motion_planner,
            weights={},
            feedback_mode="frozen",
            deadlock_patience=2,
            subgoal_commitment_steps=5,
        )
        player = SimpleNamespace(
            position=(1, 1), orientation=(0, 1), held_object=None
        )
        state = SimpleNamespace(players=[player, player], objects={})
        ranked = {
            "chosen_subgoal": "GET_TOMATO",
            "feasible_subgoals": ["GET_TOMATO", "GET_ONION", "WAIT"],
            "ranking": [
                {"subgoal": "GET_TOMATO", "total_score": 2.0},
                {"subgoal": "GET_ONION", "total_score": 1.0},
                {"subgoal": "WAIT", "total_score": 0.0},
            ],
        }
        with (
            patch(
                "durf.baseline.comfort_subgoal_agent.context_from_state",
                return_value=object(),
            ),
            patch(
                "durf.baseline.comfort_subgoal_agent.enumerate_feasible_subgoals",
                return_value=["GET_TOMATO", "GET_ONION", "WAIT"],
            ),
            patch(
                "durf.baseline.comfort_subgoal_agent.plan_subgoal",
                return_value=ranked,
            ),
            patch(
                "durf.baseline.comfort_subgoal_agent.execute_subgoal",
                side_effect=lambda _s, _m, goal, player_index: (
                    1 if goal == "GET_ONION" else 4
                ),
            ),
        ):
            agent.act(state)
            agent.act(state)
            action, subgoal = agent.act(state)

        self.assertEqual((action, subgoal), (1, "GET_ONION"))
        self.assertTrue(agent.last_decision["deadlock_detected"])
        self.assertNotEqual(subgoal, "WAIT")

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

    def test_selects_best_tenth_percentile_comfort_delta(self) -> None:
        baseline = {
            0: self._row(soup=100, comfort=0.0, discomfort=0.5),
            1: self._row(soup=100, comfort=0.0, discomfort=0.5),
            2: self._row(soup=100, comfort=0.0, discomfort=0.5),
        }
        by_lambda = {
            0.5: {
                0: self._row(soup=100, comfort=0.10, discomfort=0.4),
                1: self._row(soup=100, comfort=0.10, discomfort=0.4),
                2: self._row(soup=100, comfort=0.10, discomfort=0.4),
            },
            1.0: {
                0: self._row(soup=100, comfort=-0.20, discomfort=0.4),
                1: self._row(soup=100, comfort=0.80, discomfort=0.4),
                2: self._row(soup=100, comfort=0.80, discomfort=0.4),
            },
        }

        def fake_rollout(kind, _layout, seed, _horizon, lambda_pref):
            if kind == "h0_rule":
                return baseline[seed]
            return by_lambda[lambda_pref][seed]

        with patch.object(evaluate_comfort_multiseed, "rollout", side_effect=fake_rollout):
            result = evaluate_comfort_multiseed.evaluate(
                layout="test",
                horizon=100,
                dev_seeds=[0, 1],
                test_seeds=[2],
                lambdas=[0.5, 1.0],
            )
        self.assertEqual(result["selected_lambda"], 0.5)
        summaries = {
            row["lambda_pref"]: row["summary"] for row in result["dev_candidates"]
        }
        self.assertGreater(
            summaries[0.5]["comfort_delta_p10"],
            summaries[1.0]["comfort_delta_p10"],
        )


if __name__ == "__main__":
    unittest.main()
