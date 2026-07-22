"""Path A consistency tests: execution reuse + comfort agent wiring.

These run headless on the real ring layout (overcooked + motion planner only).
The central guarantee is that the newly factored ``execute_subgoal`` reproduces
the frozen ``rule_teacher_decision`` execution exactly, so the comfort agent
only changes subgoal *selection*, never H0's motion behavior.
"""

from __future__ import annotations

import unittest

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
            motion_planner, weights=weights, lambda_pref=1.0, ai_index=0
        )
        for state, _ in _states_from_rollout(steps=40):
            context = context_from_state(state, agent.mdp, ai_index=0)
            expected = plan_subgoal(weights, context, lambda_pref=1.0)["chosen_subgoal"]
            _, subgoal = agent.act(state)
            self.assertEqual(subgoal, expected)


if __name__ == "__main__":
    unittest.main()
