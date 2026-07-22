"""DeepSeek human-partner tests using a stubbed chat function (no real API).

Validates that: a well-formed JSON reply is honored and executed via H0; a
malformed / infeasible reply falls back to the rule teacher; and re-planning is
throttled by ``replan_every`` so we don't hammer the API every timestep.
"""

from __future__ import annotations

import unittest

from durf.baseline.deepseek_human import DeepSeekHumanAgent
from durf.baseline.h0_planner import (
    execute_subgoal,
    make_motion_planner,
    rule_teacher_decision,
)
from durf.baseline.runtime import RING_TOMATO_ONION_H0_LAYOUT, make_direct_multi_env

LAYOUT = RING_TOMATO_ONION_H0_LAYOUT


def _first_state():
    motion_planner = make_motion_planner(LAYOUT, 42, 50)
    env = make_direct_multi_env(LAYOUT, 42, horizon=50)
    env.multi_reset()
    return env.base_env.state, motion_planner


class DeepSeekHumanTests(unittest.TestCase):
    def test_valid_reply_is_executed_via_h0(self) -> None:
        state, mp = _first_state()
        calls = []

        def chat(messages):
            calls.append(messages)
            return '{"subgoal": "GET_ONION", "say": "I\'ll grab an onion."}'

        agent = DeepSeekHumanAgent(mp, player_index=1, chat_fn=chat)
        action, subgoal = agent.act(state)
        self.assertEqual(subgoal, "GET_ONION")
        self.assertEqual(agent.last_reason, "deepseek")
        self.assertEqual(agent.last_say, "I'll grab an onion.")
        self.assertEqual(
            action, execute_subgoal(state, mp, "GET_ONION", player_index=1)
        )
        self.assertEqual(len(calls), 1)

    def test_infeasible_choice_falls_back_to_rule(self) -> None:
        state, mp = _first_state()

        def chat(messages):
            # SERVE_SOUP is not feasible empty-handed at the start.
            return '{"subgoal": "SERVE_SOUP", "say": "serving"}'

        agent = DeepSeekHumanAgent(mp, player_index=1, chat_fn=chat)
        _, subgoal = agent.act(state)
        expected, _ = rule_teacher_decision(state, mp, player_index=1)
        self.assertEqual(subgoal, expected)
        self.assertIn("rule fallback", agent.last_reason)

    def test_chat_error_falls_back_to_rule(self) -> None:
        state, mp = _first_state()

        def chat(messages):
            raise RuntimeError("no network")

        agent = DeepSeekHumanAgent(mp, player_index=1, chat_fn=chat)
        _, subgoal = agent.act(state)
        expected, _ = rule_teacher_decision(state, mp, player_index=1)
        self.assertEqual(subgoal, expected)
        self.assertTrue(agent.last_reason.startswith("chat_error"))

    def test_replan_is_throttled(self) -> None:
        motion_planner = make_motion_planner(LAYOUT, 42, 50)
        env = make_direct_multi_env(LAYOUT, 42, horizon=50)
        env.multi_reset()
        count = {"n": 0}

        def chat(messages):
            count["n"] += 1
            return '{"subgoal": "GET_ONION", "say": ""}'

        agent = DeepSeekHumanAgent(
            motion_planner, player_index=1, replan_every=5, chat_fn=chat
        )
        for _ in range(10):
            state = env.base_env.state
            action, _ = agent.act(state)
            _, a0 = rule_teacher_decision(state, motion_planner, player_index=0)
            env.multi_step(int(a0), int(action))
        # Steps 0 and 5 trigger a query (assuming GET_ONION stays feasible);
        # never once per step.
        self.assertLessEqual(count["n"], 3)
        self.assertGreaterEqual(count["n"], 1)


if __name__ == "__main__":
    unittest.main()
