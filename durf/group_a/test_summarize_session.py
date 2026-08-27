"""Tests for the compact human-AI session diagnostic."""

from __future__ import annotations

import unittest

from durf.group_a.summarize_session import format_summary, summarize_records


class SummarizeSessionTests(unittest.TestCase):
    def test_reports_frozen_updates_reward_and_wait_streak(self) -> None:
        manifest = {
            "ai_mode": "comfort_subgoal",
            "feedback_mode": "frozen",
            "online_learning_enabled": False,
            "teacher_id": "control-1",
            "seed": 42,
        }
        rows = [
            {
                "total_step": str(step),
                "ai_mode": "comfort_subgoal",
                "comfort_feedback_mode": "frozen",
                "ai_subgoal": subgoal,
                "environment_reward": str(reward),
            }
            for step, subgoal, reward in (
                (1, "GET_ONION", 0),
                (2, "WAIT", 0),
                (3, "WAIT", 0),
                (4, "WAIT", 20),
            )
        ]
        summary = summarize_records(
            "sample-session",
            rows,
            manifest,
            [{"status": "ignored", "mode": "frozen"}],
        )

        self.assertEqual(summary["reward"], 20)
        self.assertEqual(summary["longest_wait_streak"]["length"], 3)
        self.assertEqual(summary["feedback_status_counts"], {"ignored": 1})
        self.assertFalse(summary["online_learning_enabled"])
        rendered = format_summary(summary)
        self.assertIn("online learning: OFF", rendered)
        self.assertIn("FROZEN control", rendered)

    def test_reports_immediate_stash_repick_cycles(self) -> None:
        rows = [
            {
                "total_step": "1",
                "ai_subgoal": "STASH_HELD_OBJECT",
                "ai_action": "5",
                "environment_reward": "0",
            },
            {
                "total_step": "2",
                "ai_subgoal": "GET_ONION",
                "ai_action": "5",
                "environment_reward": "0",
            },
            {
                "total_step": "3",
                "ai_subgoal": "WAIT",
                "ai_action": "4",
                "environment_reward": "0",
            },
        ]
        summary = summarize_records("sample", rows, {}, [])
        self.assertEqual(
            summary["immediate_stash_repick"],
            {"count": 1, "by_resource": {"onion": 1}, "steps": [2]},
        )
        self.assertIn("STASH->GET reversal", " ".join(summary["alerts"]))


if __name__ == "__main__":
    unittest.main()
