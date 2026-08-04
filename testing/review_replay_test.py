"""Tests for attribution review map replay helpers."""

from __future__ import annotations

import unittest

from durf.feedback_attribution.review_replay import (
    event_names_at_step,
    select_replay_steps,
)


class ReviewReplayTests(unittest.TestCase):
    def test_selects_context_around_attribution_window(self):
        steps = [{"total_step": step} for step in range(10, 21)]

        selected = select_replay_steps(
            steps,
            [14, 16],
            context_steps=2,
        )

        self.assertEqual(
            [step["total_step"] for step in selected],
            [12, 13, 14, 15, 16, 17, 18],
        )

    def test_invalid_window_returns_all_ordered_steps(self):
        steps = [{"total_step": 3}, {"total_step": 1}, {"total_step": 2}]

        selected = select_replay_steps(steps, None)

        self.assertEqual(
            [step["total_step"] for step in selected],
            [1, 2, 3],
        )

    def test_event_names_cover_current_frame_only(self):
        events = [
            {
                "event_type": "AI_blocked_human_path",
                "start_timestep": 10,
                "end_timestep": 12,
            },
            {
                "event_type": "AI_pick_drop_loop",
                "start_timestep": 20,
                "end_timestep": 25,
            },
        ]

        self.assertEqual(
            event_names_at_step(events, 11),
            ["AI_blocked_human_path"],
        )
        self.assertEqual(event_names_at_step(events, 15), [])


if __name__ == "__main__":
    unittest.main()
