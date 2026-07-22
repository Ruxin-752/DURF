"""StallBreaker tests on the real ring layout (no API, deterministic)."""

from __future__ import annotations

import unittest

from durf.baseline.coordination import STAY, StallBreaker
from durf.baseline.h0_planner import make_motion_planner
from durf.baseline.runtime import RING_TOMATO_ONION_H0_LAYOUT, make_direct_multi_env

LAYOUT = RING_TOMATO_ONION_H0_LAYOUT


def _state_and_mdp():
    mp = make_motion_planner(LAYOUT, 42, 40)
    env = make_direct_multi_env(LAYOUT, 42, horizon=40)
    env.multi_reset()
    return env.base_env.state, mp.mdp


class StallBreakerTests(unittest.TestCase):
    def test_patience_tolerates_short_stay(self) -> None:
        state, mdp = _state_and_mdp()
        breaker = StallBreaker(mdp, patience=2, seed=0)
        for _ in range(2):
            action, nudged = breaker.resolve(state, 0, STAY, other_index=1)
            self.assertEqual(action, STAY)
            self.assertFalse(nudged)

    def test_persistent_stall_gets_a_valid_move(self) -> None:
        state, mdp = _state_and_mdp()
        breaker = StallBreaker(mdp, patience=2, seed=0)
        actions = [breaker.resolve(state, 0, STAY, other_index=1) for _ in range(4)]
        # Beyond patience, a move (non-STAY) must be emitted.
        self.assertTrue(any(nudged for _, nudged in actions))
        moved = [a for a, nudged in actions if nudged]
        self.assertTrue(moved and all(a in range(4) for a in moved))

    def test_nudge_never_steps_onto_partner(self) -> None:
        state, mdp = _state_and_mdp()
        from overcooked_ai_py.mdp.actions import Action

        breaker = StallBreaker(mdp, patience=0, seed=1)
        other = tuple(state.players[1].position)
        pos = tuple(state.players[0].position)
        # Force a nudge and check the resulting target tile avoids the partner.
        breaker.resolve(state, 0, STAY, other_index=1)  # prime last_pos/stall
        action, nudged = breaker.resolve(state, 0, STAY, other_index=1)
        if nudged:
            target = Action.move_in_direction(pos, Action.INDEX_TO_ACTION[action])
            self.assertNotEqual(tuple(target), other)

    def test_progress_resets_stall(self) -> None:
        state, mdp = _state_and_mdp()
        breaker = StallBreaker(mdp, patience=2, seed=0)
        # A moving action keeps stall at zero, so STAY is honored afterward.
        breaker.resolve(state, 0, 0, other_index=1)  # north (moves)
        action, nudged = breaker.resolve(state, 0, STAY, other_index=1)
        self.assertEqual(action, STAY)
        self.assertFalse(nudged)


if __name__ == "__main__":
    unittest.main()
