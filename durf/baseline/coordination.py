"""Symmetric low-level anti-stall coordinator for headless simulated play.

In the real pygame game a *human* breaks physical stand-offs by simply walking
around. Our headless harness drives both seats from H0 execution
(``execute_subgoal``), which returns ``STAY`` whenever the single access tile to
a feature is occupied by the other agent. On a tight ring with single-access
pots/dispensers this produces a *static* mutual deadlock: both agents want a
tile the other is parked on, both return ``STAY`` forever.

``StallBreaker`` restores the walking-around behavior a person would supply: if
an agent has been stuck (``STAY`` / no cell change) for a few steps while it
still has a goal, it takes a valid neighboring step that does not walk onto the
partner, breaking the symmetry. Once unblocked, goal-directed execution resumes
on the next step. It is applied identically to both seats so any AI comparison
stays fair.
"""

from __future__ import annotations

import random

from overcooked_ai_py.mdp.actions import Action

STAY = int(Action.ACTION_TO_INDEX[Action.STAY])
_MOVE_INDICES = tuple(
    int(Action.ACTION_TO_INDEX[a])
    for a in Action.MOTION_ACTIONS
    if a != Action.STAY
)


class StallBreaker:
    """Per-agent stall detector that emits a de-stalling step when frozen.

    ``patience`` STAYs are tolerated (a legitimate short yield); beyond that the
    agent is nudged. ``seed`` makes the tie-breaking reproducible.
    """

    def __init__(self, mdp, *, patience: int = 2, seed: int = 0) -> None:
        self.mdp = mdp
        self.patience = int(patience)
        self._rng = random.Random(seed)
        self._last_pos: dict[int, tuple[int, int]] = {}
        self._stall: dict[int, int] = {}

    def resolve(
        self, state, player_index: int, proposed_action: int, other_index: int
    ) -> tuple[int, bool]:
        """Return ``(action, nudged)`` for ``player_index`` this step."""

        proposed_action = int(proposed_action)
        player = state.players[player_index]
        pos = tuple(player.position)

        stuck = proposed_action == STAY or pos == self._last_pos.get(player_index)
        self._stall[player_index] = self._stall.get(player_index, 0) + 1 if stuck else 0
        self._last_pos[player_index] = pos

        if self._stall.get(player_index, 0) <= self.patience:
            return proposed_action, False

        other_pos = tuple(state.players[other_index].position)
        valid = set(self.mdp.get_valid_player_positions())
        moves = list(_MOVE_INDICES)
        self._rng.shuffle(moves)
        for action in moves:
            target = Action.move_in_direction(pos, Action.INDEX_TO_ACTION[action])
            if target in valid and target != other_pos:
                self._stall[player_index] = 0
                return int(action), True
        return STAY, False
