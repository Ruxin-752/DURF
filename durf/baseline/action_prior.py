"""Small action priors used to diagnose brittle PPO subskills."""

from __future__ import annotations

from dataclasses import dataclass

from overcooked_ai_py.mdp.actions import Action, Direction


COUNTER_ONION_TO_POT_TOP = "counter_onion_to_pot_top"


PREFIX_ACTIONS = {
    COUNTER_ONION_TO_POT_TOP: (
        Action.INTERACT,  # pick onion from counter
        Direction.EAST,  # move below the pot
        Direction.NORTH,  # face the pot
        Action.INTERACT,  # place onion into pot
    ),
}


def available_priors() -> tuple[str, ...]:
    return tuple(PREFIX_ACTIONS)


def prefix_action_indices(name: str) -> list[int]:
    try:
        actions = PREFIX_ACTIONS[name]
    except KeyError as exc:
        choices = ", ".join(available_priors())
        raise ValueError(f"Unknown action prior {name!r}. Choices: {choices}") from exc
    return [Action.ACTION_TO_INDEX[action] for action in actions]


def prefix_action_chars(name: str) -> list[str]:
    try:
        actions = PREFIX_ACTIONS[name]
    except KeyError as exc:
        choices = ", ".join(available_priors())
        raise ValueError(f"Unknown action prior {name!r}. Choices: {choices}") from exc
    return [Action.ACTION_TO_CHAR[action] for action in actions]


@dataclass
class StepPrefixPrior:
    """A deterministic prefix that yields a few actions, then hands off to PPO."""

    name: str
    action_indices: list[int]
    cursor: int = 0

    @classmethod
    def from_name(cls, name: str | None) -> "StepPrefixPrior | None":
        if not name:
            return None
        return cls(name=name, action_indices=prefix_action_indices(name))

    def reset(self) -> None:
        self.cursor = 0

    def next_action(self) -> int | None:
        if self.cursor >= len(self.action_indices):
            return None
        action_index = self.action_indices[self.cursor]
        self.cursor += 1
        return action_index
