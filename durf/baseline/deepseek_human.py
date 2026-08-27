"""A DeepSeek-driven "human" partner that plays at the subgoal level.

Two H0 rule agents deadlock over shared ring resources, so the automated eval
needs a partner that reasons like a person ("I'll grab an onion; you plate the
soup"). We let DeepSeek pick, from the human seat's task-valid feasible
subgoals, ONE subgoal per decision; H0's motion planner then executes it. This
keeps the LLM at the level a human actually thinks about, needs only a handful
of API calls per episode (re-plan on completion / infeasibility / refresh), and
reuses the exact same execution path as the AI so behavior stays comparable.

    state --context(human seat)--> feasible subgoals
          --DeepSeek chat--> {"subgoal": ..., "say": ...}
          --execute_subgoal(player_index=1)--> action

The chat function is injectable so the agent is unit-testable without the API.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Callable

from durf.baseline.comfort_reward import context_from_state
from durf.baseline.h0_planner import execute_subgoal, rule_teacher_decision

from src.subgoal_planner import enumerate_feasible_subgoals

STAY = 4

SYSTEM_PROMPT = (
    "You are a person playing the cooperative game Overcooked with an AI "
    "teammate. You share one kitchen. Your goal is to cook and serve soups as "
    "fast as possible WITHOUT getting in your teammate's way: divide the labor, "
    "don't chase the same ingredient or pot your teammate is already handling, "
    "and yield when you would block them. You act by choosing ONE high-level "
    "subgoal from the allowed list. Reply with ONLY a JSON object: "
    '{"subgoal": "<one of the allowed subgoals>", "say": "<short natural '
    'sentence to your teammate>"}. No prose outside the JSON.'
)

SUBGOAL_GLOSS = {
    "GET_TOMATO": "walk to a tomato dispenser and pick up a tomato",
    "GET_ONION": "walk to an onion dispenser and pick up an onion",
    "PUT_TOMATO_IN_POT": "carry your tomato to a pot that still needs one",
    "PUT_ONION_IN_POT": "carry your onion to a pot that still needs one",
    "GET_DISH": "fetch a clean dish to plate a ready soup",
    "PICKUP_SOUP": "plate the soup from a ready pot",
    "SERVE_SOUP": "carry the plated soup to a serving window",
    "STASH_HELD_OBJECT": "put an unusable held object on an empty counter",
    "WAIT": "stay put to yield space to your teammate",
}


def _held(player) -> str:
    obj = getattr(player, "held_object", None)
    name = getattr(obj, "name", None) if obj is not None else None
    return name or "nothing"


@dataclass
class DeepSeekHumanAgent:
    """Subgoal-level human partner (player seat) driven by DeepSeek."""

    motion_planner: object
    player_index: int = 1
    replan_every: int = 6
    chat_fn: Callable[[list[dict[str, str]]], str] | None = None
    temperature: float = 0.3

    mdp: object = field(init=False)
    _step: int = field(init=False, default=0)
    _subgoal: str | None = field(init=False, default=None)
    last_say: str = field(init=False, default="")
    last_reason: str = field(init=False, default="")

    def __post_init__(self) -> None:
        self.mdp = self.motion_planner.mdp
        if self.chat_fn is None:
            from durf.group_a.deepseek_chat import chat_once

            self.chat_fn = chat_once

    # -- prompt construction -------------------------------------------------
    def _feasible(self, state) -> list[str]:
        context = context_from_state(state, self.mdp, ai_index=self.player_index)
        return enumerate_feasible_subgoals(context)

    def _describe(self, state, feasible: list[str]) -> str:
        players = state.players
        me = players[self.player_index]
        mate = players[1 - self.player_index]
        pot_states = self.mdp.get_pot_states(state)
        ready = len(self.mdp.get_ready_pots(pot_states))
        cooking = len(self.mdp.get_cooking_pots(pot_states))
        allowed = "\n".join(f"  - {s}: {SUBGOAL_GLOSS.get(s, s)}" for s in feasible)
        return (
            f"You are holding: {_held(me)}.\n"
            f"Your teammate is holding: {_held(mate)}.\n"
            f"Pots: {ready} ready, {cooking} cooking.\n"
            f"Allowed subgoals right now:\n{allowed}\n"
            "Pick the subgoal that helps finish a soup while staying out of your "
            "teammate's way."
        )

    # -- LLM query -----------------------------------------------------------
    @staticmethod
    def _parse(text: str, feasible: list[str]) -> tuple[str | None, str]:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if not match:
            return None, ""
        try:
            data = json.loads(match.group(0))
        except json.JSONDecodeError:
            return None, ""
        subgoal = str(data.get("subgoal", "")).strip().upper()
        say = str(data.get("say", "")).strip()
        if subgoal not in feasible:
            return None, say
        return subgoal, say

    def _query(self, state, feasible: list[str]) -> str:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": self._describe(state, feasible)},
        ]
        try:
            reply = self.chat_fn(messages)
        except Exception as exc:  # network/config error -> rule fallback
            self.last_reason = f"chat_error:{exc}"
            return self._rule_fallback(state, feasible)
        subgoal, say = self._parse(reply, feasible)
        if subgoal is None:
            self.last_reason = "parse_or_infeasible -> rule fallback"
            return self._rule_fallback(state, feasible)
        self.last_say = say
        self.last_reason = "deepseek"
        return subgoal

    def _rule_fallback(self, state, feasible: list[str]) -> str:
        subgoal, _ = rule_teacher_decision(
            state, self.motion_planner, player_index=self.player_index
        )
        return subgoal if subgoal in feasible else feasible[0]

    # -- main entry ----------------------------------------------------------
    def act(self, state) -> tuple[int, str]:
        feasible = self._feasible(state)
        need_replan = (
            self._subgoal is None
            or self._subgoal not in feasible
            or self._step % max(1, self.replan_every) == 0
        )
        if need_replan:
            self._subgoal = self._query(state, feasible)
        self._step += 1
        action = execute_subgoal(
            state, self.motion_planner, self._subgoal, player_index=self.player_index
        )
        return int(action), self._subgoal
