"""Headless sim-human sessions: an AI plays the human role and gives feedback.

Why this exists
---------------
Real human pilot sessions are slow to collect.  This module replaces the
keyboard/chat human in ``play_with_baseline`` with a scripted persona that:

- drives the human player with the same rule teacher used for the AI
  (``rule_teacher_action`` on player index 1), so the human completes the
  task instead of standing in a corner;
- gives feedback about the AI's coordination and task behavior from
  deterministic templates aligned with the attribution vocabulary.

The output format is identical to a real pygame session
(``trajectory.csv``, ``chat_messages.csv``, ``session_metadata.json``), so
``demo_offline_attribution`` and the rest of the pipeline consume sim
sessions unchanged.  Sessions are tagged ``data_source: synthetic_sim_human``
so they can never be confused with real participant data.

AI decision pipeline
--------------------
The task -> recovery -> coordination -> safety chain mirrors
``play_with_baseline``'s closures.  Keep both in sync when changing AI
behavior; the smoke test compares them on a fixed seed/action sequence.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from durf.baseline.collect_rule_teacher_dataset import (
    SUBGOAL_TO_INDEX,
    SUBGOALS,
    choose_task_candidate,
    counters_for_put_down,
    first_action_to_feature,
    make_motion_planner,
    pots_needing_ingredient,
    rule_teacher_candidates,
)
from durf.baseline.coordination import (
    choose_reroute_action,
    CONTINUE_CURRENT_SUBGOAL,
    YIELD,
    CoordinationController,
    build_coordination_candidates,
    path_conflict_type,
)
from durf.baseline.runtime import REPO_ROOT, make_direct_multi_env
from durf.feedback_attribution.condition_features import extract_condition_features
from durf.feedback_attribution.event_detectors import (
    PREP_FETCH_SUBGOALS,
    PREP_SUBGOALS,
    ai_ignoring_pot,
)
from durf.hu.subgoal_reranker import load_runtime_hu, runtime_hu_score, warn_unknown_runtime_subgoal

STAY = 4
INTERACT = 5
ACTION_NAMES = ("north", "south", "east", "west", "stay", "interact")

DEFAULT_LAYOUT = "ring_tomato_onion_10x6_h0_full_task"
DEFAULT_EXECUTOR = (
    REPO_ROOT
    / "outputs"
    / "subgoal_executors"
    / "h0_p0_rule_executor_v5_fast_two_delivery_recovery"
    / "executor.keras"
)

TRAJECTORY_FIELDS = [
    "timestamp_utc",
    "episode",
    "episode_step",
    "total_step",
    "layout",
    "ai_action",
    "ai_action_name",
    "ai_subgoal",
    "ai_condition_features_json",
    "ai_subgoal_candidates_json",
    "task_decision_json",
    "coordination_decision_json",
    "ai_event",
    "human_action",
    "human_action_name",
    "environment_reward",
    "episode_reward",
    "predict_ms",
    "environment_step_ms",
    "done",
    "state_before_json",
    "state_after_json",
]

CHAT_FIELDS = [
    "timestamp_utc",
    "episode",
    "episode_step",
    "total_step",
    "layout",
    "role",
    "content",
]

PAUSE_FIELDS = [
    "timestamp_utc",
    "event",
    "episode",
    "episode_step",
    "total_step",
    "layout",
    "pygame_ticks",
]


# ---------------------------------------------------------------------------
# Feedback templates (deterministic floor; aligned with attribution vocabulary)
# ---------------------------------------------------------------------------

# Coordination-domain feedback.  Keyed by (option_was, polarity) where
# option_was is the AI's actual coordination choice.
_COORD_POS_YIELD = [
    "thanks, i'll squeeze past now",
    "good, you held still so i could get by",
    "nice, that let me through",
]
_COORD_NEG_YIELD = [
    "you didn't need to move, you were one step from finishing",
    "you should have stayed, you were almost there",
]
_COORD_POS_CONTINUE = [
    "right, finish what you were doing",
    "good, keep going, don't mind me",
    "yes, you were closer to your task, stick with it",
]
_COORD_NEG_CONTINUE = [
    "you're blocking me, move",
    "you're in my way, step aside",
    "step aside, i can't pass",
]

# Task-domain feedback.
_TASK_POS_GET_DISH = [
    "soup is ready, go grab a plate",
    "get a dish, it's time to serve",
]
_TASK_POS_PUT_DOWN = [
    "you don't need that, put it down",
    "stop holding that, drop it on a counter",
]
_TASK_POS_GET_INGREDIENT = [
    "we need onions, go get one",
    "get a tomato, the pot is waiting",
]
_TASK_POS_PUT_INGREDIENT = [
    "good, put that into the pot",
    "yes, add it to the pot now",
]
_TASK_POS_SERVE = [
    "great, take it to the serving window",
    "nice, deliver that soup",
]

# Task-domain feedback for missed opportunities.  Text is crafted so the
# keyword matcher scores the intended event only:
#   ready-pot  -> pot/soup/ready/serve/dish  (no "prepare"/"waiting")
#   prep-wait  -> prepare/next/wait/cook     (no "soup"/"dish"/"plate")
_TASK_NEG_READY_POT = [
    "the pot is ready, go get a dish",
    "soup is ready, you should get a dish",
]
_TASK_NEG_PREP_WAIT = [
    "you're waiting empty handed, go get the next ingredient",
    "prepare the next ingredient while the pot cooks",
]


def _pick(templates: list[str], rng: np.random.Generator) -> str:
    return str(rng.choice(templates))


@dataclass
class SimHumanConfig:
    """Persona parameters controlling human behavior and feedback tone."""

    persona: str = "cooperative"  # cooperative | selfish
    feedback_min_gap: int = 15
    seed: int = 0


class SimHuman:
    """Stateful stand-in for the human player.

    Behavior: the rule teacher drives the human's task actions.  When the rule
    teacher is stalled (no path to its target), the persona escapes instead of
    waiting forever:

    - ``selfish`` walks toward the AI and pushes into its tile when adjacent
      (never detours around it), so ``human_entering_ai_tile`` conflicts fire;
    - ``cooperative`` / ``polite`` / ``lenient`` move back toward the task
      (pot) and occasionally bump the AI when it blocks the only way, instead
      of silently deadlocking.

    Feedback: evaluated after each step from the AI's coordination decision
    and condition features, throttled by ``feedback_min_gap``.  Polarity
    follows the persona's blocking tolerance (see ``_evaluate_coordination``):
    cooperative/lenient tolerate persistence when one step from the goal,
    selfish/polite demand passage regardless, lenient never criticizes
    persistence even when far from the goal.
    """

    def __init__(self, config: SimHumanConfig) -> None:
        self.config = config
        self.rng = np.random.default_rng(config.seed)
        self.steps_since_feedback = config.feedback_min_gap + 1
        # Task-event streaks, mirrored from event_detectors thresholds.
        self._ready_pot_streak = 0
        self._prep_wait_streak = 0
        self._task_steps_since_feedback = config.feedback_min_gap + 1

    # -- action selection ---------------------------------------------------

    def choose_action(self, state, motion_planner) -> int:
        candidates = rule_teacher_candidates(state, motion_planner, 1)
        chosen = choose_task_candidate(candidates)
        ai_pos = tuple(state.players[0].position)
        my_pos = tuple(state.players[1].position)
        chosen_targets = [
            tuple(target)
            for target in (chosen.metadata or {}).get("target_positions") or []
            if isinstance(target, (list, tuple)) and len(target) == 2
        ]

        if chosen.subgoal == "WAIT":
            reason = str(chosen.reason or "")
            stalled = reason == "fallback_wait" or reason.startswith("no_path")
            if not stalled:
                # Deliberate wait (e.g. holding a dish while soup cooks).
                return int(chosen.action)
            if _manhattan(my_pos, ai_pos) == 1:
                push = _push_into_ai_action(my_pos, ai_pos)
                if push is not None:
                    if self.config.persona == "selfish":
                        return push
                    if self.rng.random() < 0.3:
                        # Polite bump so the runtime sees the blocked attempt
                        # instead of a silent standoff.
                        return push
            escape_targets = _pot_positions_for_waiting(motion_planner.mdp, state)
            escape = _escape_action(
                motion_planner.mdp,
                my_pos,
                ai_pos,
                escape_targets,
                self.rng,
                toward_other=(self.config.persona == "selfish"),
            )
            if escape is not None:
                return escape
            return int(chosen.action)

        # A selfish human never detours: if the AI stands on the straight line
        # to the task target, approach and push so the conflict fires.  When
        # the AI is 2+ tiles away, step toward it first; the next step pushes
        # into its tile.  This produces conflicts while the AI is still far
        # from its own subgoal target (decoupling 'adjacent to target' from
        # 'blocking the human').
        if self.config.persona == "selfish" and _ai_blocks_target_line(
            my_pos, ai_pos, chosen_targets
        ):
            if _manhattan(my_pos, ai_pos) == 1:
                push = _push_into_ai_action(my_pos, ai_pos)
                if push is not None:
                    return push
            step = _step_toward_action(my_pos, ai_pos)
            if step is not None:
                return step
        return int(chosen.action)

    # -- feedback generation ------------------------------------------------

    def _evaluate_coordination(
        self,
        decision: dict[str, Any],
        features: dict[str, Any],
    ) -> str | None:
        selected = decision.get("selected") if decision else None
        if selected not in (YIELD, CONTINUE_CURRENT_SUBGOAL):
            return None
        human_trying_to_pass = bool(features.get("human_trying_to_pass"))
        ai_adjacent_target = bool(features.get("ai_adjacent_to_current_subgoal_target"))
        if selected == YIELD:
            if human_trying_to_pass:
                return _pick(_COORD_POS_YIELD, self.rng)
            if ai_adjacent_target:
                # The AI stepped off its task one tile short of the finish.
                return _pick(_COORD_NEG_YIELD, self.rng)
            return None
        # AI continued its current subgoal.
        # CONTINUE polarity table: blocking × proximity.  Two orthogonal
        # axes: behavior (polite detour vs selfish push) and polarity
        # (whether the persona tolerates the AI persisting while blocking).
        #   cooperative: tolerates blocking only when one step from the goal
        #   selfish    : never tolerates blocking
        #   polite     : polite behavior, strict polarity (never tolerates)
        #   lenient    : polite behavior, never criticizes persistence
        if human_trying_to_pass:
            if ai_adjacent_target:
                continue_ok = self.config.persona in (
                    "cooperative",
                    "lenient",
                )
            else:
                continue_ok = self.config.persona == "lenient"
            if continue_ok:
                return _pick(_COORD_POS_CONTINUE, self.rng)
            return _pick(_COORD_NEG_CONTINUE, self.rng)
        if ai_adjacent_target:
            # One step from finishing and not blocking: persistence is right.
            return _pick(_COORD_POS_CONTINUE, self.rng)
        return None

    # -- task-event streaks --------------------------------------------------
    # Thresholds mirror event_detectors so every emitted feedback has a real
    # candidate event to match:
    #   AI_ignored_ready_or_nearly_ready_pot      -> streak >= 3
    #   AI_failed_to_prepare_ingredient_while_waiting -> streak >= 4

    def _update_task_streaks(
        self,
        state_after: dict,
        ai_action_name: str,
        ai_subgoal: str | None = None,
        ai_candidate_subgoals: list[str] | None = None,
    ) -> None:
        pot_states = state_after.get("pot_states") or {}
        ai_held = (state_after.get("ai_held_object") or {}).get("name")
        human_held = (state_after.get("human_held_object") or {}).get("name")
        ai_pos = state_after.get("ai_pos")

        ready_positions = _pot_positions(pot_states, ("cooking", "ready"))
        ai_can_pickup_soup = ai_held in {None, "dish"}
        nearest_ready_dist = _nearest_distance(ai_pos, ready_positions)
        ai_handled_pot = (
            ai_action_name == "interact" and nearest_ready_dist in {0, 1}
        )
        # Same rule as event_detectors.ai_ignoring_pot: an AI already walking
        # for a dish / collecting the soup / standing by is not ignoring the
        # pot, and during a mere cooking wait only an idle AI is.  Without it
        # the persona said "the pot is ready, go get a dish" while the pot was
        # still cooking and the AI was fetching the next tomato.
        ignoring = (
            bool(ready_positions)
            and ai_can_pickup_soup
            and not ai_handled_pot
            and ai_ignoring_pot(
                {"ai_subgoal": ai_subgoal},
                pot_is_ready=bool(_pot_positions(pot_states, ("ready",))),
                human_holding=human_held,
            )
        )
        if ignoring:
            self._ready_pot_streak += 1
        else:
            self._ready_pot_streak = 0

        cooking = bool(pot_positions := _pot_positions(pot_states, ("cooking",)))
        ai_picked_up_ingredient = (
            ai_action_name == "interact" and ai_held in {"onion", "tomato"}
        )
        ai_not_prepping = (
            ai_held is None
            and not ai_picked_up_ingredient
            and ai_subgoal not in PREP_SUBGOALS
        )
        # Mirror of event_detectors.prep_option_available: no complaint when
        # the runtime had no ingredient fetch on the table (next batch already
        # staged on counters).
        prep_available = ai_candidate_subgoals is None or any(
            name in PREP_FETCH_SUBGOALS for name in ai_candidate_subgoals
        )
        if cooking and ai_not_prepping and human_held == "dish" and prep_available:
            self._prep_wait_streak += 1
        else:
            self._prep_wait_streak = 0

    def maybe_feedback(
        self,
        *,
        coordination_decision: dict[str, Any],
        condition_features: dict[str, Any],
        state_after: dict[str, Any],
        ai_action_name: str,
        ai_subgoal: str | None = None,
        ai_candidate_subgoals: list[str] | None = None,
    ) -> str | None:
        self.steps_since_feedback += 1
        if self.steps_since_feedback < self.config.feedback_min_gap:
            return None

        self._update_task_streaks(
            state_after, ai_action_name, ai_subgoal, ai_candidate_subgoals
        )
        if self._ready_pot_streak >= 3:
            self._ready_pot_streak = 0
            self.steps_since_feedback = 0
            return _pick(_TASK_NEG_READY_POT, self.rng)
        if self._prep_wait_streak >= 4:
            self._prep_wait_streak = 0
            self.steps_since_feedback = 0
            return _pick(_TASK_NEG_PREP_WAIT, self.rng)

        text = self._evaluate_coordination(coordination_decision, condition_features)
        if text is not None:
            self.steps_since_feedback = 0
            return text
        return None


# ---------------------------------------------------------------------------
# AI decision pipeline (mirror of play_with_baseline closures)
# ---------------------------------------------------------------------------


def _utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def _motion_target(position, action_index: int) -> list[int]:
    action_name = ACTION_NAMES[int(action_index)]
    if action_name == "north":
        return [position[0], position[1] - 1]
    if action_name == "south":
        return [position[0], position[1] + 1]
    if action_name == "east":
        return [position[0] + 1, position[1]]
    if action_name == "west":
        return [position[0] - 1, position[1]]
    return list(position)


def _action_moves(action_index: int) -> bool:
    return ACTION_NAMES[int(action_index)] in {"north", "south", "east", "west"}


def _manhattan(a, b) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def _pot_positions(pot_states: dict, keys: tuple[str, ...]) -> list[tuple[int, int]]:
    positions: list[tuple[int, int]] = []
    for key in keys:
        positions.extend(tuple(position) for position in pot_states.get(key, []) or [])
    return positions


def _nearest_distance(pos, positions) -> int | None:
    if not pos or not positions:
        return None
    return min(_manhattan(tuple(pos), position) for position in positions)


def _push_into_ai_action(my_pos, ai_pos) -> int | None:
    """Action that attempts to move into the AI's tile, if adjacent."""
    if _manhattan(my_pos, ai_pos) != 1:
        return None
    for direction in range(4):
        if tuple(_motion_target(my_pos, direction)) == tuple(ai_pos):
            return direction
    return None


def _step_toward_action(my_pos, other_pos) -> int | None:
    """Walkable direction that reduces Manhattan distance to the other agent.

    Used to turn a ``ai_blocks_target_line`` situation with the AI 2+ tiles
    away into an adjacent push on the next step, so coordination conflicts can
    fire while the AI is still far from its own subgoal target.
    """
    best_action = None
    best_distance = _manhattan(my_pos, other_pos)
    for direction in range(4):
        candidate = tuple(_motion_target(my_pos, direction))
        distance = _manhattan(candidate, other_pos)
        if distance < best_distance:
            best_action = direction
            best_distance = distance
    return best_action


def _ai_blocks_target_line(
    my_pos: tuple[int, int],
    ai_pos: tuple[int, int],
    target_positions: list[tuple[int, int]],
) -> bool:
    """The AI stands on the first two steps toward the nearest task target."""
    if not target_positions:
        return False
    goal = min(
        target_positions,
        key=lambda target: _manhattan(my_pos, tuple(target)),
    )
    dx = goal[0] - my_pos[0]
    dy = goal[1] - my_pos[1]
    if abs(dx) >= abs(dy) and dx != 0:
        step = 1 if dx > 0 else -1
        ahead = (my_pos[0] + step, my_pos[1])
        ahead2 = (my_pos[0] + 2 * step, my_pos[1])
    elif dy != 0:
        step = 1 if dy > 0 else -1
        ahead = (my_pos[0], my_pos[1] + step)
        ahead2 = (my_pos[0], my_pos[1] + 2 * step)
    else:
        return False
    return tuple(ai_pos) in {ahead, ahead2}


def _escape_action(
    mdp,
    my_pos,
    other_pos,
    target_positions: list[tuple[int, int]],
    rng: np.random.Generator,
    *,
    toward_other: bool,
) -> int | None:
    """Walkable detour for a stalled agent (never into the partner's tile).

    With task targets, prefer the neighbor closest to the nearest target.
    Without targets, approach the partner (selfish) or flee it (cooperative).
    Random jitter breaks symmetric mirrors where both agents would otherwise
    bounce between the same two tiles forever.
    """
    valid = set(mdp.get_valid_player_positions())
    neighbors: list[tuple[int, tuple[int, int]]] = []
    for direction in range(4):
        target = tuple(_motion_target(my_pos, direction))
        if target in valid and target != tuple(other_pos):
            neighbors.append((direction, target))
    if not neighbors:
        return None
    if target_positions:
        scored = [
            (
                min(_manhattan(target, goal) for goal in target_positions),
                rng.random(),
                direction,
            )
            for direction, target in neighbors
        ]
        return min(scored)[2]
    scored = [
        (
            (-1 if toward_other else 1) * _manhattan(target, other_pos),
            rng.random(),
            direction,
        )
        for direction, target in neighbors
    ]
    return max(scored)[2]


def _choose_yield_action(mdp, ai_pos, human_pos, human_target) -> int | None:
    valid_positions = set(mdp.get_valid_player_positions())
    avoid = {tuple(human_pos), tuple(human_target)}
    best_action = None
    best_score = None
    for action_index, action_name in enumerate(ACTION_NAMES[:4]):
        candidate = _motion_target(ai_pos, action_index)
        candidate_tuple = tuple(candidate)
        if candidate_tuple not in valid_positions or candidate_tuple in avoid:
            continue
        score = (
            abs(candidate[0] - human_pos[0])
            + abs(candidate[1] - human_pos[1])
            + abs(candidate[0] - human_target[0])
            + abs(candidate[1] - human_target[1])
        )
        if best_score is None or score > best_score:
            best_action = action_index
            best_score = score
    return best_action


def _pot_positions_for_waiting(mdp, state) -> list[tuple[int, int]]:
    pot_states = mdp.get_pot_states(state)
    positions: list[tuple[int, int]] = []
    for key in ("ready", "cooking"):
        positions.extend(tuple(pos) for pos in pot_states.get(key, []) or [])
    if positions:
        return positions
    return [tuple(pos) for pos in mdp.get_pot_locations()]


def _detect_subgoal_issue(state, subgoal_name: str, mdp) -> str:
    # AI_HELD_DISH_BEFORE_SOUP_READY and AI_HELD_UNNEEDED_INGREDIENT used to
    # live here and be force-corrected downstream in _recovery_action_override,
    # unconditionally overriding whatever the task/Hu layer had already
    # chosen. Both states are now covered by real, scored candidates in
    # generate_candidate_subgoals (WAIT_NEAR_POT / PUT_DOWN_OBJECT / WAIT),
    # reproducing the exact same H0 action at hu_task_tolerance=0 -- so Hu
    # can finally have a say here instead of being silently overruled (same
    # change as play_with_baseline.py). Only a genuinely stale committed
    # choice (the world changed since this subgoal was picked) still needs a
    # post-hoc recovery step.
    if subgoal_name in ("PUT_TOMATO_IN_POT", "PUT_ONION_IN_POT"):
        ingredient = "tomato" if subgoal_name == "PUT_TOMATO_IN_POT" else "onion"
        if not pots_needing_ingredient(state, mdp, ingredient):
            return "STALE_PUT_INGREDIENT_SUBGOAL"
    return ""


def _put_down_unneeded_object_action(state, motion_planner) -> int:
    action = first_action_to_feature(
        motion_planner,
        state.players[0],
        counters_for_put_down(state, motion_planner.mdp, motion_planner, state.players[0]),
        {state.players[1].position},
    )
    return int(action) if action is not None else STAY


def _recovery_action_override(
    proposed_ai_action: int,
    subgoal_name: str,
    state,
    motion_planner,
) -> tuple[int, str]:
    issue = _detect_subgoal_issue(
        state,
        subgoal_name,
        motion_planner.mdp,
    )
    if not issue:
        return proposed_ai_action, ""
    return _put_down_unneeded_object_action(state, motion_planner), issue


class AiRuntime:
    """Stateful mirror of play_with_baseline's AI decision chain."""

    def __init__(
        self,
        *,
        env,
        motion_planner,
        subgoal_model=None,
        hu_model=None,
        hu_task_tolerance: float = 0.0,
        hu_coordination_lambda: float = 0.0,
        hu_apply: bool = False,
        stubborn_prob: float = 0.0,
        stalled_escape: bool = False,
        stall_escape_delay: int = 8,
        seed: int = 0,
    ) -> None:
        self.env = env
        self.motion_planner = motion_planner
        self.subgoal_model = subgoal_model
        self.hu_model = hu_model
        self.hu_task_tolerance = hu_task_tolerance
        self.hu_coordination_lambda = hu_coordination_lambda
        self.hu_apply = hu_apply
        self.coordination = CoordinationController(
            min_commit_steps=1,
            max_option_steps=3,
            yield_cooldown_steps=2,
        )
        self.rng = np.random.default_rng(seed + 1)
        self.stubborn_prob = stubborn_prob
        self.stalled_escape = stalled_escape
        self.stall_escape_delay = stall_escape_delay
        self.stall_steps = 0
        # Last movement direction of the human, kept across stationary steps
        # so "AI on human path" stays a geometric fact while "human trying to
        # pass" requires an active movement intent this step.
        self._last_human_delta: tuple[int, int] | None = None

    def reset_episode(self) -> None:
        self.coordination.reset()
        self.stall_steps = 0
        self._last_human_delta = None

    def step(
        self,
        *,
        human_action: int,
        episode: int,
        total_step: int,
        hu_user_id: str,
    ) -> dict[str, Any]:
        state = self.env.base_env.state
        candidates = rule_teacher_candidates(
            state,
            self.motion_planner,
            0,
        )
        facts = _state_facts(self.env)
        condition_features = extract_condition_features(
            {
                "state_facts": facts,
                "extra": {"state_before": facts},
                "human_action_name": ACTION_NAMES[int(human_action)],
            }
        )
        if self.hu_model is not None:
            for candidate in candidates:
                try:
                    candidate.hu_score = runtime_hu_score(
                        self.hu_model,
                        "task",
                        hu_user_id,
                        condition_features,
                        candidate.subgoal,
                    )
                except KeyError:
                    candidate.hu_score = 0.0
                    warn_unknown_runtime_subgoal("task", candidate.subgoal)
        chosen = choose_task_candidate(
            candidates,
            task_tolerance=self.hu_task_tolerance if self.hu_apply else 0.0,
        )
        subgoal_name = chosen.subgoal
        planner_action = chosen.action
        target_positions = [
            tuple(position)
            for position in chosen.metadata.get("target_positions") or []
            if isinstance(position, (list, tuple)) and len(position) == 2
        ]
        ai_pos = tuple(self.env.base_env.state.players[0].position)
        condition_features["ai_current_subgoal"] = subgoal_name
        condition_features["ai_adjacent_to_current_subgoal_target"] = any(
            _manhattan(ai_pos, position) == 1
            for position in target_positions
        )
        serialized_candidates = [
            {
                "subgoal": candidate.subgoal,
                "task_score": candidate.task_score,
                "hu_score": candidate.hu_score,
                "final_score": candidate.final_score,
                "reason": candidate.reason,
                "feasible": candidate.feasible,
                "metadata": candidate.metadata,
            }
            for candidate in candidates
        ]
        task_decision = {
            "record_type": "runtime_decision",
            "decision_id": f"task:e{episode}:t{total_step + 1}",
            "decision_level": "task",
            "current_timestep": total_step + 1,
            "condition_at_decision": condition_features,
            "candidate_set": [
                candidate.get("subgoal")
                for candidate in serialized_candidates
            ],
            "candidates": serialized_candidates,
            "selected": subgoal_name,
            "selected_action": int(planner_action),
            "hu_applied": bool(self.hu_apply and self.hu_task_tolerance > 0.0),
            "hu_task_tolerance": self.hu_task_tolerance,
        }

        # Every subgoal the generator can produce is executed by the motion
        # planner.  The learned executor used to sit behind this point as a
        # fallback, but the whitelist it guarded had grown to cover the whole
        # vocabulary, so the network was never called in any recorded session
        # (same path as play_with_baseline).

        recovered_action, recovery_event = _recovery_action_override(
            int(planner_action),
            subgoal_name,
            state,
            self.motion_planner,
        )

        human_pos = tuple(state.players[1].position)
        human_target = tuple(_motion_target(human_pos, human_action))
        ai_target = tuple(_motion_target(ai_pos, recovered_action))
        conflict_type = path_conflict_type(
            ai_pos=ai_pos,
            human_pos=human_pos,
            ai_target=ai_target,
            human_target=human_target,
            ai_is_moving=_action_moves(recovered_action),
            human_is_moving=_action_moves(human_action),
        )
        coordination_event = ""
        current_coordination_decision: dict[str, Any] = {}
        coordinated_action = recovered_action
        if conflict_type is not None:
            condition_features = dict(condition_features)
            human_delta = None
            if _action_moves(human_action):
                human_delta = (
                    human_target[0] - human_pos[0],
                    human_target[1] - human_pos[1],
                )
                self._last_human_delta = human_delta
            # Geometric blocking: the AI occupies the human's next tile or the
            # one after it along the last known movement direction.  This holds
            # even while the human is stationary, decoupling "on the path"
            # (a spatial fact) from "trying to pass" (an active intent).
            if human_delta is not None:
                on_path = human_target == ai_pos or (
                    human_target[0] + human_delta[0],
                    human_target[1] + human_delta[1],
                ) == ai_pos
            elif self._last_human_delta is not None:
                projected = (
                    human_pos[0] + self._last_human_delta[0],
                    human_pos[1] + self._last_human_delta[1],
                )
                on_path = projected == ai_pos or (
                    projected[0] + self._last_human_delta[0],
                    projected[1] + self._last_human_delta[1],
                ) == ai_pos
            else:
                on_path = False
            condition_features["ai_on_human_path"] = on_path
            condition_features["human_trying_to_pass"] = (
                human_delta is not None and on_path
            )
            # WAIT / BACK_OFF / REROUTE are execution-level refinements of a
            # single Hu-scored YIELD decision, not options Hu chooses between:
            # REROUTE (keep making task progress on a path that avoids the
            # human) beats BACK_OFF (retreat to the most separating open
            # tile) beats WAIT (stay put) when the stronger options aren't
            # available.  This applies to every conflict type (same rule as
            # play_with_baseline's coordination_action_decision).
            reroute_action = choose_reroute_action(
                self.motion_planner,
                self.env.base_env.state.players[0],
                target_positions,
                human_pos,
                human_target,
                recovered_action,
            )
            if reroute_action is not None:
                yield_action = reroute_action
                yield_mode = "reroute"
            else:
                back_off_action = _choose_yield_action(
                    self.motion_planner.mdp,
                    ai_pos,
                    human_pos,
                    human_target,
                )
                if back_off_action is not None:
                    yield_action = back_off_action
                    yield_mode = "back_off"
                else:
                    yield_action = None
                    yield_mode = "wait"
            coordination_candidates = build_coordination_candidates(
                proposed_action=recovered_action,
                yield_action=yield_action,
                stay_action=STAY,
                conflict_type=conflict_type,
                ai_adjacent_to_current_subgoal_target=bool(
                    condition_features.get(
                        "ai_adjacent_to_current_subgoal_target"
                    )
                ),
                yield_mode=yield_mode,
            )
            stubborn = (
                self.stubborn_prob > 0.0
                and self.rng.random() < self.stubborn_prob
            )
            if stubborn:
                # Simulated "opinionated partner": force the AI to hold its
                # task subgoal so CONTINUE>YIELD scenarios can be collected.
                stubborn_decision = {
                    "record_type": "runtime_decision",
                    "decision_id": (
                        f"coordination:e{episode}:t{total_step + 1}"
                    ),
                    "decision_level": "coordination",
                    "current_timestep": total_step + 1,
                    "condition_at_decision": condition_features,
                    "candidate_set": [candidate.option for candidate in coordination_candidates],
                    "candidates": [
                        candidate.to_dict()
                        for candidate in coordination_candidates
                    ],
                    "selected": CONTINUE_CURRENT_SUBGOAL,
                    "selected_action": int(recovered_action),
                    "hu_applied": False,
                    "sim_stubborn_override": True,
                }
                current_coordination_decision = stubborn_decision
                coordination_event = "sim_stubborn_override"
            else:
                def coordination_hu_score(option: str) -> float:
                    if self.hu_model is None:
                        return 0.0
                    return runtime_hu_score(
                        self.hu_model,
                        "coordination",
                        hu_user_id,
                        condition_features,
                        option,
                    )

                result = self.coordination.resolve(
                    timestep=total_step + 1,
                    episode=episode,
                    task_subgoal=subgoal_name,
                    conflict_type=conflict_type,
                    ai_pos=ai_pos,
                    human_pos=human_pos,
                    candidates=coordination_candidates,
                    condition_features=condition_features,
                    hu_score=(
                        coordination_hu_score
                        if self.hu_model is not None
                        else None
                    ),
                    hu_lambda=self.hu_coordination_lambda,
                    apply_hu=self.hu_apply,
                )
                if result is not None:
                    coordinated_action = result.action
                    coordination_event = result.event
                    current_coordination_decision = result.decision

        ai_action = int(coordinated_action)
        if not 0 <= ai_action < len(ACTION_NAMES):
            ai_action = STAY
            safety_event = "AI_SAFETY_REJECTED_INVALID_ACTION"
        else:
            safety_event = ""
        # Stalled-plan escape: an empty-handed WAIT with no coordination or
        # recovery action means the rule teacher has no path at all (typically
        # the partner blocks its only target).  Waiting a few steps models a
        # polite partner, then it clears the way instead of deadlocking.
        if self.stalled_escape and not safety_event:
            ai_held = getattr(state.players[0], "held_object", None)
            stalled = (
                subgoal_name == "WAIT"
                and ai_held is None
                and ai_action == STAY
                and not current_coordination_decision
                and not recovery_event
            )
            if stalled:
                self.stall_steps += 1
                if self.stall_steps >= self.stall_escape_delay:
                    escape = _escape_action(
                        self.motion_planner.mdp,
                        ai_pos,
                        human_pos,
                        _pot_positions_for_waiting(
                            self.motion_planner.mdp,
                            state,
                        ),
                        self.rng,
                        toward_other=False,
                    )
                    if escape is not None:
                        ai_action = escape
                        self.stall_steps = 0
                        coordination_event = "sim_stalled_escape"
            else:
                self.stall_steps = 0
        current_ai_event = (
            safety_event or coordination_event or recovery_event
        )
        return {
            "action": ai_action,
            "subgoal": subgoal_name,
            "subgoal_candidates": serialized_candidates,
            "condition_features": condition_features,
            "task_decision": task_decision,
            "coordination_decision": current_coordination_decision,
            "event": current_ai_event,
        }


# ---------------------------------------------------------------------------
# Session writer
# ---------------------------------------------------------------------------


def _state_facts(env) -> dict[str, Any]:
    state = env.base_env.state
    mdp = env.base_env.mdp
    players = list(getattr(state, "players", []))
    ai_player = players[0] if len(players) > 0 else None
    human_player = players[1] if len(players) > 1 else None
    objects = getattr(state, "objects", {})

    def player_summary(player):
        if player is None:
            return None
        return {
            "position": list(player.position),
            "orientation": list(player.orientation),
            "held_object": object_summary(getattr(player, "held_object", None)),
        }

    def object_summary(obj):
        if obj is None:
            return None
        if hasattr(obj, "name"):
            return {"name": obj.name}
        return None

    def pot_state_summary():
        if not hasattr(mdp, "get_pot_states"):
            return None
        try:
            raw = mdp.get_pot_states(state)
            return {
                str(key): [list(position) for position in values]
                for key, values in raw.items()
            }
        except Exception:
            return None

    return {
        "ai_pos": list(ai_player.position) if ai_player else None,
        "human_pos": list(human_player.position) if human_player else None,
        "ai_held_object": object_summary(
            getattr(ai_player, "held_object", None)
        ),
        "human_held_object": object_summary(
            getattr(human_player, "held_object", None)
        ),
        "players": [
            player_summary(player) for player in players if player is not None
        ],
        "objects": [
            {
                "position": list(position),
                "object": object_summary(obj),
            }
            for position, obj in getattr(objects, "items", lambda: [])()
        ],
        "pot_states": pot_state_summary(),
        "layout_features": {
            "layout_name": getattr(env, "layout_name", None),
            "terrain": ["".join(row) for row in getattr(mdp, "terrain_mtx", [])],
        },
    }


def _json_dumps(value) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def _write_row(writer: csv.DictWriter, handle, row: dict) -> None:
    writer.writerow(row)
    handle.flush()


def _new_session_dir(output_dir: str | Path) -> Path:
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    root = Path(output_dir).resolve()
    session_dir = root / run_id
    # Second-resolution ids collide when sessions are launched back to back
    # (batch sweeps in one process); suffix instead of failing.
    suffix = 1
    while session_dir.exists():
        session_dir = root / f"{run_id}_{suffix:02d}"
        suffix += 1
    session_dir.mkdir(parents=True, exist_ok=False)
    return session_dir


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--layout", default=DEFAULT_LAYOUT)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--horizon", type=int, default=800)
    parser.add_argument("--episodes", type=int, default=2)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument(
        "--subgoal-executor",
        type=Path,
        default=DEFAULT_EXECUTOR,
        help="Keras executor used for subgoals outside the rule teacher set.",
    )
    parser.add_argument(
        "--load-retired-executor",
        action="store_true",
        help=(
            "Load the retired learned subgoal executor. Nothing calls it -- the "
            "motion planner executes every subgoal -- so this exists only for "
            "reproducing older runs."
        ),
    )
    parser.add_argument(
        "--sim-human",
        choices=("cooperative", "selfish", "polite", "lenient"),
        default="cooperative",
        help=(
            "Persona playing the human role. Behavior axis: selfish pushes "
            "into the AI, the others detour politely. Polarity axis: "
            "cooperative/lenient tolerate persistence when one step from the "
            "goal; selfish/polite demand passage; lenient never criticizes "
            "persistence."
        ),
    )
    parser.add_argument("--sim-seed", type=int, default=0)
    parser.add_argument(
        "--sim-feedback-min-gap",
        type=int,
        default=15,
        help="Minimum steps between two human feedback messages.",
    )
    parser.add_argument(
        "--sim-ai-stubborn-prob",
        type=float,
        default=0.0,
        help=(
            "Probability per conflict that the AI holds its task subgoal "
            "instead of yielding (creates CONTINUE>YIELD scenarios)."
        ),
    )
    parser.add_argument(
        "--sim-stall-escape-delay",
        type=int,
        default=8,
        help=(
            "Steps an empty-handed WAIT AI stays before escaping a stalled "
            "plan (clears chokepoints the rule teacher cannot route around)."
        ),
    )
    parser.add_argument("--hu-user-id", default="PILOT01")
    parser.add_argument(
        "--hu-model",
        type=Path,
        default=None,
        help="Frozen Hu_general or protocol-v2 PerUserAdapter JSON.",
    )
    parser.add_argument("--hu-task-tolerance", type=float, default=0.0)
    parser.add_argument("--hu-coordination-lambda", type=float, default=0.0)
    parser.add_argument("--hu-apply", action="store_true")
    parser.add_argument("--output-dir", default=str(REPO_ROOT / "outputs" / "human_ai_sessions"))
    return parser.parse_args()


def run_sim_session(args: argparse.Namespace) -> Path:
    if not 0.0 <= args.sim_ai_stubborn_prob <= 1.0:
        raise ValueError("--sim-ai-stubborn-prob must be in [0, 1]")

    # The learned executor is retired: every subgoal is executed by the motion
    # planner.  The file is only loaded when explicitly asked for, so a run no
    # longer pays the TensorFlow import cost for a model nothing calls.
    subgoal_model = None
    if args.load_retired_executor and args.subgoal_executor and args.subgoal_executor.exists():
        import tensorflow as tf

        subgoal_model = tf.keras.models.load_model(args.subgoal_executor)

    hu_model = load_runtime_hu(args.hu_model) if args.hu_model else None
    if args.hu_task_tolerance < 0 or args.hu_coordination_lambda < 0:
        raise ValueError("Hu lambdas cannot be negative")

    env = make_direct_multi_env(args.layout, args.seed, horizon=args.horizon)
    motion_planner = make_motion_planner(args.layout, args.seed, args.horizon)
    ai_runtime = AiRuntime(
        env=env,
        motion_planner=motion_planner,
        subgoal_model=subgoal_model,
        hu_model=hu_model,
        hu_task_tolerance=args.hu_task_tolerance,
        hu_coordination_lambda=args.hu_coordination_lambda,
        hu_apply=args.hu_apply,
        stubborn_prob=args.sim_ai_stubborn_prob,
        stalled_escape=True,
        stall_escape_delay=args.sim_stall_escape_delay,
        seed=args.sim_seed,
    )
    sim_human = SimHuman(
        SimHumanConfig(
            persona=args.sim_human,
            feedback_min_gap=args.sim_feedback_min_gap,
            seed=args.sim_seed,
        )
    )

    session_dir = _new_session_dir(args.output_dir)
    trajectory_path = session_dir / "trajectory.csv"
    chat_path = session_dir / "chat_messages.csv"
    pause_path = session_dir / "pause_events.csv"
    metadata_path = session_dir / "session_metadata.json"
    metadata_path.write_text(
        json.dumps(
            {
                "build_id": "sim-human-v1",
                "agent": None,
                "ai_mode": "subgoal_executor",
                "subgoal_executor": str(args.subgoal_executor),
                "layout": args.layout,
                "seed": args.seed,
                "horizon": args.horizon,
                "human_player_index": 1,
                "ai_player_index": 0,
                "hu_user_id": args.hu_user_id,
                "hu_model": str(args.hu_model) if args.hu_model else None,
                "hu_task_tolerance": args.hu_task_tolerance,
                "hu_coordination_lambda": args.hu_coordination_lambda,
                "hu_apply": args.hu_apply,
                "data_source": "synthetic_sim_human",
                "sim_human": {
                    "persona": args.sim_human,
                    "seed": args.sim_seed,
                    "feedback_min_gap": args.sim_feedback_min_gap,
                },
                "sim_ai_stubborn_prob": args.sim_ai_stubborn_prob,
                "sim_stall_escape_delay": args.sim_stall_escape_delay,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    trajectory_handle = trajectory_path.open("w", newline="", encoding="utf-8")
    chat_handle = chat_path.open("w", newline="", encoding="utf-8")
    pause_handle = pause_path.open("w", newline="", encoding="utf-8")
    trajectory_writer = csv.DictWriter(trajectory_handle, fieldnames=TRAJECTORY_FIELDS)
    chat_writer = csv.DictWriter(chat_handle, fieldnames=CHAT_FIELDS)
    pause_writer = csv.DictWriter(pause_handle, fieldnames=PAUSE_FIELDS)
    trajectory_writer.writeheader()
    chat_writer.writeheader()
    pause_writer.writeheader()

    ai_obs, _ = env.multi_reset()
    episode = 1
    episode_step = 0
    total_step = 0
    episode_reward = 0.0

    try:
        while True:
            human_action = sim_human.choose_action(
                env.base_env.state,
                motion_planner,
            )
            decision = ai_runtime.step(
                human_action=human_action,
                episode=episode,
                total_step=total_step,
                hu_user_id=args.hu_user_id,
            )
            state_before = _state_facts(env)
            (ai_obs, _), (reward, _), done, _ = env.multi_step(
                decision["action"],
                human_action,
            )
            state_after = _state_facts(env)
            episode_step += 1
            total_step += 1
            episode_reward += float(reward)

            _write_row(
                trajectory_writer,
                trajectory_handle,
                {
                    "timestamp_utc": _utc_timestamp(),
                    "episode": episode,
                    "episode_step": episode_step,
                    "total_step": total_step,
                    "layout": args.layout,
                    "ai_action": decision["action"],
                    "ai_action_name": ACTION_NAMES[decision["action"]],
                    "ai_subgoal": decision["subgoal"],
                    "ai_condition_features_json": _json_dumps(
                        decision["condition_features"]
                    ),
                    "ai_subgoal_candidates_json": _json_dumps(
                        decision["subgoal_candidates"]
                    ),
                    "task_decision_json": _json_dumps(decision["task_decision"]),
                    "coordination_decision_json": _json_dumps(
                        decision["coordination_decision"]
                    ),
                    "ai_event": decision["event"],
                    "human_action": human_action,
                    "human_action_name": ACTION_NAMES[human_action],
                    "environment_reward": float(reward),
                    "episode_reward": episode_reward,
                    "predict_ms": 0.0,
                    "environment_step_ms": 0.0,
                    "done": bool(done),
                    "state_before_json": _json_dumps(state_before),
                    "state_after_json": _json_dumps(state_after),
                },
            )

            feedback_text = sim_human.maybe_feedback(
                coordination_decision=decision["coordination_decision"],
                condition_features=decision["condition_features"],
                state_after=state_after,
                ai_action_name=ACTION_NAMES[decision["action"]],
                ai_subgoal=decision["subgoal"],
                ai_candidate_subgoals=[
                    candidate.get("subgoal")
                    for candidate in decision.get("subgoal_candidates") or []
                ],
            )
            if feedback_text:
                _write_row(
                    chat_writer,
                    chat_handle,
                    {
                        "timestamp_utc": _utc_timestamp(),
                        "episode": episode,
                        "episode_step": episode_step,
                        "total_step": total_step,
                        "layout": args.layout,
                        "role": "user",
                        "content": feedback_text,
                    },
                )

            if args.max_steps is not None and total_step >= args.max_steps:
                print(f"stopped at max steps ({total_step})")
                break
            if done:
                print(
                    f"Episode {episode}: reward={episode_reward:.1f}, "
                    f"steps={episode_step}"
                )
                if episode >= args.episodes:
                    break
                episode += 1
                episode_step = 0
                episode_reward = 0.0
                ai_obs, _ = env.multi_reset()
                ai_runtime.reset_episode()
    finally:
        trajectory_handle.close()
        chat_handle.close()
        pause_handle.close()
        env.close()

    print(f"Session: {session_dir}")
    return session_dir


def main() -> int:
    args = parse_args()
    run_sim_session(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
