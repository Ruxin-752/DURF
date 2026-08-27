"""Live H0 + learned-comfort subgoal agent (Path A).

The direct realization of "adapt to H0/subgoal": at every step the agent lets
H0 propose the task-valid feasible subgoals, re-ranks them with the reward
learned from language feedback, and executes the chosen subgoal through H0's
motion planner. Comfort therefore drives *which subgoal* is pursued -- not just
how smoothly a fixed subgoal is executed.

    state --context_from_state--> SubgoalContext
          --enumerate_feasible_subgoals--> feasible (H0 safety floor)
          --plan_subgoal(w, ...)--> chosen subgoal (comfort reranked)
          --execute_subgoal--> action (H0 motion planner)

Four explicit feedback modes are available: frozen weights, paper-style Route
1 Literal, Route 1 PseudoPragmatic, and Route 2 neural reward inference. Route
2 treats the ten-model mean as a Gaussian reward observation by default, as in
the original ensemble agent; an explicit legacy EMA option remains available.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from collections import Counter
from pathlib import Path

from overcooked_ai_py.mdp.actions import Action

from durf.baseline.comfort_reward import ADAPTED_ROOT, context_from_state
from durf.baseline.h0_planner import execute_subgoal, plan_h0_subgoal

from src.subgoal_planner import enumerate_feasible_subgoals, plan_subgoal  # noqa: E402
from src.feature_schema import load_features  # noqa: E402
from src.route1_online import OnlineRoute1Learner, ROUTE1_MODES  # noqa: E402
from src.subgoal_featurizer import SubgoalContext  # noqa: E402
from src.trajectory_featurizer import (  # noqa: E402
    featurize_trajectory_events,
    featurize_trajectory_steps,
)

STAY = 4
FEEDBACK_MODES = ("frozen", "route1-literal", "route1-pseudopragmatic", "route2")
DEFAULT_WEIGHTS_PATH = ADAPTED_ROOT / "outputs" / "route2" / "learned_comfort_weights.json"
GOLD_WEIGHTS_PATH = ADAPTED_ROOT / "data" / "gold_comfort_weights.json"
PAPER_CV_ENSEMBLE_PATH = (
    ADAPTED_ROOT
    / "outputs"
    / "route2"
    / "paper_aligned_v5_seed137_selected"
    / "ensemble_manifest.json"
)
PAPER_ALIGNED_MODEL_PATH = (
    ADAPTED_ROOT
    / "outputs"
    / "route2"
    / "paper_aligned_proactivity_v4"
    / "profile_multiseed"
    / "model.pt"
)
HUMAN_FEEDBACK_MODEL_PATH = (
    ADAPTED_ROOT
    / "outputs"
    / "route2"
    / "paper_aligned_proactivity_v4"
    / "local_only_multiseed"
    / "model.pt"
)
LEGACY_PAPER_ALIGNED_MODEL_PATH = (
    ADAPTED_ROOT
    / "outputs"
    / "route2"
    / "paper_aligned_36configs"
    / "multiseed_adam"
    / "model.pt"
)
LEGACY_MODEL_PATH = ADAPTED_ROOT / "outputs" / "route2" / "model.pt"
DEFAULT_MODEL_PATH = next(
    (
        path
        for path in (
            PAPER_CV_ENSEMBLE_PATH,
            HUMAN_FEEDBACK_MODEL_PATH,
            PAPER_ALIGNED_MODEL_PATH,
            LEGACY_PAPER_ALIGNED_MODEL_PATH,
            LEGACY_MODEL_PATH,
        )
        if path.exists()
    ),
    LEGACY_MODEL_PATH,
)


def load_comfort_weights(path: str | Path | None = None) -> dict[str, float]:
    """Load learned comfort weights, falling back to the gold teacher weights."""

    import json

    candidate = Path(path) if path else DEFAULT_WEIGHTS_PATH
    if not candidate.exists():
        candidate = GOLD_WEIGHTS_PATH
    with candidate.open("r", encoding="utf-8") as handle:
        raw = json.load(handle)
    return {str(k): float(v) for k, v in raw.items()}


class ComfortSubgoalAgent:
    """Choose subgoals with the learned reward; execute them through H0."""

    def __init__(
        self,
        motion_planner,
        *,
        weights: dict[str, float] | None = None,
        weights_path: str | Path | None = None,
        lambda_pref: float = 1.0,
        ai_index: int = 0,
        model_path: str | Path | None = None,
        online_blend: float | None = None,
        route2_observation_precision: float = 2.0,
        route2_human_observation_precision: float | None = None,
        require_reference_gate: bool = False,
        feedback_mode: str = "route2",
        route1_prior: str = "zero",
        route1_lookback: int = 25,
        human_feedback_precision: float = 4.0,
        learner_state_path: str | Path | None = None,
        resume_learner_state: bool = False,
        max_consecutive_wait: int = 3,
        subgoal_commitment_steps: int = 2,
        switch_margin: float = 0.05,
        deadlock_patience: int = 2,
        allow_policy_overrides: bool = False,
    ) -> None:
        if feedback_mode not in FEEDBACK_MODES:
            raise ValueError(f"Unknown feedback mode {feedback_mode!r}; choose from {FEEDBACK_MODES}")
        if abs(float(lambda_pref) - 1.0) > 1e-12:
            raise ValueError(
                "paper-aligned live decisions require lambda_pref=1.0; "
                "change reward weights through language feedback instead"
            )
        if route1_prior not in {"zero", "frozen"}:
            raise ValueError("route1_prior must be 'zero' or 'frozen'")
        if route1_lookback <= 0:
            raise ValueError("route1_lookback must be positive")
        if human_feedback_precision <= 0:
            raise ValueError("human_feedback_precision must be positive")
        if route2_observation_precision <= 0:
            raise ValueError("route2_observation_precision must be positive")
        if (
            route2_human_observation_precision is not None
            and route2_human_observation_precision <= 0
        ):
            raise ValueError("route2_human_observation_precision must be positive")
        if online_blend is not None and not 0.0 <= float(online_blend) <= 1.0:
            raise ValueError("online_blend must be between 0 and 1")
        if max_consecutive_wait < 0:
            raise ValueError("max_consecutive_wait must be non-negative")
        if subgoal_commitment_steps < 0:
            raise ValueError("subgoal_commitment_steps must be non-negative")
        if switch_margin < 0:
            raise ValueError("switch_margin must be non-negative")
        if deadlock_patience < 1:
            raise ValueError("deadlock_patience must be positive")
        self.motion_planner = motion_planner
        self.mdp = motion_planner.mdp
        if weights is not None:
            self.weights = {str(key): float(value) for key, value in weights.items()}
        elif feedback_mode == "route2" and weights_path is None:
            # A neutral prior leaves behavior with H0 until language supplies w.
            self.weights = {feature: 0.0 for feature in load_features()}
        else:
            self.weights = load_comfort_weights(weights_path)
        frozen_weights = dict(self.weights)
        self.lambda_pref = float(lambda_pref)
        self.ai_index = int(ai_index)
        self.model_path = Path(model_path) if model_path else DEFAULT_MODEL_PATH
        self.online_blend = None if online_blend is None else float(online_blend)
        self.route2_observation_precision = float(route2_observation_precision)
        # The paper's controlled observations use precision 2. Live human
        # language can opt into an explicitly calibrated, lower precision
        # without changing the Gaussian update rule or the formal paper mode.
        self.route2_human_observation_precision = (
            None
            if route2_human_observation_precision is None
            else float(route2_human_observation_precision)
        )
        # The original Route 2 applies every feedback item directly. Reference
        # classification is therefore audit-only unless an explicit ablation
        # requests phrase filtering.
        self.require_reference_gate = bool(require_reference_gate)
        self.feedback_mode = feedback_mode
        self.route1_prior = route1_prior
        self.route1_lookback = int(route1_lookback)
        self.human_feedback_precision = float(human_feedback_precision)
        self.learner_state_path = (
            Path(learner_state_path) if learner_state_path is not None else None
        )
        self.max_consecutive_wait = int(max_consecutive_wait)
        self.subgoal_commitment_steps = int(subgoal_commitment_steps)
        self.switch_margin = float(switch_margin)
        self.deadlock_patience = int(deadlock_patience)
        # Disabled by default: these legacy engineering fallbacks can override
        # w·phi and therefore are not part of the paper-aligned live policy.
        self.allow_policy_overrides = bool(allow_policy_overrides)
        self.consecutive_waits = 0
        self.committed_subgoal: str | None = None
        self.commitment_age = 0
        self._last_state_signature: str | None = None
        self._last_executed_action: int | None = None
        self._last_executed_subgoal: str | None = None
        self._last_task_progress_signature: str | None = None
        self._stalled_wait_steps = 0
        self._unchanged_state_steps = 0
        self._deadlock_detected = False
        self._recent_motion_signatures: list[tuple] = []
        self._recent_executed_subgoals: list[str] = []
        self._motion_cycle_steps = 0
        # A cross-subgoal A-B-A-B loop means that the instantaneous w·phi
        # winner is changing with the agent's position.  Once detected, keep
        # pursuing the winner selected at the detection state until real task
        # progress occurs (or the target becomes infeasible).  This is an
        # execution-consistency constraint, not a language-to-action rule.
        self._oscillation_commitment_subgoal: str | None = None
        self._oscillation_commitment_task_signature: str | None = None
        self._oscillation_commitment_release_reason: str | None = None
        self._last_observed_task_signature: str | None = None
        self._task_progressed_since_last_action = False
        self._route2 = None  # lazy single-checkpoint or CV-ensemble predictor
        self._route2_precision = {
            feature: 1.0 / 25.0 for feature in load_features()
        }
        self._route1 = None
        self._route1_prior_mean = (
            frozen_weights if feedback_mode in ROUTE1_MODES and route1_prior == "frozen" else None
        )
        if feedback_mode in ROUTE1_MODES:
            self._route1 = OnlineRoute1Learner(
                load_features(),
                mode=feedback_mode,
                prior_mean=self._route1_prior_mean,
                source_precision={"human_live": self.human_feedback_precision},
            )
            self.weights = self._route1.weights()
        self.base_weights = dict(self.weights)
        self._base_route2_precision = dict(self._route2_precision)
        self.last_decision: dict | None = None
        self.decision_history: list[dict] = []
        self.trajectory_history: list[dict] = []
        self.feedback_history: list[dict] = []
        self.last_feedback_update: str = ""
        # Event IDs, unlike text, are safe idempotency keys. Identical wording
        # with a new event ID remains an independent precision-2 observation.
        self._seen_feedback_event_ids: set[str] = set()
        self._processed_feedback_event_ids: set[str] = set()
        if resume_learner_state:
            if self.learner_state_path is None:
                raise ValueError("resume_learner_state requires learner_state_path")
            self.load_learner_state(self.learner_state_path)

    @staticmethod
    def _state_signature(state) -> str:
        """Stable signature of actual player/object state, excluding timestep."""

        players = []
        for player in getattr(state, "players", ()):
            held = getattr(player, "held_object", None)
            players.append(
                (
                    tuple(getattr(player, "position", ())),
                    tuple(getattr(player, "orientation", ())),
                    getattr(held, "name", None),
                )
            )
        objects = []
        for position, obj in sorted(getattr(state, "objects", {}).items()):
            objects.append(
                (
                    tuple(position),
                    getattr(obj, "name", None),
                    tuple(
                        getattr(ingredient, "name", ingredient)
                        for ingredient in getattr(obj, "ingredients", ())
                    ),
                    bool(getattr(obj, "is_cooking", False)),
                    bool(getattr(obj, "is_ready", False)),
                )
            )
        return repr((players, objects))

    @staticmethod
    def _task_progress_signature(state) -> str:
        """Task progress, excluding harmless player motion and orientation."""

        held_objects = tuple(
            getattr(getattr(player, "held_object", None), "name", None)
            for player in getattr(state, "players", ())
        )
        objects = []
        for position, obj in sorted(getattr(state, "objects", {}).items()):
            objects.append(
                (
                    tuple(position),
                    getattr(obj, "name", None),
                    tuple(
                        getattr(ingredient, "name", ingredient)
                        for ingredient in getattr(obj, "ingredients", ())
                    ),
                    bool(getattr(obj, "is_cooking", False)),
                    bool(getattr(obj, "is_ready", False)),
                )
            )
        return repr((held_objects, objects))

    def _observe_execution_progress(self, state) -> None:
        signature = self._state_signature(state)
        unchanged = (
            self._last_state_signature is not None
            and signature == self._last_state_signature
            and self._last_executed_action is not None
        )
        self._unchanged_state_steps = self._unchanged_state_steps + 1 if unchanged else 0
        self._deadlock_detected = self._unchanged_state_steps >= self.deadlock_patience

        task_signature = self._task_progress_signature(state)
        self._task_progressed_since_last_action = bool(
            self._last_observed_task_signature is not None
            and task_signature != self._last_observed_task_signature
        )
        if (
            self._oscillation_commitment_subgoal is not None
            and self._oscillation_commitment_task_signature is not None
            and task_signature != self._oscillation_commitment_task_signature
        ):
            self._oscillation_commitment_subgoal = None
            self._oscillation_commitment_task_signature = None
            self._oscillation_commitment_release_reason = "task_progress"
        stalled_after_wait = (
            self._last_task_progress_signature is not None
            and task_signature == self._last_task_progress_signature
            and self._last_executed_subgoal == "WAIT"
            and self._last_executed_action == STAY
        )
        self._stalled_wait_steps = (
            self._stalled_wait_steps + 1 if stalled_after_wait else 0
        )
        players = list(getattr(state, "players", ()))
        ai = players[self.ai_index] if len(players) > self.ai_index else None
        held = getattr(ai, "held_object", None) if ai is not None else None
        motion_signature = (
            tuple(getattr(ai, "position", ())),
            tuple(getattr(ai, "orientation", ())),
            getattr(held, "name", None),
        )
        in_two_step_cycle = bool(
            len(self._recent_motion_signatures) >= 2
            and motion_signature == self._recent_motion_signatures[-2]
            and self._last_executed_subgoal not in {None, "WAIT"}
            and self._last_executed_action not in {None, STAY}
            and self._last_task_progress_signature is not None
            and task_signature == self._last_task_progress_signature
        )
        self._motion_cycle_steps = (
            self._motion_cycle_steps + 1 if in_two_step_cycle else 0
        )
        self._recent_motion_signatures.append(motion_signature)
        del self._recent_motion_signatures[:-4]

    def _agent_blocks_human_target_access(self, state, context) -> bool:
        """Conservative live check used only to deny a WAIT exemption."""

        explicit = getattr(context, "agent_blocks_human_target_access", None)
        if explicit is not None:
            return bool(explicit)
        wait_effect = getattr(context, "candidate_path_effects", {}).get("WAIT")
        if wait_effect in {"blocks", "blocks_human_path", "worsens"}:
            return True

        players = list(getattr(state, "players", ()))
        if len(players) < 2:
            return False
        human_index = 1 - self.ai_index if len(players) == 2 else (self.ai_index + 1) % len(players)
        ai_position = tuple(getattr(players[self.ai_index], "position", ()))
        human_holding = getattr(
            getattr(players[human_index], "held_object", None), "name", None
        )
        if human_holding in {"tomato", "onion", "dish"}:
            targets = list(self.mdp.get_pot_locations())
        elif human_holding == "soup":
            targets = list(self.mdp.get_serving_locations())
        else:
            return False
        valid = set(self.mdp.get_valid_player_positions())
        for target in targets:
            tx, ty = target
            access_tiles = {
                (tx + dx, ty + dy)
                for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1))
            } & valid
            if ai_position in access_tiles:
                return True
        return False

    def _passive_cooking_wait_is_valid(self, state, context, h0_subgoal: str) -> bool:
        """True when the current recipe is complete and only needs cook time."""

        context = SubgoalContext.coerce(context)
        held = context.agent_holding
        recipe = Counter(context.recipe)
        cooking_complete = any(
            snapshot["status"] == "cooking"
            and all(
                Counter(snapshot["ingredients"])[name] >= count
                for name, count in recipe.items()
            )
            for snapshot in context.normalized_pot_snapshots
        )
        return bool(
            context.soup_cooking
            and cooking_complete
            and not context.soup_ready
            and not context.has_open_recipe_work
            # A prefetched dish is the one held object that remains directly
            # useful when cooking completes. STASH is H0's feasibility fallback
            # until then, but learned w·phi may legitimately prefer to wait.
            and held in {None, "dish"}
            and (
                h0_subgoal == "WAIT"
                or (held == "dish" and h0_subgoal == "STASH_HELD_OBJECT")
            )
            and not self._agent_blocks_human_target_access(state, context)
        )

    def _candidate_next_actions(
        self,
        state,
        feasible: list[str],
        *,
        yield_action: int | None = None,
    ) -> dict[str, int]:
        """Audit the concrete next action for every live candidate."""

        next_actions = {}
        for subgoal in feasible:
            if subgoal == "WAIT":
                next_actions[subgoal] = STAY
            elif subgoal == "YIELD_PATH" and yield_action is not None:
                next_actions[subgoal] = int(yield_action)
            else:
                next_actions[subgoal] = int(
                    execute_subgoal(
                        state,
                        self.motion_planner,
                        subgoal,
                        player_index=self.ai_index,
                    )
                )
        return next_actions

    def _joint_motion_conflict(
        self, state, ai_action: int, human_action: int | None
    ) -> bool:
        """Whether the proposed joint motion is guaranteed to collide."""

        if human_action is None:
            return False
        players = list(getattr(state, "players", ()))
        if len(players) < 2:
            return False
        human_index = (
            1 - self.ai_index
            if len(players) == 2
            else (self.ai_index + 1) % len(players)
        )
        try:
            ai_direction = Action.INDEX_TO_ACTION[int(ai_action)]
            human_direction = Action.INDEX_TO_ACTION[int(human_action)]
        except (IndexError, KeyError, TypeError, ValueError):
            return False
        ai_position = tuple(players[self.ai_index].position)
        human_position = tuple(players[human_index].position)
        ai_moves = ai_direction in Action.MOTION_ACTIONS and ai_direction != Action.STAY
        human_moves = (
            human_direction in Action.MOTION_ACTIONS
            and human_direction != Action.STAY
        )
        ai_target = (
            Action.move_in_direction(ai_position, ai_direction)
            if ai_moves
            else ai_position
        )
        human_target = (
            Action.move_in_direction(human_position, human_direction)
            if human_moves
            else human_position
        )
        valid = set(self.mdp.get_valid_player_positions())
        if ai_moves and ai_target not in valid:
            ai_moves = False
            ai_target = ai_position
        if human_moves and human_target not in valid:
            human_moves = False
            human_target = human_position
        same_destination = ai_target == human_target and (ai_moves or human_moves)
        swaps_positions = (
            ai_moves
            and human_moves
            and ai_target == human_position
            and human_target == ai_position
        )
        return bool(same_destination or swaps_positions)

    def _joint_collision_yield_action(
        self, state, human_action: int | None
    ) -> int | None:
        """Find a generic legal step away from the teammate's current motion."""

        if human_action is None:
            return None
        players = list(getattr(state, "players", ()))
        if len(players) < 2:
            return None
        human_index = (
            1 - self.ai_index
            if len(players) == 2
            else (self.ai_index + 1) % len(players)
        )
        try:
            human_direction = Action.INDEX_TO_ACTION[int(human_action)]
        except (IndexError, KeyError, TypeError, ValueError):
            return None
        if human_direction not in Action.MOTION_ACTIONS or human_direction == Action.STAY:
            return None
        ai_position = tuple(players[self.ai_index].position)
        human_position = tuple(players[human_index].position)
        human_target = Action.move_in_direction(human_position, human_direction)
        valid = set(self.mdp.get_valid_player_positions())
        if human_target not in valid:
            return None
        occupied = {
            tuple(player.position)
            for index, player in enumerate(players)
            if index != self.ai_index
        }
        options: list[tuple[tuple[int, int, int], int]] = []
        for order, direction in enumerate(Action.MOTION_ACTIONS):
            if direction == Action.STAY:
                continue
            destination = Action.move_in_direction(ai_position, direction)
            if (
                destination not in valid
                or destination in occupied
                or destination == human_target
            ):
                continue
            action_index = int(Action.ACTION_TO_INDEX[direction])
            if self._joint_motion_conflict(state, action_index, human_action):
                continue
            distance = abs(destination[0] - human_target[0]) + abs(
                destination[1] - human_target[1]
            )
            onward = sum(
                Action.move_in_direction(destination, nxt) in valid
                and Action.move_in_direction(destination, nxt) not in occupied
                for nxt in Action.MOTION_ACTIONS
                if nxt != Action.STAY
            )
            options.append(((-distance, -onward, order), action_index))
        return min(options, key=lambda item: item[0])[1] if options else None

    def _stalled_motion_subgoal(self, state) -> str | None:
        """A goal that has been pursued without task progress for too long.

        Moving back and forth can change the full state every step while still
        making no task progress. Once this happens, the committed subgoal is
        temporarily removed from the feasible action set. This is an execution
        feasibility constraint; all remaining candidates are still ranked only
        by w dot phi.
        """

        if self._motion_cycle_steps < self.deadlock_patience:
            return None
        return self._last_executed_subgoal

    @staticmethod
    def _score_for(decision: dict, subgoal: str) -> float | None:
        item = next(
            (row for row in decision.get("ranking", []) if row.get("subgoal") == subgoal),
            None,
        )
        return float(item["total_score"]) if item is not None else None

    def _stabilize_decision(self, decision: dict, *, force_switch: bool = False) -> dict:
        raw_choice = decision["chosen_subgoal"]
        current = self.committed_subgoal
        feasible = decision.get("feasible_subgoals", [])
        reason = "raw_argmax"

        if force_switch:
            alternatives = [
                row["subgoal"]
                for row in decision.get("ranking", [])
                if row["subgoal"] != "WAIT" and row["subgoal"] != current
            ]
            if alternatives:
                decision["chosen_subgoal"] = alternatives[0]
                reason = "deadlock_nonwait_fallback"
            elif raw_choice == "WAIT":
                nonwait = [name for name in feasible if name != "WAIT"]
                if nonwait:
                    decision["chosen_subgoal"] = nonwait[0]
                    reason = "deadlock_nonwait_fallback"
        elif current in feasible and raw_choice != current:
            current_score = self._score_for(decision, current)
            challenger_score = self._score_for(decision, raw_choice)
            if self.commitment_age < self.subgoal_commitment_steps:
                decision["chosen_subgoal"] = current
                reason = "subgoal_commitment"
            elif (
                current_score is not None
                and challenger_score is not None
                and challenger_score < current_score + self.switch_margin
            ):
                decision["chosen_subgoal"] = current
                reason = "switch_hysteresis"

        decision["raw_chosen_subgoal"] = raw_choice
        decision["policy_switch"] = (
            current is not None and decision["chosen_subgoal"] != current
        )
        decision["stability_reason"] = reason
        decision["commitment_age"] = self.commitment_age
        return decision

    def plan(self, state, *, human_action: int | None = None) -> dict:
        """Return the full re-ranking decision for the current state."""

        context = context_from_state(state, self.mdp, ai_index=self.ai_index)
        feasible = enumerate_feasible_subgoals(context)
        # YIELD_PATH is a live geometry candidate, not a learned language-to-
        # action rule.  It enters the set only when the teammate is adjacent,
        # faces the AI, and the executor found a legal one-step escape.
        path_effects = (
            context.get("candidate_path_effects", {})
            if isinstance(context, dict)
            else context.candidate_path_effects
        )
        if (
            path_effects.get("YIELD_PATH") == "clears"
            and "YIELD_PATH" not in feasible
        ):
            wait_index = feasible.index("WAIT") if "WAIT" in feasible else len(feasible)
            feasible.insert(wait_index, "YIELD_PATH")
        candidate_next_actions = self._candidate_next_actions(state, feasible)
        initial_joint_conflicts = [
            subgoal
            for subgoal, action in candidate_next_actions.items()
            if self._joint_motion_conflict(state, action, human_action)
        ]
        joint_yield_action = (
            self._joint_collision_yield_action(state, human_action)
            if initial_joint_conflicts
            else None
        )
        if joint_yield_action is not None:
            if isinstance(context, dict):
                effects = context.setdefault("candidate_path_effects", {})
            else:
                effects = context.candidate_path_effects
            effects["YIELD_PATH"] = "clears"
            if "YIELD_PATH" not in feasible:
                wait_index = feasible.index("WAIT") if "WAIT" in feasible else len(feasible)
                feasible.insert(wait_index, "YIELD_PATH")
            candidate_next_actions = self._candidate_next_actions(
                state, feasible, yield_action=joint_yield_action
            )
        joint_conflicts = [
            subgoal
            for subgoal, action in candidate_next_actions.items()
            if self._joint_motion_conflict(state, action, human_action)
        ]
        # Joint collision is a physical infeasibility. Keep WAIT only when no
        # collision-free candidate exists; otherwise w·phi ranks the remaining
        # executable candidates, including the generic YIELD_PATH option.
        collision_free = [
            subgoal for subgoal in feasible if subgoal not in joint_conflicts
        ]
        joint_motion_filter_applied = bool(joint_conflicts and collision_free)
        if joint_motion_filter_applied:
            feasible = collision_free

        preconstraint_feasible = list(feasible)
        h0_tie_fallback = plan_h0_subgoal(state, self.mdp, player_index=self.ai_index)
        stalled_motion_subgoal = self._stalled_motion_subgoal(state)
        recent_cycle_subgoals = list(
            dict.fromkeys(
                self._recent_executed_subgoals[
                    -(max(1, self.deadlock_patience) + 1) :
                ]
            )
        )
        cross_subgoal_motion_cycle = bool(
            stalled_motion_subgoal is not None
            and len(recent_cycle_subgoals) > 1
        )
        # Removing only the most recent target in a cross-target cycle forces
        # the other target to win, and then forces the first one on the next
        # step.  That was the GET_DISH <-> GET_TOMATO hesitation seen in the
        # live log.  Preserve the old removal rule only for a single-goal
        # executor loop; cross-goal loops use a stable w·phi commitment below.
        removable_stalled_subgoal = (
            None if cross_subgoal_motion_cycle else stalled_motion_subgoal
        )
        stalled_motion_constraint_applied = bool(
            removable_stalled_subgoal in feasible
            and any(
                subgoal != removable_stalled_subgoal
                for subgoal in feasible
            )
        )
        if stalled_motion_constraint_applied:
            feasible = [
                subgoal
                for subgoal in feasible
                if subgoal != removable_stalled_subgoal
            ]
        executable_productive = [
            subgoal
            for subgoal in feasible
            if subgoal != "WAIT" and candidate_next_actions[subgoal] != STAY
        ]
        motion_unexecutable = [
            subgoal
            for subgoal in feasible
            if subgoal != "WAIT" and candidate_next_actions[subgoal] == STAY
        ]
        # Motion feasibility is part of the live action set, not a reward
        # preference. A non-WAIT subgoal whose concrete executor can only STAY
        # cannot compete in argmax(w dot phi) this step. WAIT remains available
        # when every productive candidate is currently blocked/unreachable.
        if "WAIT" in feasible:
            feasible = [
                subgoal
                for subgoal in feasible
                if subgoal == "WAIT" or subgoal in executable_productive
            ]
        elif executable_productive:
            feasible = list(executable_productive)

        precommitment_feasible = list(feasible)
        oscillation_commitment_applied = False
        active_commitment = self._oscillation_commitment_subgoal
        if active_commitment is not None:
            if (
                active_commitment in feasible
                and candidate_next_actions.get(active_commitment, STAY) != STAY
            ):
                feasible = [active_commitment]
                executable_productive = [active_commitment]
                oscillation_commitment_applied = True
            else:
                self._oscillation_commitment_subgoal = None
                self._oscillation_commitment_task_signature = None
                self._oscillation_commitment_release_reason = (
                    "target_unavailable_or_unexecutable"
                )
                active_commitment = None
        passive_cooking_wait = self._passive_cooking_wait_is_valid(
            state, context, h0_tie_fallback
        )
        stalled_wait_threshold_reached = (
            self.max_consecutive_wait > 0
            and self.consecutive_waits >= self.max_consecutive_wait
            and self._stalled_wait_steps >= self.max_consecutive_wait
            and self._last_executed_subgoal == "WAIT"
            and self._last_executed_action == STAY
        )
        liveness_constraint_applied = bool(
            stalled_wait_threshold_reached
            and executable_productive
            and not passive_cooking_wait
            and "WAIT" in feasible
        )
        if liveness_constraint_applied:
            feasible = [subgoal for subgoal in feasible if subgoal != "WAIT"]
            liveness_reason = "stalled_wait_infeasible"
        elif passive_cooking_wait and stalled_wait_threshold_reached:
            liveness_reason = "passive_cooking_wait_exempt"
        elif stalled_wait_threshold_reached and not executable_productive:
            liveness_reason = "no_executable_productive_alternative"
        elif self.max_consecutive_wait <= 0:
            liveness_reason = "constraint_disabled"
        else:
            liveness_reason = "stall_threshold_not_reached"
        decision = plan_subgoal(
            self.weights,
            context,
            lambda_pref=self.lambda_pref,
            feasible_subgoals=feasible,
            tie_fallback=h0_tie_fallback,
        )
        option_uncommitted_choice = decision["chosen_subgoal"]
        option_commitment_triggered = False
        if (
            active_commitment is None
            and self._last_executed_subgoal not in {None, "WAIT"}
            and self._last_executed_subgoal in feasible
            and candidate_next_actions.get(self._last_executed_subgoal, STAY) != STAY
            and not self._task_progressed_since_last_action
        ):
            # Subgoals are options: after w·phi chooses one, pursue it until
            # task progress or infeasibility instead of greedily changing the
            # target after each navigation step.  The first target still comes
            # solely from the learned reward argmax.
            active_commitment = self._last_executed_subgoal
            self._oscillation_commitment_subgoal = active_commitment
            self._oscillation_commitment_task_signature = (
                self._task_progress_signature(state)
            )
            option_commitment_triggered = True
            if decision["chosen_subgoal"] != active_commitment:
                decision = plan_subgoal(
                    self.weights,
                    context,
                    lambda_pref=self.lambda_pref,
                    feasible_subgoals=[active_commitment],
                    tie_fallback=h0_tie_fallback,
                )
                oscillation_commitment_applied = True
        oscillation_commitment_triggered = False
        if (
            active_commitment is None
            and cross_subgoal_motion_cycle
            and decision["chosen_subgoal"] != "WAIT"
        ):
            # The lock target is the current reward argmax.  No object/action
            # name is privileged here; feedback can change which target wins.
            self._oscillation_commitment_subgoal = decision["chosen_subgoal"]
            self._oscillation_commitment_task_signature = (
                self._task_progress_signature(state)
            )
            active_commitment = decision["chosen_subgoal"]
            oscillation_commitment_triggered = True
        if self.allow_policy_overrides:
            decision = self._stabilize_decision(
                decision, force_switch=self._deadlock_detected
            )
        else:
            decision["raw_chosen_subgoal"] = decision["chosen_subgoal"]
            decision["policy_switch"] = (
                self.committed_subgoal is not None
                and decision["chosen_subgoal"] != self.committed_subgoal
            )
            decision["stability_reason"] = decision["decision_source"]
            decision["commitment_age"] = self.commitment_age
        if oscillation_commitment_applied:
            decision["stability_reason"] = "subgoal_option_commitment"
        decision["context"] = context
        decision["h0_tie_fallback"] = h0_tie_fallback
        decision["policy_overrides_enabled"] = self.allow_policy_overrides
        # This is a feasibility constraint, not an additive reward or a
        # hand-authored tie-break: every retained candidate is still ranked by
        # exactly w dot phi.
        decision["liveness_constraint_applied"] = liveness_constraint_applied
        decision["liveness_reason"] = liveness_reason
        decision["liveness_removed_subgoals"] = (
            ["WAIT"] if liveness_constraint_applied else []
        )
        decision["preconstraint_feasible_subgoals"] = preconstraint_feasible
        decision["candidate_next_actions"] = candidate_next_actions
        decision["human_action_for_joint_feasibility"] = human_action
        decision["joint_motion_conflicts"] = joint_conflicts
        decision["joint_motion_filter_applied"] = joint_motion_filter_applied
        decision["joint_collision_yield_action"] = joint_yield_action
        decision["stalled_motion_constraint_applied"] = (
            stalled_motion_constraint_applied
        )
        decision["stalled_motion_removed_subgoal"] = (
            removable_stalled_subgoal
            if decision["stalled_motion_constraint_applied"]
            else None
        )
        decision["motion_cycle_steps"] = self._motion_cycle_steps
        decision["cross_subgoal_motion_cycle"] = cross_subgoal_motion_cycle
        decision["recent_cycle_subgoals"] = recent_cycle_subgoals
        decision["oscillation_commitment_applied"] = (
            oscillation_commitment_applied
        )
        decision["oscillation_commitment_triggered"] = (
            oscillation_commitment_triggered
        )
        decision["option_commitment_triggered"] = option_commitment_triggered
        decision["option_uncommitted_chosen_subgoal"] = option_uncommitted_choice
        decision["oscillation_commitment_subgoal"] = active_commitment
        decision["oscillation_commitment_release_reason"] = (
            self._oscillation_commitment_release_reason
        )
        decision["precommitment_feasible_subgoals"] = precommitment_feasible
        self._oscillation_commitment_release_reason = None
        decision["motion_feasibility_filter_applied"] = bool(motion_unexecutable)
        decision["motion_unexecutable_subgoals"] = motion_unexecutable
        decision["executable_productive_subgoals"] = executable_productive
        decision["stalled_wait_steps"] = self._stalled_wait_steps
        decision["passive_cooking_wait_exempt"] = passive_cooking_wait
        # Backward-compatible audit fields.
        decision["wait_guard_applied"] = liveness_constraint_applied
        decision["productive_alternative_available"] = bool(executable_productive)
        decision["deadlock_detected"] = self._deadlock_detected
        decision["unchanged_state_steps"] = self._unchanged_state_steps
        self.last_decision = decision
        self.decision_history.append(decision)
        del self.decision_history[:-self.route1_lookback]
        return decision

    def act(self, state, *, human_action: int | None = None) -> tuple[int, str]:
        """Return ``(action_index, chosen_subgoal)`` for the AI player."""

        self._observe_execution_progress(state)
        decision = self.plan(state, human_action=human_action)
        subgoal = decision["chosen_subgoal"]
        action = decision.get("candidate_next_actions", {}).get(subgoal)
        if action is None:
            action = execute_subgoal(
                state, self.motion_planner, subgoal, player_index=self.ai_index
            )
        if (
            self.allow_policy_overrides
            and (self._deadlock_detected or (subgoal == "WAIT" and decision["wait_guard_applied"]))
            and (subgoal == "WAIT" or int(action) == STAY)
        ):
            for row in decision.get("ranking", []):
                fallback = row["subgoal"]
                if fallback in {"WAIT", subgoal}:
                    continue
                fallback_action = execute_subgoal(
                    state, self.motion_planner, fallback, player_index=self.ai_index
                )
                subgoal, action = fallback, fallback_action
                decision["chosen_subgoal"] = fallback
                decision["stability_reason"] = "executable_nonwait_fallback"
                if int(action) != STAY:
                    break
        decision["policy_switch"] = (
            self.committed_subgoal is not None
            and subgoal != self.committed_subgoal
        )
        if subgoal == self.committed_subgoal:
            self.commitment_age += 1
        else:
            self.committed_subgoal = subgoal
            self.commitment_age = 1
        self._last_state_signature = self._state_signature(state)
        self._last_task_progress_signature = self._task_progress_signature(state)
        self._last_observed_task_signature = self._last_task_progress_signature
        self._last_executed_action = int(action)
        self._last_executed_subgoal = subgoal
        self._recent_executed_subgoals.append(subgoal)
        del self._recent_executed_subgoals[:-4]
        decision["executed_action"] = int(action)
        self.consecutive_waits = self.consecutive_waits + 1 if subgoal == "WAIT" else 0
        return int(action), subgoal

    def _ensure_route2(self):
        if self._route2 is not None:
            return self._route2
        if not self.model_path.exists():
            raise FileNotFoundError(f"Route 2 checkpoint not found: {self.model_path}")
        from src.neural_inference import load_predictor

        self._route2 = load_predictor(self.model_path)
        return self._route2

    def _decision_after_update(self) -> str | None:
        if not self.last_decision:
            return None
        context = self.last_decision.get("context")
        feasible = list(
            self.last_decision.get("precommitment_feasible_subgoals") or []
        )
        if not feasible:
            feasible = [
                item["subgoal"] for item in self.last_decision.get("ranking", [])
            ]
        if context is None or not feasible:
            return None
        decision = plan_subgoal(
            self.weights,
            context,
            lambda_pref=self.lambda_pref,
            feasible_subgoals=feasible,
            tie_fallback=self.last_decision.get("h0_tie_fallback"),
        )
        if self.allow_policy_overrides:
            decision = self._stabilize_decision(decision)
        return decision["chosen_subgoal"]

    @staticmethod
    def _annotate_policy_switch(trace: dict) -> None:
        accepted = trace.get("status") == "updated"
        before = trace.get("before_subgoal")
        after = trace.get("after_subgoal")
        trace["policy_switch"] = bool(accepted and before != after)
        trace["accepted_but_no_policy_switch"] = bool(
            accepted and before is not None and before == after
        )

    @staticmethod
    def _route2_reference_gate(text: str) -> dict:
        """Audit phrase references and prepare an optional filtering ablation.

        This audit (or explicit ablation gate) is not evidence that the
        classifier is accurate on real human language. Every raw prediction and
        its calibrated threshold remains in the trace for later auditing.
        """

        from src.phrase_reference_classifier import classify_utterance

        try:
            predictions = classify_utterance(text)
        except Exception as exc:  # classifier failure must never update reward
            return {
                "accepted": False,
                "reason": "classifier_error",
                "predictions": [],
                "error_type": type(exc).__name__,
                "error": str(exc),
                "accepted_phrases": [],
                "grounded_text": "",
                "policy": "at_least_one_non_other_confident_phrase",
                "human_language_accuracy_claim": None,
            }

        audited: list[dict] = []
        accepted_phrases: list[str] = []
        rejection_reasons: list[dict] = []
        for index, raw in enumerate(predictions):
            prediction = dict(raw)
            label = str(prediction.get("reference_type") or "")
            confidence = float(prediction.get("confidence", 0.0))
            threshold_raw = prediction.get("threshold")
            threshold = float(threshold_raw) if threshold_raw is not None else None
            abstained = bool(prediction.get("abstained", False))
            prediction["reference_type"] = label
            prediction["confidence"] = confidence
            prediction["threshold"] = threshold
            prediction["abstained"] = abstained
            rejection_reason = None
            if label == "other":
                rejection_reason = "reference_type_other"
            elif threshold is None:
                rejection_reason = "missing_confidence_threshold"
            elif abstained:
                rejection_reason = "classifier_abstained"
            elif confidence < threshold:
                rejection_reason = "below_class_confidence_threshold"
            phrase = str(prediction.get("phrase") or "").strip()
            if rejection_reason is None and phrase:
                accepted_phrases.append(phrase)
                prediction["gate_accepted"] = True
                prediction["gate_rejection_reason"] = None
            else:
                rejection_reason = rejection_reason or "empty_phrase"
                rejection_reasons.append(
                    {
                        "phrase_index": index,
                        "reason": rejection_reason,
                    }
                )
                prediction["gate_accepted"] = False
                prediction["gate_rejection_reason"] = rejection_reason
            audited.append(prediction)

        if not audited:
            rejection_reasons.append(
                {"phrase_index": None, "reason": "no_phrase_prediction"}
            )
        accepted = bool(accepted_phrases)
        if accepted:
            primary_reason = None
        elif any(
                item["reason"] == "reference_type_other"
                for item in rejection_reasons
        ):
            primary_reason = "reference_type_other"
        else:
            primary_reason = (
                rejection_reasons[0]["reason"] if rejection_reasons else None
            )
        return {
            "accepted": accepted,
            "reason": primary_reason,
            "rejection_reasons": rejection_reasons,
            "predictions": audited,
            "accepted_phrases": accepted_phrases,
            "grounded_text": " ".join(accepted_phrases),
            "policy": "at_least_one_non_other_confident_phrase",
            "human_language_accuracy_claim": None,
        }

    def _recent_trajectory_features(self) -> tuple[dict[str, float], str]:
        """Ground evaluative feedback to recent executed steps, with a plan fallback."""

        executed = featurize_trajectory_steps(self.trajectory_history)
        if executed:
            return executed, "executed_steps"

        merged: dict[str, float] = {}
        for decision in self.decision_history:
            chosen = decision.get("chosen_subgoal")
            selected = next(
                (
                    item
                    for item in decision.get("ranking", [])
                    if item.get("subgoal") == chosen
                ),
                None,
            )
            for feature, value in (selected or {}).get("features", {}).items():
                merged[str(feature)] = merged.get(str(feature), 0.0) + float(value)
        return merged, "selected_subgoals"

    def _recent_trajectory_events(self) -> list[dict]:
        """Per-step Route 1 context; Route 2 continues using aggregate counts."""

        return featurize_trajectory_events(self.trajectory_history)

    def record_transition(
        self,
        *,
        state_before: dict,
        state_after: dict,
        ai_action_name: str,
        human_action_name: str,
        environment_reward: float,
    ) -> None:
        """Append one normalized live step for paper-style trajectory grounding."""

        self.trajectory_history.append(
            {
                "total_step": (
                    int(self.trajectory_history[-1].get("total_step", 0)) + 1
                    if self.trajectory_history
                    else 1
                ),
                "ai_action_name": ai_action_name,
                "human_action_name": human_action_name,
                "environment_reward": float(environment_reward),
                "state_facts": state_after,
                "extra": {"state_before": state_before},
            }
        )
        del self.trajectory_history[:-self.route1_lookback]

    def update_from_feedback(
        self,
        text: str,
        *,
        feedback_event_id: str | None = None,
        blend: float | None = None,
        oracle_feedback: dict | None = None,
        feedback_form_prediction: dict | None = None,
        interpretation: str = "inferred",
        source: str = "human_live",
        confidence: float = 1.0,
    ) -> dict:
        """Update the selected Route 1/Route 2 learner and return an audit trace."""

        cleaned = (text or "").strip()
        if not cleaned:
            return {"status": "empty", "mode": self.feedback_mode, "text": ""}

        before_subgoal = (
            self.last_decision.get("chosen_subgoal") if self.last_decision else None
        )
        if self.feedback_mode == "frozen":
            trace = {
                "status": "ignored",
                "mode": "frozen",
                "text": cleaned,
                "before_subgoal": before_subgoal,
                "after_subgoal": before_subgoal,
            }
            self.feedback_history.append(trace)
            self.last_feedback_update = cleaned
            return trace

        if self.feedback_mode in ROUTE1_MODES:
            if self._route1 is None:
                raise RuntimeError("Route 1 learner was not initialized")
            trajectory_features, trajectory_source = self._recent_trajectory_features()
            recent_events = self._recent_trajectory_events()
            trace = self._route1.update(
                cleaned,
                decision=self.last_decision,
                trajectory_features=trajectory_features,
                recent_events=recent_events,
                total_step=(
                    int(recent_events[-1]["total_step"]) if recent_events else None
                ),
                oracle_feedback=oracle_feedback,
                feedback_form_prediction=feedback_form_prediction,
                interpretation=interpretation,
                source=source,
                confidence=confidence,
            )
            trace["trajectory_window"] = (
                len(self.trajectory_history)
                if self.trajectory_history
                else len(self.decision_history)
            )
            trace["trajectory_source"] = trajectory_source
            self.weights = self._route1.weights()
            trace["before_subgoal"] = before_subgoal
            if trace.get("status") == "updated":
                self._oscillation_commitment_subgoal = None
                self._oscillation_commitment_task_signature = None
                self._recent_motion_signatures.clear()
                self._recent_executed_subgoals.clear()
                self._motion_cycle_steps = 0
                self._last_executed_subgoal = None
            trace["after_subgoal"] = self._decision_after_update()
            self._annotate_policy_switch(trace)
            self.last_feedback_update = cleaned
            if trace["status"] == "updated" and self.learner_state_path is not None:
                saved = self.save_learner_state(self.learner_state_path)
                trace["checkpoint_path"] = str(saved)
                trace["checkpoint_sha256"] = hashlib.sha256(saved.read_bytes()).hexdigest()
            self.feedback_history.append(trace)
            return trace

        event_id = str(feedback_event_id or "").strip() or None
        if event_id is not None and event_id in self._seen_feedback_event_ids:
            current_precision = (
                sum(self._route2_precision.values()) / len(self._route2_precision)
            )
            trace = {
                "status": "ignored",
                "trace_state": "ignored_duplicate_feedback_event",
                "mode": "route2",
                "interpretation": "neural_gated",
                "text": cleaned,
                "feedback_event_id": event_id,
                "rejection_reason": "duplicate_feedback_event_id",
                "posterior_changed": False,
                "mean_prior_precision": current_precision,
                "mean_posterior_precision": current_precision,
                "source": source,
                "before_subgoal": before_subgoal,
                "after_subgoal": before_subgoal,
            }
            self._annotate_policy_switch(trace)
            self.feedback_history.append(trace)
            self.last_feedback_update = cleaned
            return trace

        reference_gate = self._route2_reference_gate(cleaned)
        reference_gate["enforced"] = self.require_reference_gate
        reference_gate["decision"] = (
            "audit_only"
            if not self.require_reference_gate
            else "accepted"
            if reference_gate["accepted"]
            else "rejected"
        )
        if self.require_reference_gate and not reference_gate["accepted"]:
            if event_id is not None:
                self._seen_feedback_event_ids.add(event_id)
            reason = str(reference_gate.get("reason") or "reference_gate_rejected")
            is_other = reason == "reference_type_other"
            classifier_error = reason == "classifier_error"
            trace = {
                "status": (
                    "ignored"
                    if is_other
                    else "rejected_classifier_error"
                    if classifier_error
                    else "rejected_low_confidence"
                ),
                "trace_state": (
                    "ignored_reference_other"
                    if is_other
                    else "rejected_reference_classifier_error"
                    if classifier_error
                    else "rejected_reference_classifier"
                ),
                "mode": "route2",
                "interpretation": "neural_gated",
                "text": cleaned,
                "feedback_event_id": event_id,
                "rejection_reason": reason,
                "reference_gate": reference_gate,
                "posterior_changed": False,
                "mean_prior_precision": (
                    sum(self._route2_precision.values()) / len(self._route2_precision)
                ),
                "mean_posterior_precision": (
                    sum(self._route2_precision.values()) / len(self._route2_precision)
                ),
                "source": source,
                "before_subgoal": before_subgoal,
                "after_subgoal": before_subgoal,
            }
            self._annotate_policy_switch(trace)
            self.feedback_history.append(trace)
            self.last_feedback_update = cleaned
            return trace

        model_input_text = (
            str(reference_gate["grounded_text"])
            if self.require_reference_gate
            else cleaned
        )

        from src.neural_inference import predict_reward_distribution, predict_reward_vector

        predictor = self._ensure_route2()
        # Keep compatibility with older callers/tests that inject the historical
        # ``(model, vocab, features, use_feature_counts)`` checkpoint tuple.
        legacy_predictor = isinstance(predictor, tuple)
        if legacy_predictor:
            model, vocab, features, use_feature_counts = predictor
        else:
            features = predictor["features"]
            use_feature_counts = predictor["use_feature_counts"]
        trajectory_features, trajectory_source = self._recent_trajectory_features()
        from src.observations import reference_vector

        counts = reference_vector(trajectory_features, features).tolist()
        if legacy_predictor:
            w_hat = predict_reward_vector(
                model,
                vocab,
                features,
                model_input_text,
                feature_counts=counts if use_feature_counts else None,
            )
            prediction = {
                "weights": w_hat,
                "uncertainty": {feature: 0.0 for feature in features},
                "predictor_kind": "single",
                "ensemble_size": 1,
            }
        else:
            prediction = predict_reward_distribution(
                predictor,
                model_input_text,
                feature_counts=counts,
            )
        w_hat = prediction["weights"]
        uncertainty = prediction["uncertainty"]
        requested_blend = self.online_blend if blend is None else float(blend)
        before_weights = dict(self.weights)
        before_precision = dict(self._route2_precision)
        if requested_blend is None:
            # Original EnsembleFeedforwardNeuralAgent semantics: the averaged
            # neural reward is a fixed-precision Gaussian observation of w.
            update_rule = "paper_independent_gaussian"
            use_human_calibration = (
                source == "human_live"
                and self.route2_human_observation_precision is not None
            )
            observation_precision = (
                self.route2_human_observation_precision
                if use_human_calibration
                else self.route2_observation_precision
            )
            precision_policy = (
                "explicit_human_calibration"
                if use_human_calibration
                else "paper_fixed_precision"
            )
            alpha = None
            for feature, value in w_hat.items():
                prior_precision = self._route2_precision.get(feature, 1.0 / 25.0)
                posterior_precision = prior_precision + observation_precision
                prior_information = prior_precision * self.weights.get(feature, 0.0)
                observed_information = observation_precision * float(value)
                self.weights[feature] = (
                    prior_information + observed_information
                ) / posterior_precision
                self._route2_precision[feature] = posterior_precision
        else:
            # Retained only for explicit backward-compatible ablations.
            update_rule = "legacy_exponential_blend"
            observation_precision = None
            precision_policy = "legacy_ema"
            alpha = max(0.0, min(1.0, requested_blend))
            for feature, value in w_hat.items():
                current = self.weights.get(feature, 0.0)
                self.weights[feature] = (1.0 - alpha) * current + alpha * float(value)
        deltas = {
            feature: self.weights[feature] - before_weights.get(feature, 0.0)
            for feature in self.weights
            if abs(self.weights[feature] - before_weights.get(feature, 0.0)) > 1e-9
        }
        changed = sorted(deltas, key=lambda feature: abs(deltas[feature]), reverse=True)
        # New language must be allowed to change the selected target
        # immediately.  A commitment created from the old weights is stale.
        self._oscillation_commitment_subgoal = None
        self._oscillation_commitment_task_signature = None
        self._recent_motion_signatures.clear()
        self._recent_executed_subgoals.clear()
        self._motion_cycle_steps = 0
        self._last_executed_subgoal = None
        trace = {
            "status": "updated",
            "mode": "route2",
            "interpretation": "neural",
            "text": cleaned,
            "model_input_text": model_input_text,
            "feedback_event_id": event_id,
            "reference_gate": reference_gate,
            "posterior_changed": True,
            "update_rule": update_rule,
            "blend": alpha,
            "observation_precision": observation_precision,
            "observation_precision_policy": precision_policy,
            "mean_prior_precision": sum(before_precision.values()) / len(before_precision),
            "mean_posterior_precision": (
                sum(self._route2_precision.values()) / len(self._route2_precision)
            ),
            "trajectory_source": trajectory_source,
            "trajectory_window": len(self.trajectory_history) or len(self.decision_history),
            "use_feature_counts": use_feature_counts,
            "predictor_kind": prediction["predictor_kind"],
            "ensemble_size": prediction["ensemble_size"],
            "mean_model_disagreement": sum(uncertainty.values()) / len(uncertainty),
            "max_model_disagreement": max(uncertainty.values()),
            "nonzero_trajectory_features": sum(abs(value) > 0 for value in counts),
            "source": source,
            "before_subgoal": before_subgoal,
            "after_subgoal": self._decision_after_update(),
            "weight_delta": {feature: deltas[feature] for feature in changed},
            "top_changes": [
                {
                    "feature": feature,
                    "before": before_weights.get(feature, 0.0),
                    "after": self.weights[feature],
                    "delta": deltas[feature],
                }
                for feature in changed[:12]
            ],
        }
        self._annotate_policy_switch(trace)
        self.last_feedback_update = cleaned
        if event_id is not None:
            self._seen_feedback_event_ids.add(event_id)
            self._processed_feedback_event_ids.add(event_id)
        if self.learner_state_path is not None:
            saved = self.save_learner_state(self.learner_state_path)
            trace["checkpoint_path"] = str(saved)
            trace["checkpoint_sha256"] = hashlib.sha256(saved.read_bytes()).hexdigest()
        self.feedback_history.append(trace)
        return trace

    def learner_state_dict(self) -> dict:
        """Return a versioned online learner checkpoint."""

        payload = {
            "version": 3,
            "feedback_mode": self.feedback_mode,
            "route1_prior": self.route1_prior,
            "human_feedback_precision": self.human_feedback_precision,
            "score_formula": "w_dot_phi",
            "base_weights_sha256": hashlib.sha256(
                json.dumps(self.base_weights, sort_keys=True).encode("utf-8")
            ).hexdigest(),
        }
        if self._route1 is not None:
            payload["route1"] = self._route1.state_dict()
        else:
            payload["weights"] = dict(self.weights)
        if self.feedback_mode == "route2":
            from src.neural_inference import predictor_sha256

            payload["model_path"] = str(self.model_path)
            payload["model_sha256"] = predictor_sha256(self.model_path)
            payload["route2_posterior"] = {
                "mean": dict(self.weights),
                "precision": dict(self._route2_precision),
                "observation_precision": self.route2_observation_precision,
                "human_observation_precision": self.route2_human_observation_precision,
                "processed_feedback_event_ids": sorted(
                    self._processed_feedback_event_ids
                ),
                "require_reference_gate": self.require_reference_gate,
                "update_rule": (
                    "paper_independent_gaussian"
                    if self.online_blend is None
                    else "legacy_exponential_blend"
                ),
                # The rule name alone is not enough to resume an EMA exactly:
                # different alpha values produce different posteriors.
                "online_blend": self.online_blend,
            }
        return payload

    def save_learner_state(self, path: str | Path | None = None) -> Path:
        """Atomically save the complete posterior for exact process resume."""

        target = Path(path) if path is not None else self.learner_state_path
        if target is None:
            raise ValueError("learner state path is not configured")
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_suffix(target.suffix + ".tmp")
        temporary.write_text(
            json.dumps(self.learner_state_dict(), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, target)
        return target

    def load_learner_state(self, path: str | Path | None = None) -> None:
        """Restore a compatible learner checkpoint."""

        target = Path(path) if path is not None else self.learner_state_path
        if target is None:
            raise ValueError("learner state path is not configured")
        state = json.loads(target.read_text(encoding="utf-8"))
        version = int(state.get("version", -1))
        if version not in {1, 2, 3}:
            raise ValueError("unsupported comfort learner checkpoint version")
        if state.get("feedback_mode") != self.feedback_mode:
            raise ValueError("checkpoint feedback mode does not match agent")
        if self.feedback_mode == "route2":
            if version < 2 or state.get("score_formula") != "w_dot_phi":
                raise ValueError(
                    "Route 2 checkpoint predates paper-aligned w dot phi scoring; "
                    "start a new learner state"
                )
            current_base_weights_sha256 = hashlib.sha256(
                json.dumps(self.base_weights, sort_keys=True).encode("utf-8")
            ).hexdigest()
            if state.get("base_weights_sha256") != current_base_weights_sha256:
                raise ValueError(
                    "Route 2 checkpoint base weights do not match agent"
                )
            from src.neural_inference import predictor_sha256

            expected_hash = predictor_sha256(self.model_path)
            if state.get("model_sha256") != expected_hash:
                raise ValueError(
                    "Route 2 checkpoint was created by a different neural model; "
                    "start a new learner state"
                )
            posterior = state.get("route2_posterior")
            if not isinstance(posterior, dict):
                raise ValueError(
                    "Route 2 checkpoint lacks exact update-configuration identity; "
                    "start a new learner state"
                )
            required_identity_fields = {
                "observation_precision",
                "update_rule",
                "online_blend",
            }
            missing_identity_fields = sorted(required_identity_fields - set(posterior))
            if missing_identity_fields:
                raise ValueError(
                    "Route 2 checkpoint lacks exact update-configuration identity "
                    f"({', '.join(missing_identity_fields)}); start a new learner state"
                )

            try:
                saved_observation_precision = float(
                    posterior["observation_precision"]
                )
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    "Route 2 checkpoint has invalid observation precision"
                ) from exc
            if (
                saved_observation_precision <= 0
                or not math.isfinite(saved_observation_precision)
                or not math.isclose(
                    saved_observation_precision,
                    self.route2_observation_precision,
                    rel_tol=0.0,
                    abs_tol=1e-12,
                )
            ):
                raise ValueError(
                    "Route 2 checkpoint observation precision does not match agent"
                )

            expected_update_rule = (
                "paper_independent_gaussian"
                if self.online_blend is None
                else "legacy_exponential_blend"
            )
            if posterior["update_rule"] != expected_update_rule:
                raise ValueError(
                    "Route 2 checkpoint update rule does not match agent"
                )

            saved_online_blend = posterior["online_blend"]
            if saved_online_blend is not None:
                try:
                    saved_online_blend = float(saved_online_blend)
                except (TypeError, ValueError) as exc:
                    raise ValueError(
                        "Route 2 checkpoint has invalid online blend"
                    ) from exc
                if not math.isfinite(saved_online_blend):
                    raise ValueError(
                        "Route 2 checkpoint has invalid online blend"
                    )
            blend_matches = (
                saved_online_blend is None and self.online_blend is None
            ) or (
                saved_online_blend is not None
                and self.online_blend is not None
                and math.isclose(
                    saved_online_blend,
                    self.online_blend,
                    rel_tol=0.0,
                    abs_tol=1e-12,
                )
            )
            if not blend_matches:
                raise ValueError(
                    "Route 2 checkpoint online blend does not match agent"
                )

            # Parse and validate the complete Route 2 payload before mutating
            # any live learner state. Callers may catch a resume error and keep
            # using this agent, so a rejected checkpoint must be all-or-nothing.
            try:
                restored = {
                    str(key): float(value)
                    for key, value in (posterior.get("mean") or {}).items()
                }
            except (AttributeError, TypeError, ValueError) as exc:
                raise ValueError(
                    "Route 2 checkpoint has an invalid posterior mean"
                ) from exc
            feature_set = set(load_features())
            if set(restored) != feature_set:
                raise ValueError("Route 2 checkpoint does not cover the 53-feature schema")
            if any(not math.isfinite(value) for value in restored.values()):
                raise ValueError("Route 2 checkpoint has a non-finite posterior mean")

            raw_precision = posterior.get("precision")
            if not isinstance(raw_precision, dict) or not raw_precision:
                raise ValueError(
                    "Route 2 checkpoint has an invalid posterior precision"
                )
            try:
                restored_precision = {
                    str(key): float(value) for key, value in raw_precision.items()
                }
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    "Route 2 checkpoint has an invalid posterior precision"
                ) from exc
            if set(restored_precision) != feature_set or any(
                value <= 0 or not math.isfinite(value)
                for value in restored_precision.values()
            ):
                raise ValueError(
                    "Route 2 checkpoint has an invalid posterior precision"
                )

            saved_gate_mode = bool(posterior.get("require_reference_gate", False))
            if saved_gate_mode != self.require_reference_gate:
                raise ValueError(
                    "Route 2 checkpoint reference-gate mode does not match agent"
                )
            saved_human_precision = posterior.get("human_observation_precision")
            if saved_human_precision is not None:
                try:
                    saved_human_precision = float(saved_human_precision)
                except (TypeError, ValueError) as exc:
                    raise ValueError(
                        "Route 2 checkpoint has invalid human precision calibration"
                    ) from exc
            if saved_human_precision != self.route2_human_observation_precision:
                raise ValueError(
                    "Route 2 checkpoint human precision calibration does not match agent"
                )
            processed_ids = posterior.get("processed_feedback_event_ids") or []
            if not isinstance(processed_ids, list) or not all(
                isinstance(value, str) and value.strip() for value in processed_ids
            ):
                raise ValueError(
                    "Route 2 checkpoint has invalid processed feedback event IDs"
                )

            # Atomic commit: no validation capable of raising remains below.
            self.weights = restored
            self._route2_precision = restored_precision
            self._processed_feedback_event_ids = set(processed_ids)
            self._seen_feedback_event_ids = set(processed_ids)
            return
        if self._route1 is not None:
            self._route1.load_state_dict(state["route1"])
            self.weights = self._route1.weights()
        else:
            try:
                restored = {
                    str(key): float(value)
                    for key, value in (state.get("weights") or {}).items()
                }
            except (AttributeError, TypeError, ValueError) as exc:
                raise ValueError("checkpoint has invalid weights") from exc
            self.weights = restored

    def reset_weights(self) -> None:
        """Restore the initial frozen vector or Route 1 prior."""

        if self._route1 is not None:
            self._route1.reset(prior_mean=self._route1_prior_mean)
            self.weights = self._route1.weights()
        else:
            self.weights = dict(self.base_weights)
            self._route2_precision = dict(self._base_route2_precision)
        self.reset_context()
        self.feedback_history.clear()
        self.last_feedback_update = ""
        self._seen_feedback_event_ids.clear()
        self._processed_feedback_event_ids.clear()

    def reset_context(self) -> None:
        """Clear temporal grounding state while preserving the learned posterior."""

        self.last_decision = None
        self.decision_history.clear()
        self.trajectory_history.clear()
        self.consecutive_waits = 0
        self.committed_subgoal = None
        self.commitment_age = 0
        self._last_state_signature = None
        self._last_task_progress_signature = None
        self._last_executed_action = None
        self._last_executed_subgoal = None
        self._stalled_wait_steps = 0
        self._unchanged_state_steps = 0
        self._deadlock_detected = False
        self._recent_motion_signatures.clear()
        self._recent_executed_subgoals.clear()
        self._motion_cycle_steps = 0
        self._oscillation_commitment_subgoal = None
        self._oscillation_commitment_task_signature = None
        self._oscillation_commitment_release_reason = None
        self._last_observed_task_signature = None
        self._task_progressed_since_last_action = False
