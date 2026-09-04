"""Explicit, preference-aware coordination decisions for the play runtime.

Task planning answers what the AI should do.  This module answers how the AI
should locally execute that task when a human path conflict appears.  It keeps
coordination options separate from task subgoals so Hu can score each decision
domain without comparing incompatible choices such as GET_TOMATO and YIELD.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from durf.baseline.collect_rule_teacher_dataset import first_action_to_feature


CONTINUE_CURRENT_SUBGOAL = "CONTINUE_CURRENT_SUBGOAL"
YIELD = "YIELD"

# Only runtime-constructible options. HOLD_POSITION collapsed into YIELD's
# stay action; REROUTE is a replan decision and lives at the task layer, so it
# must not appear in the coordination option vocabulary.
COORDINATION_OPTIONS = (
    CONTINUE_CURRENT_SUBGOAL,
    YIELD,
)

CONFLICT_HUMAN_ENTERING_AI_TILE = "human_entering_ai_tile"
CONFLICT_CONTESTED_DESTINATION = "contested_destination"
CONFLICT_AI_ENTERING_HUMAN_TILE = "ai_entering_human_tile"
# The human's destination is free, but the AI occupies the tile just beyond
# it along the human's motion direction, so the human would have to detour.
CONFLICT_AI_BLOCKING_HUMAN_ROUTE = "ai_blocking_human_route"


@dataclass
class CoordinationCandidate:
    option: str
    action: int
    base_score: float
    reason: str
    feasible: bool = True
    hu_score: float = 0.0
    final_score: float | None = None
    # Whether the loaded Hu has labelled evidence about this option (same
    # rule as the task layer: no evidence, no opinion).
    hu_supported: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "option": self.option,
            "action": self.action,
            "base_score": self.base_score,
            "hu_score": self.hu_score,
            "final_score": self.final_score,
            "reason": self.reason,
            "feasible": self.feasible,
            "hu_supported": self.hu_supported,
        }


@dataclass
class ActiveCoordination:
    decision_id: str
    option: str
    task_subgoal: str
    start_timestep: int
    commit_until: int
    expires_at: int
    conflict_type: str
    initial_ai_pos: tuple[int, int]
    initial_human_pos: tuple[int, int]


@dataclass
class CoordinationResult:
    action: int
    event: str
    decision: dict[str, Any]


def path_conflict_type(
    *,
    ai_pos: tuple[int, int],
    human_pos: tuple[int, int],
    ai_target: tuple[int, int],
    human_target: tuple[int, int],
    ai_is_moving: bool,
    human_is_moving: bool,
) -> str | None:
    """Classify immediate path conflicts without assigning normative value.

    ``ai_blocking_human_route`` extends the trigger surface beyond instant
    geometry: the human's destination itself is free, but the AI occupies the
    tile just beyond it along the human's motion direction, so the human would
    have to detour around the AI.  Without this case, "AI standing so the
    human must route around it" never triggers a coordination decision and
    coordination feedback is never collected.
    """
    if human_is_moving and human_target == ai_pos:
        return CONFLICT_HUMAN_ENTERING_AI_TILE
    if human_is_moving and ai_is_moving and human_target == ai_target:
        return CONFLICT_CONTESTED_DESTINATION
    if ai_is_moving and ai_target == human_pos:
        return CONFLICT_AI_ENTERING_HUMAN_TILE
    if human_is_moving and human_target != human_pos:
        human_delta = (
            human_target[0] - human_pos[0],
            human_target[1] - human_pos[1],
        )
        step_after_target = (
            human_target[0] + human_delta[0],
            human_target[1] + human_delta[1],
        )
        if step_after_target == ai_pos:
            return CONFLICT_AI_BLOCKING_HUMAN_ROUTE
    return None


def choose_reroute_action(
    motion_planner,
    ai_player,
    target_positions,
    human_pos: tuple[int, int],
    human_target: tuple[int, int],
    proposed_action: int,
) -> int | None:
    """First step of an alternate route to the task target that avoids the
    human's current tile and its destination, when one exists and genuinely
    differs from the plan already in conflict.

    Returns ``None`` when there is nothing to reroute to: no task target on
    record, or the planner cannot find any step that avoids the human (a
    single momentary tile block rarely severs every path, but a narrow
    corridor can), or the avoiding route's first step happens to coincide
    with the plan already in conflict (a purely dynamic, same-timestep
    conflict that a static reroute cannot resolve).  Either way the caller
    is expected to fall back to BACK_OFF and then WAIT.
    """
    if not target_positions:
        return None
    blocked = {tuple(human_pos), tuple(human_target)}
    reroute_action = first_action_to_feature(
        motion_planner, ai_player, list(target_positions), blocked_positions=blocked
    )
    if reroute_action is None or int(reroute_action) == int(proposed_action):
        return None
    return int(reroute_action)


def build_coordination_candidates(
    *,
    proposed_action: int,
    yield_action: int | None,
    stay_action: int,
    conflict_type: str,
    ai_adjacent_to_current_subgoal_target: bool = False,
    yield_on_conflict_score: float = 1.0,
    continue_score: float = 0.0,
    yield_mode: str | None = None,
) -> list[CoordinationCandidate]:
    """Build the minimal executable coordination choice set.

    The default prior favors YIELD to reproduce the old polite wrapper when Hu
    is disabled.  When the AI is one step from completing its task subgoal
    (``ai_adjacent_to_current_subgoal_target``), persistence beats politeness:
    the human should route around, so the prior is reversed to prefer
    CONTINUE_CURRENT_SUBGOAL.  Hu may still reverse either ordering.

    ``yield_mode`` records how the caller picked ``yield_action`` (e.g.
    ``"reroute"``, ``"back_off"``, ``"wait"``) purely for audit purposes.  It
    is folded into the YIELD candidate's ``reason`` string, never into the
    coordination option vocabulary itself: Hu still only ever chooses between
    CONTINUE_CURRENT_SUBGOAL and YIELD, and has no say in which yield_mode was
    used to execute a YIELD once chosen.
    """
    effective_yield_action = stay_action if yield_action is None else yield_action
    if ai_adjacent_to_current_subgoal_target:
        continue_prior = yield_on_conflict_score
        yield_prior = continue_score
        continue_reason = f"continue_task_during_{conflict_type}_near_task_goal"
    else:
        continue_prior = continue_score
        yield_prior = yield_on_conflict_score
        continue_reason = f"continue_task_during_{conflict_type}"
    yield_reason = f"yield_during_{conflict_type}"
    if yield_mode:
        yield_reason = f"{yield_reason}_via_{yield_mode}"
    return [
        CoordinationCandidate(
            option=CONTINUE_CURRENT_SUBGOAL,
            action=proposed_action,
            base_score=continue_prior,
            reason=continue_reason,
        ),
        CoordinationCandidate(
            option=YIELD,
            action=effective_yield_action,
            base_score=yield_prior,
            reason=yield_reason,
        ),
    ]


class CoordinationController:
    """Stateful short-horizon controller for coordination options.

    A selected option is committed briefly to avoid timestep-by-timestep
    oscillation.  BOTH options are bounded: if the conflict is still there
    when an option expires, a short cooldown makes that option infeasible so
    the other one gets its turn.  For YIELD that stops the AI retreating
    forever; for CONTINUE (added 2026-09-04) it stops the AI insisting
    forever.  The second bound was invisible under the prior alone -- the
    prior only insists when one step from the goal -- and became load-bearing
    the moment a learned preference could choose CONTINUE in a far-from-goal
    head-on: the move is blocked, nothing changes, the same decision recurs,
    and a preference for insisting compounds into a permanent deadlock
    (observed as 524 consecutive CONTINUE decisions and a 0-reward episode).
    """

    def __init__(
        self,
        *,
        min_commit_steps: int = 1,
        max_option_steps: int = 3,
        yield_cooldown_steps: int = 2,
        continue_cooldown_steps: int | None = None,
    ) -> None:
        if min_commit_steps < 1:
            raise ValueError("min_commit_steps must be at least 1")
        if max_option_steps < min_commit_steps:
            raise ValueError("max_option_steps must be >= min_commit_steps")
        if yield_cooldown_steps < 0:
            raise ValueError("yield_cooldown_steps cannot be negative")
        if continue_cooldown_steps is None:
            continue_cooldown_steps = yield_cooldown_steps
        if continue_cooldown_steps < 0:
            raise ValueError("continue_cooldown_steps cannot be negative")
        self.min_commit_steps = min_commit_steps
        self.max_option_steps = max_option_steps
        self.yield_cooldown_steps = yield_cooldown_steps
        self.continue_cooldown_steps = continue_cooldown_steps
        self.active: ActiveCoordination | None = None
        self.yield_cooldown_until = -1
        self.continue_cooldown_until = -1
        self._decision_counter = 0

    def reset(self) -> None:
        self.active = None
        self.yield_cooldown_until = -1
        self.continue_cooldown_until = -1
        self._decision_counter = 0

    def clear_if_no_conflict(self) -> None:
        self.active = None
        self.yield_cooldown_until = -1
        self.continue_cooldown_until = -1

    def resolve(
        self,
        *,
        timestep: int,
        episode: int,
        task_subgoal: str,
        conflict_type: str | None,
        ai_pos: tuple[int, int],
        human_pos: tuple[int, int],
        candidates: list[CoordinationCandidate],
        condition_features: dict[str, Any],
        hu_score: Callable[[str], float] | None = None,
        hu_tolerance: float = 0.0,
        apply_hu: bool = False,
        hu_supported: Callable[[str], bool] | None = None,
    ) -> CoordinationResult | None:
        """Resolve a conflict: the prior satisfices, the preference orders.

        Same rule as the task layer (2026-09-04), for the same reason: the
        prior (0/1 conflict priority) and ``hu_score`` (pairwise-logistic
        output, order meaningful, magnitude an artefact) are not
        commensurable, so ``prior + lambda * hu`` needed an exchange rate
        nobody could state -- pitfall (2).  Instead every feasible option
        within ``hu_tolerance`` prior points of the best prior forms the
        acceptable set and the preference orders it; ties fall back to the
        prior.  With a two-level prior this is a switch and is documented as
        one: 0 keeps the prior's pick, >= 1 lets the user's preference decide
        yield-vs-continue outright (the prior only breaks ties).  There is no
        step-cost band here on purpose: YIELD already carries a commitment
        horizon (max_option_steps) and a cooldown, which is what bounds a
        "wait for me" preference; a per-decision cost would add nothing the
        horizon does not already give.

        An option Hu has no labelled evidence about can never be the reason
        the decision leaves the prior's pick (``hu_supported``); the pick
        itself is always eligible.
        """
        if conflict_type is None:
            self.clear_if_no_conflict()
            return None

        active = self.active
        if active is not None:
            same_context = (
                active.task_subgoal == task_subgoal
                and active.conflict_type == conflict_type
            )
            if not same_context:
                self.active = None
                active = None
            elif timestep > active.expires_at:
                # The option ran its full horizon and the conflict is still
                # here: give the other option its turn.
                # The expired option steps aside and the other one is freed,
                # so at least one option is always feasible.
                if active.option == YIELD:
                    self.yield_cooldown_until = (
                        timestep + self.yield_cooldown_steps
                    )
                    self.continue_cooldown_until = -1
                else:
                    self.continue_cooldown_until = (
                        timestep + self.continue_cooldown_steps
                    )
                    self.yield_cooldown_until = -1
                self.active = None
                active = None

        for candidate in candidates:
            if hu_score is not None:
                candidate.hu_score = float(hu_score(candidate.option))
            if hu_supported is not None:
                candidate.hu_supported = bool(hu_supported(candidate.option))
            if (
                candidate.option == YIELD
                and timestep <= self.yield_cooldown_until
            ):
                candidate.feasible = False
                candidate.reason += "_cooldown"
            if (
                candidate.option == CONTINUE_CURRENT_SUBGOAL
                and timestep <= self.continue_cooldown_until
            ):
                candidate.feasible = False
                candidate.reason += "_cooldown"
            # Never added to: the prior is the option's score, the preference
            # only ever orders (see docstring).
            candidate.final_score = candidate.base_score

        selected: CoordinationCandidate | None = None
        status = "started"
        # Keep the short option stable for its bounded lifetime.  Material
        # context changes (task/conflict type) already clear it above.
        if active is not None and timestep <= active.expires_at:
            selected = next(
                (
                    candidate
                    for candidate in candidates
                    if candidate.option == active.option and candidate.feasible
                ),
                None,
            )
            if selected is not None:
                status = "continued"

        tolerance = max(0.0, float(hu_tolerance)) if apply_hu else 0.0
        prior_pick: CoordinationCandidate | None = None
        acceptable: list[CoordinationCandidate] = []
        if selected is None:
            feasible = [candidate for candidate in candidates if candidate.feasible]
            if not feasible:
                return None
            best_prior = max(candidate.base_score for candidate in feasible)
            prior_pick = max(feasible, key=lambda candidate: candidate.base_score)
            acceptable = [
                candidate
                for candidate in feasible
                if candidate.base_score >= best_prior - tolerance
            ]
            pool = [prior_pick] + [
                candidate
                for candidate in acceptable
                if candidate is not prior_pick and candidate.hu_supported
            ]
            selected = max(
                pool,
                key=lambda candidate: (candidate.hu_score, candidate.base_score),
            )
            self._decision_counter += 1
            decision_id = (
                f"coord:e{episode}:t{timestep}:n{self._decision_counter}"
            )
            self.active = ActiveCoordination(
                decision_id=decision_id,
                option=selected.option,
                task_subgoal=task_subgoal,
                start_timestep=timestep,
                commit_until=timestep + self.min_commit_steps - 1,
                expires_at=timestep + self.max_option_steps - 1,
                conflict_type=conflict_type,
                initial_ai_pos=ai_pos,
                initial_human_pos=human_pos,
            )
            active = self.active
        else:
            active = self.active

        if active is None:
            raise RuntimeError("Coordination decision lost active state")

        event = (
            "AI_successfully_yielded"
            if selected.option == YIELD
            else "AI_maintained_current_subgoal_during_conflict"
        )
        return CoordinationResult(
            action=selected.action,
            event=event,
            decision={
                "record_type": "runtime_decision",
                "decision_id": active.decision_id,
                "decision_level": "coordination",
                "status": status,
                "start_timestep": active.start_timestep,
                "current_timestep": timestep,
                "commit_until": active.commit_until,
                "expires_at": active.expires_at,
                "conflict_type": conflict_type,
                "task_subgoal": task_subgoal,
                "condition_at_decision": condition_features,
                "candidate_set": [candidate.option for candidate in candidates],
                "candidates": [candidate.to_dict() for candidate in candidates],
                "selected": selected.option,
                "selected_action": selected.action,
                "hu_applied": bool(apply_hu and tolerance > 0.0),
                "hu_coordination_tolerance": tolerance,
                "tolerance_unit": "prior_points",
                "prior_pick": prior_pick.option if prior_pick is not None else None,
                "acceptable_set_size": len(acceptable) if prior_pick is not None else None,
                "preference_override": (
                    selected.option != prior_pick.option
                    if prior_pick is not None
                    else False
                ),
            },
        )
