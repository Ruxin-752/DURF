"""Map attributed events to pairwise subgoal preference labels.

This file is intentionally small and explicit.  It is the bridge from
event-level attribution to Hu's training target:

    condition + preferred_subgoal > rejected_subgoal
"""

from __future__ import annotations

from typing import Any


PLANNER_TASK_SUBGOALS = (
    "GET_TOMATO",
    "PUT_TOMATO_IN_POT",
    "GET_ONION",
    "PUT_ONION_IN_POT",
    "GET_DISH",
    "PICKUP_SOUP",
    "SERVE_SOUP",
    "WAIT",
)

# Hu can reason over a slightly richer task set than the current task planner.
TASK_HU_SUBGOALS = (
    *PLANNER_TASK_SUBGOALS,
    "WAIT_NEAR_POT",
    "PUT_DOWN_OBJECT",
    "GET_USEFUL_INGREDIENT",
)

# Only options that are actually constructed as runtime candidates are kept.
# HOLD_POSITION duplicates YIELD's stay-action; REROUTE is a replanning-level
# decision that needs its own path-planner workflow, not a coordination cell.
COORDINATION_SUBGOALS = (
    "CONTINUE_CURRENT_SUBGOAL",
    "YIELD",
)

HU_SUBGOALS = (*TASK_HU_SUBGOALS, *COORDINATION_SUBGOALS)

# Backward-compatible name used by older imports.
TASK_SUBGOALS = PLANNER_TASK_SUBGOALS

SUBGOAL_ALIASES = {
    "DELIVER_SOUP": "SERVE_SOUP",
    "DELIVERY": "SERVE_SOUP",
    "GET_PLATE": "GET_DISH",
    "PICK_DISH": "GET_DISH",
    "PICK_UP_DISH": "GET_DISH",
    "PICKUP_DISH": "GET_DISH",
    "GET_INGREDIENT": "GET_USEFUL_INGREDIENT",
    "GET_USEFUL_OBJECT": "GET_USEFUL_INGREDIENT",
    "CARRY_INGREDIENT": "CONTINUE_CURRENT_SUBGOAL",
    "KEEP_CARRYING_INGREDIENT": "CONTINUE_CURRENT_SUBGOAL",
    "INSIST_CURRENT_SUBGOAL": "CONTINUE_CURRENT_SUBGOAL",
    "DO_NOT_WAIT": "WAIT",
}


# Complete pairs are provided only when the event itself identifies both the
# desired alternative and the behavior that should lose priority. Partial
# defaults are review suggestions, not automatically trainable Hu labels.
EVENT_SUBGOAL_DEFAULTS: dict[str, dict[str, list[str]]] = {
    "AI_blocked_human_path": {
        "preferred": ["YIELD"],
        "rejected": ["CONTINUE_CURRENT_SUBGOAL"],
    },
    "AI_failed_to_yield": {
        "preferred": ["YIELD"],
        "rejected": ["CONTINUE_CURRENT_SUBGOAL"],
    },
    "AI_successfully_yielded": {
        "preferred": ["YIELD"],
        "rejected": [],
    },
    "AI_ignored_ready_or_nearly_ready_pot": {
        "preferred": ["GET_DISH", "PICKUP_SOUP"],
        "rejected": [],
    },
    "AI_missed_plate_pickup_opportunity": {
        "preferred": ["GET_DISH"],
        "rejected": [],
    },
    "AI_failed_to_prepare_ingredient_while_waiting": {
        "preferred": ["GET_USEFUL_INGREDIENT"],
        "rejected": ["WAIT"],
    },
    "AI_successfully_put_ingredient_into_pot": {
        "preferred": [],
        "rejected": [],
    },
    "AI_successfully_picked_up_soup": {
        "preferred": ["PICKUP_SOUP"],
        "rejected": [],
    },
    "AI_successfully_delivered_soup": {
        "preferred": ["SERVE_SOUP"],
        "rejected": [],
    },
}

NEGATIVE_EVENT_VALENCES = {"negative_problem", "missed_opportunity"}


def canonical_subgoal(value: Any) -> str | None:
    text = str(value).strip()
    if not text:
        return None
    upper = text.upper()
    canonical = SUBGOAL_ALIASES.get(upper, upper)
    if canonical in HU_SUBGOALS:
        return canonical
    return None


def normalize_subgoals(values: Any) -> list[str]:
    if values is None:
        return []
    if isinstance(values, str):
        values = [values]
    normalized = []
    for value in values:
        text = canonical_subgoal(value)
        if text and text not in normalized:
            normalized.append(text)
    return normalized


def infer_subgoal_preferences(
    *,
    target_event: str | None,
    preferred_subgoals: Any = None,
    rejected_subgoals: Any = None,
    observed_subgoal: Any = None,
    alternative_subgoals: Any = None,
    event_valence: str | None = None,
) -> tuple[list[str], list[str]]:
    """Return preferred/rejected subgoals for Hu pairwise labels.

    Explicit semantic output is never completed with guessed values. When both
    explicit sides are empty, conservative event defaults may suggest one or
    both sides. For a negative event, the actually observed subgoal may fill
    the rejected side only when a distinct preferred alternative is known.
    """

    preferred = normalize_subgoals(preferred_subgoals)
    rejected = normalize_subgoals(rejected_subgoals)
    if preferred or rejected:
        return preferred, rejected

    defaults = EVENT_SUBGOAL_DEFAULTS.get(str(target_event), {})
    preferred = list(defaults.get("preferred", []))
    rejected = list(defaults.get("rejected", []))

    if not preferred:
        preferred = normalize_subgoals(alternative_subgoals)

    if preferred and not rejected and event_valence in NEGATIVE_EVENT_VALENCES:
        observed = normalize_subgoals([observed_subgoal])
        rejected = [
            subgoal
            for subgoal in observed
            if subgoal not in preferred
        ]

    return preferred, rejected
