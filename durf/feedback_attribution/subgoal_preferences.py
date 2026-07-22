"""Map attributed events to pairwise subgoal preference labels.

This file is intentionally small and explicit.  It is the bridge from
event-level attribution to Hu's training target:

    condition + preferred_subgoal > rejected_subgoal
"""

from __future__ import annotations

from typing import Any


TASK_SUBGOALS = (
    "GET_TOMATO",
    "PUT_TOMATO_IN_POT",
    "GET_ONION",
    "PUT_ONION_IN_POT",
    "GET_DISH",
    "PICKUP_SOUP",
    "SERVE_SOUP",
    "WAIT",
)

# Hu can reason over a slightly richer set than the current task planner.
HU_SUBGOALS = (
    *TASK_SUBGOALS,
    "YIELD",
    "WAIT_NEAR_POT",
    "PUT_DOWN_OBJECT",
    "GET_USEFUL_INGREDIENT",
    "CONTINUE_CURRENT_SUBGOAL",
)

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
        "rejected": ["CONTINUE_CURRENT_SUBGOAL"],
    },
    "AI_ignored_ready_or_nearly_ready_pot": {
        "preferred": ["GET_DISH", "PICKUP_SOUP"],
        "rejected": ["WAIT", "GET_TOMATO", "GET_ONION"],
    },
    "AI_missed_plate_pickup_opportunity": {
        "preferred": ["GET_DISH"],
        "rejected": ["GET_TOMATO", "GET_ONION", "WAIT"],
    },
    "AI_failed_to_prepare_ingredient_while_waiting": {
        "preferred": ["GET_USEFUL_INGREDIENT"],
        "rejected": ["WAIT"],
    },
    "AI_pick_drop_loop": {
        "preferred": ["CONTINUE_CURRENT_SUBGOAL", "PUT_DOWN_OBJECT"],
        "rejected": ["WAIT"],
    },
    "AI_missed_useful_ingredient_pickup": {
        "preferred": ["GET_USEFUL_INGREDIENT"],
        "rejected": ["CONTINUE_CURRENT_SUBGOAL", "WAIT"],
    },
    "AI_chose_redundant_ingredient_task": {
        "preferred": ["GET_ONION", "GET_TOMATO"],
        "rejected": ["CONTINUE_CURRENT_SUBGOAL"],
    },
    "AI_successfully_put_ingredient_into_pot": {
        "preferred": ["PUT_TOMATO_IN_POT", "PUT_ONION_IN_POT"],
        "rejected": ["WAIT"],
    },
    "AI_successfully_picked_up_soup": {
        "preferred": ["PICKUP_SOUP"],
        "rejected": ["WAIT"],
    },
    "AI_successfully_delivered_soup": {
        "preferred": ["SERVE_SOUP"],
        "rejected": ["WAIT"],
    },
    "AI_wandered_without_task_progress": {
        "preferred": ["GET_USEFUL_INGREDIENT", "GET_DISH", "YIELD"],
        "rejected": ["WAIT", "CONTINUE_CURRENT_SUBGOAL"],
    },
    "AI_held_unneeded_object_too_long": {
        "preferred": ["PUT_DOWN_OBJECT"],
        "rejected": ["CONTINUE_CURRENT_SUBGOAL", "WAIT"],
    },
    "AI_put_object_on_unhelpful_counter": {
        "preferred": ["PUT_DOWN_OBJECT"],
        "rejected": ["CONTINUE_CURRENT_SUBGOAL"],
    },
    "AI_should_stage_object_near_pot": {
        "preferred": ["PUT_DOWN_OBJECT", "WAIT_NEAR_POT"],
        "rejected": ["CONTINUE_CURRENT_SUBGOAL"],
    },
}


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
) -> tuple[list[str], list[str]]:
    """Return preferred/rejected subgoals for Hu pairwise labels.

    LLM output wins if it already supplied subgoals.  Otherwise we use the
    conservative event defaults above.
    """

    preferred = normalize_subgoals(preferred_subgoals)
    rejected = normalize_subgoals(rejected_subgoals)
    if preferred and rejected:
        return preferred, rejected

    defaults = EVENT_SUBGOAL_DEFAULTS.get(str(target_event), {})
    if not preferred:
        preferred = list(defaults.get("preferred", []))
    if not rejected:
        rejected = list(defaults.get("rejected", []))
    return preferred, rejected
