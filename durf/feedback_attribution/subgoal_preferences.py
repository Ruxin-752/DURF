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

# Hu's task vocabulary is exactly the set of subgoals the runtime candidate
# generator can actually put on the table (collect_rule_teacher_dataset.py):
# a preference on a name that never appears as a runtime candidate can never
# change a decision, however many labels it collects.
TASK_HU_SUBGOALS = (
    *PLANNER_TASK_SUBGOALS,
    "WAIT_NEAR_POT",
    "PUT_DOWN_OBJECT",
)

# Attribution may name a subgoal at a coarser grain than the runtime uses.
# "GET_USEFUL_INGREDIENT" ("prepare the next ingredient", "take the onion off
# the counter") is legitimate feedback, but the runtime only ever offers
# GET_TOMATO / GET_ONION, so hu_dataset_builder resolves it to whichever of
# those the decision's real candidate set contained before a pair is formed.
# It is accepted as an attribution label and is NOT a Hu dimension.
ATTRIBUTION_ONLY_TASK_SUBGOALS = ("GET_USEFUL_INGREDIENT",)
USEFUL_INGREDIENT_RUNTIME_SUBGOALS = ("GET_TOMATO", "GET_ONION")

# Only options that are actually constructed as runtime candidates are kept.
# HOLD_POSITION duplicates YIELD's stay-action; REROUTE is a replanning-level
# decision that needs its own path-planner workflow, not a coordination cell.
COORDINATION_SUBGOALS = (
    "CONTINUE_CURRENT_SUBGOAL",
    "YIELD",
)

HU_SUBGOALS = (*TASK_HU_SUBGOALS, *COORDINATION_SUBGOALS)
ATTRIBUTION_SUBGOALS = (*HU_SUBGOALS, *ATTRIBUTION_ONLY_TASK_SUBGOALS)

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
    "AI_failed_to_yield_or_clear_path": {
        "preferred": ["YIELD"],
        "rejected": ["CONTINUE_CURRENT_SUBGOAL"],
    },
    "AI_successfully_yielded": {
        "preferred": ["YIELD"],
        "rejected": [],
    },
    "AI_maintained_current_subgoal_during_conflict": {
        "preferred": ["CONTINUE_CURRENT_SUBGOAL"],
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


def infer_explicit_preference_from_text(
    feedback_text: str | None,
) -> tuple[list[str], list[str], str | None]:
    """Extract only high-confidence, explicit policy preferences.

    This is intentionally separate from event attribution. A participant may
    describe what the AI should do in a future situation before that event has
    actually occurred. Such feedback is useful for Hu, but it must not be
    mislabeled as evidence that an event happened.

    The fallback is deliberately conservative; richer language parsing belongs
    to the LLM attributor. The returned source is used to keep these labels
    auditable in the provenance file.
    """
    text = str(feedback_text or "").strip().lower()
    if not text:
        return [], [], None

    # Coordination preference: continue the current task instead of yielding.
    no_yield = any(
        phrase in text
        for phrase in (
            "don't yield",
            "do not yield",
            "dont yield",
            "no need to yield",
            "not yield",
            "shouldn't yield",
            "should not yield",
            "insist",
            "keep going",
            "don't stop",
            "do not stop",
            "don't change route",
            "do not change route",
            "don't pause",
            "do not pause",
        )
    )
    yield_preference = any(
        phrase in text
        for phrase in (
            "let me through",
            "step aside",
            "move out of the way",
            "you should yield",
            "should yield",
            "need to yield",
        )
    )
    if no_yield and not yield_preference:
        return ["CONTINUE_CURRENT_SUBGOAL"], ["YIELD"], "explicit_text_fallback"
    if yield_preference and not no_yield:
        return ["YIELD"], ["CONTINUE_CURRENT_SUBGOAL"], "explicit_text_fallback"

    # Task preference: take the dish when the human is covering the final
    # ingredient or when the feedback explicitly contrasts dish vs ingredient.
    asks_for_dish = any(
        phrase in text
        for phrase in (
            "get plate",
            "get the plate",
            "get dish",
            "get the dish",
            "pick up the plate",
            "pick up plate",
        )
    )
    contrasts_ingredient = any(
        phrase in text
        for phrase in (
            "instead of more ingredient",
            "instead of getting ingredient",
            "rather than ingredient",
            "last ingredient",
        )
    )
    if asks_for_dish and contrasts_ingredient:
        return ["GET_DISH"], ["GET_USEFUL_INGREDIENT"], "explicit_text_fallback"

    # During cooking, preparing/staging another ingredient is preferred to
    # waiting. This is a task-level preference, not a claim that a failure
    # event occurred.
    asks_to_prepare = any(
        phrase in text
        for phrase in (
            "prepare ingredient",
            "prepare food",
            "get ingredient while",
            "drop it near the pot",
            "put it near the pot",
        )
    )
    if asks_to_prepare:
        return ["GET_USEFUL_INGREDIENT"], ["WAIT"], "explicit_text_fallback"

    return [], [], None


def canonical_subgoal(value: Any) -> str | None:
    text = str(value).strip()
    if not text:
        return None
    upper = text.upper()
    canonical = SUBGOAL_ALIASES.get(upper, upper)
    if canonical in ATTRIBUTION_SUBGOALS:
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


def resolve_useful_ingredient(
    subgoals: list[str],
    candidate_subgoals: Any,
) -> tuple[list[str], list[str]]:
    """Replace GET_USEFUL_INGREDIENT with the runtime fetch subgoal(s) that the
    decision's candidate set actually contained.

    Returns ``(resolved, unresolved)``.  ``unresolved`` is non-empty when the
    runtime offered no ingredient fetch at that decision at all -- then the
    preference is not expressible there and the pair must be dropped rather
    than trained on a phantom option.  ``candidate_subgoals`` accepts the
    trajectory's ``ai_subgoal_candidates`` entries (dicts with ``subgoal``) or
    plain names.
    """
    names: list[str] = []
    for candidate in candidate_subgoals or []:
        name = candidate.get("subgoal") if isinstance(candidate, dict) else candidate
        if isinstance(name, str) and name not in names:
            names.append(name)
    concrete = [name for name in USEFUL_INGREDIENT_RUNTIME_SUBGOALS if name in names]
    resolved: list[str] = []
    unresolved: list[str] = []
    for subgoal in subgoals:
        if subgoal not in ATTRIBUTION_ONLY_TASK_SUBGOALS:
            if subgoal not in resolved:
                resolved.append(subgoal)
            continue
        if not concrete:
            unresolved.append(subgoal)
            continue
        for name in concrete:
            if name not in resolved:
                resolved.append(name)
    return resolved, unresolved


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
