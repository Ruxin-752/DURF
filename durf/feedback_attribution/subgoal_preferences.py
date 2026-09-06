"""Map attributed events to pairwise subgoal preference labels.

This file is intentionally small and explicit.  It is the bridge from
event-level attribution to Hu's training target:

    condition + preferred_subgoal > rejected_subgoal
"""

from __future__ import annotations

import re
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
    # "you keep going for the plate" is about a task object, not a corridor.
    # A no-yield reading needs the sentence to be free of task-object talk
    # unless it also names the corridor situation.
    task_object_talk = any(
        word in text for word in ("plate", "dish", "ingredient", "tomato", "onion", "pot")
    )
    corridor_talk = any(
        word in text for word in ("way", "path", "block", "pass", "route", "corridor", "move", "yield", "stop")
    )
    if no_yield and task_object_talk and not corridor_talk:
        no_yield = False
    if no_yield and not yield_preference:
        return ["CONTINUE_CURRENT_SUBGOAL"], ["YIELD"], "explicit_text_fallback"
    if yield_preference and not no_yield:
        return ["YIELD"], ["CONTINUE_CURRENT_SUBGOAL"], "explicit_text_fallback"

    # Task-domain division of labour. Both halves of the statement are
    # required ("I take X, you take Y"), so a passing remark about a plate is
    # never read as a policy. These are genuine preferences, not corrections:
    # both options are feasible and near-equal in task points, which is
    # exactly where a preference is allowed to decide (see
    # mechanism_blueprint_v1.md S3.3).
    speaker_takes_plate = any(
        phrase in text
        for phrase in (
            "i'll get the plate",
            "i will get the plate",
            "i'll get the dish",
            "i'll take the plate",
            "i've got the plate",
            "ive got the plate",
            "leave the plate to me",
            "leave the dish to me",
        )
    )
    asks_partner_to_prep = any(
        phrase in text
        for phrase in (
            "next ingredient",
            "next batch",
            "keep prepping",
            "start prepping",
        )
    )
    asks_partner_to_take_plate = any(
        phrase in text
        for phrase in (
            "you get the plate",
            "you get the dish",
            "you grab the plate",
            "you grab the dish",
            "grab the plate",
            "grab the dish",
            "you take the plate",
            "you take the dish",
            "plates are yours",
            "plate is yours",
            "dishes are yours",
        )
    )
    speaker_takes_ingredients = any(
        phrase in text
        for phrase in (
            "i'll handle the ingredient",
            "i'll handle the next ingredient",
            "i'll take care of the ingredient",
            "i'll get the ingredient",
            "leave the ingredients to me",
            "i'll do the prepping",
            "ingredients are mine",
        )
    )
    if speaker_takes_plate and asks_partner_to_prep:
        return (
            ["GET_USEFUL_INGREDIENT"],
            ["GET_DISH"],
            "explicit_text_fallback",
        )
    if asks_partner_to_take_plate and speaker_takes_ingredients:
        return (
            ["GET_DISH"],
            ["GET_USEFUL_INGREDIENT"],
            "explicit_text_fallback",
        )

    # Ingredient order: which of the two recipe ingredients to fetch first.
    # Two-sided like the others: a bare "get the onion" is a correction about
    # NOW, not a standing order preference.  Only when the speaker contrasts
    # the two ingredients (or claims one for themself) is it read as policy.
    onion_first = any(
        phrase in text
        for phrase in (
            "onion first",
            "start with the onion",
            "start with an onion",
            "get the onion first",
            "you handle the onion",
            "you do the onion",
            "onion instead",
        )
    ) and ("tomato" in text or "onion first" in text or "onion instead" in text)
    tomato_first = any(
        phrase in text
        for phrase in (
            "tomato first",
            "tomatoes first",
            "start with a tomato",
            "start with the tomato",
            "you just do tomatoes",
            "you do tomatoes",
            "you handle the tomato",
            "onion goes in last",
            "onion can wait",
        )
    ) and ("onion" in text or "tomato first" in text or "tomatoes first" in text)
    if onion_first and not tomato_first:
        return ["GET_ONION"], ["GET_TOMATO"], "explicit_text_fallback"
    if tomato_first and not onion_first:
        return ["GET_TOMATO"], ["GET_ONION"], "explicit_text_fallback"

    # Backup dish: whether to fetch a second plate while the partner already
    # carries one.  "another"/"second"/"spare" + plate/dish is the positive
    # signal; "already have the plate" + a prohibition is the negative one.
    mentions_plate = any(word in text for word in ("plate", "dish"))
    wants_backup = mentions_plate and any(
        phrase in text
        for phrase in (
            "another one", "another plate", "another dish", "second plate",
            "second dish", "spare plate", "spare dish", "grab another", "get another",
            "one too",
        )
    )
    # A negation right before "another / second / spare plate" flips the
    # meaning.  The LLM-generated bank produced "don't bother with another
    # dish", "no need for a second plate", "don't waste time on another plate"
    # -- all read as WANTING a backup by the phrase list alone.
    negated_backup = bool(
        re.search(
            r"(don'?t|do not|no need for|no |never|skip|forget|waste time on|bother with)"
            r"[^.;,]{0,25}?(another|second|spare|extra|2nd)\s+(plate|dish)",
            text,
        )
    )
    refuses_backup = mentions_plate and (
        negated_backup
        or (
            any(
                phrase in text
                for phrase in (
                    "don't go get another", "do not get another", "dont get another",
                    "don't get another", "one plate is enough", "one dish is enough",
                    "two of us carry", "leave the dish", "leave the plate",
                )
            )
            and any(
                phrase in text
                for phrase in ("already", "i have", "i've got", "one plate", "one dish", "two of us")
            )
        )
    )
    # "don't go get another one" contains "another one": the prohibition wins.
    wants_backup = wants_backup and not refuses_backup
    if wants_backup and not refuses_backup:
        return ["GET_DISH"], ["GET_TOMATO", "GET_ONION"], "explicit_text_fallback"
    if refuses_backup and not wants_backup:
        return ["GET_TOMATO", "GET_ONION"], ["GET_DISH"], "explicit_text_fallback"

    # Free hands: put the plate down while the soup cooks and go do something
    # else, rather than waiting by the pot with it.
    puts_plate_down = mentions_plate and any(
        phrase in text
        for phrase in (
            "put the plate down", "put the dish down", "drop it and", "drop the plate",
            "put it down for now", "put the dish down for now", "put the plate down and go",
            "put down the plate", "put down the dish",
        )
    )
    to_do_something = any(
        phrase in text
        for phrase in ("go do something", "get ingredients", "go get", "go find", "find some ingredient", "do something", "pick it up when")
    )
    if puts_plate_down and to_do_something:
        return ["PUT_DOWN_OBJECT"], ["WAIT_NEAR_POT"], "explicit_text_fallback"

    # Hold the plate rather than dropping it while the soup is not ready.
    keeps_plate = any(
        phrase in text
        for phrase in (
            "don't put the plate down",
            "do not put the plate down",
            "dont put the plate down",
            "don't put the dish down",
            "keep hold of the plate",
            "keep hold of the dish",
            "hang on to the plate",
            "hang onto the plate",
            "hang on to the dish",
            "hang onto the dish",
            "keep the plate",
            "keep the dish",
        )
    )
    stays_by_pot = any(
        phrase in text
        for phrase in ("by the pot", "near the pot", "at the pot")
    )
    if keeps_plate and stays_by_pot:
        return (
            ["WAIT_NEAR_POT"],
            ["PUT_DOWN_OBJECT"],
            "explicit_text_fallback",
        )

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
