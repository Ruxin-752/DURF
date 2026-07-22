"""Deterministic natural-language templates for synthetic feedback.

Given a teacher :class:`~src.subgoal_teacher.FeedbackIntent` and its context,
render short, human-sounding feedback sentences. Templates are the *coverage
floor*: every (subgoal x role x comfort/task theme) combination gets at least
a few deterministic variants, so the corpus is never empty and never depends on
a network call. The LLM branch in ``generate_synthetic_feedback.py`` adds
linguistic diversity on top of these.

The text intentionally mentions the relevant resource + coordination theme so
that both the Route 2 tokenizer and the Route 1 keyword baseline have signal.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - type hints only
    from .subgoal_featurizer import SubgoalContext
    from .subgoal_teacher import FeedbackIntent


# Subgoal -> (imperative verb phrase, gerund phrase, short noun phrase).
SUBGOAL_PHRASES: dict[str, tuple[str, str, str]] = {
    "GET_TOMATO": ("grab a tomato", "grabbing a tomato", "the tomato"),
    "PUT_TOMATO_IN_POT": ("put the tomato in the pot", "putting the tomato in the pot", "the tomato"),
    "GET_ONION": ("grab an onion", "grabbing an onion", "the onion"),
    "PUT_ONION_IN_POT": ("put the onion in the pot", "putting the onion in the pot", "the onion"),
    "GET_DISH": ("grab a dish", "grabbing a dish", "the dish"),
    "PICKUP_SOUP": ("plate the soup", "plating the soup", "the soup"),
    "SERVE_SOUP": ("serve the soup", "serving the soup", "the soup"),
    # Noun phrase feeds grounding keywords in validate_feedback_corpus; include
    # "staying put" so LLM paraphrases of WAIT are not dropped as ungrounded.
    "WAIT": ("hold back for a second", "holding back", "staying put / your spot"),
}

# Ordered (feature-set, theme) rules. First match wins.
POSITIVE_THEME_RULES: tuple[tuple[frozenset[str], str], ...] = (
    (frozenset({"complementary_to_human", "avoids_duplicate_human_task"}), "split_work"),
    (
        frozenset(
            {
                "respects_human_intent",
                "clears_human_path",
                "clears_serving_access",
                "clears_human_shortest_path",
                "avoids_human_shortest_path",
            }
        ),
        "make_room",
    ),
    (
        frozenset(
            {
                "supports_serving",
                "serve_ready_soup",
                "completes_recipe",
                "dish_needed_for_ready_soup",
                "pick_ready_soup",
            }
        ),
        "get_order_out",
    ),
    (
        frozenset(
            {
                "adds_needed_tomato",
                "adds_needed_onion",
                "recipe_needs_onion",
                "matches_current_order",
                "moves_toward_needed_object",
            }
        ),
        "needed_ingredient",
    ),
)

NEGATIVE_THEME_RULES: tuple[tuple[frozenset[str], str], ...] = (
    (
        frozenset(
            {
                "blocks_human_path",
                "blocks_serving_route",
                "human_wait_cost",
                "cuts_in_front_of_human",
            }
        ),
        "blocking",
    ),
    (frozenset({"steals_human_target"}), "stealing"),
    (frozenset({"duplicate_human_task", "crowds_human_target"}), "duplicating"),
    (
        frozenset(
            {"wrong_ingredient", "breaks_recipe", "adds_extra_tomato", "adds_extra_onion"}
        ),
        "wrong_ingredient",
    ),
    (frozenset({"frustrates_human"}), "annoying"),
    (
        frozenset({"delays_serving", "time_cost", "moves_away_from_needed_object"}),
        "wasting_time",
    ),
)

POSITIVE_THEME_PHRASES: dict[str, str] = {
    "split_work": "that splits the work nicely",
    "make_room": "thanks for leaving me room",
    "get_order_out": "that gets the order out",
    "needed_ingredient": "that's exactly the ingredient we need",
    "good_move": "that's the right move",
}

NEGATIVE_THEME_PHRASES: dict[str, str] = {
    "blocking": "you're blocking my way",
    "stealing": "you took the thing I needed",
    "duplicating": "we're both doing the same task",
    "wrong_ingredient": "that's the wrong ingredient for this order",
    "annoying": "that's really getting in my way",
    "wasting_time": "that just wastes time",
    "bad_move": "that's not helping",
}


def _theme(referenced_features: list[str], *, positive: bool) -> str:
    feature_set = set(referenced_features)
    rules = POSITIVE_THEME_RULES if positive else NEGATIVE_THEME_RULES
    for keys, theme in rules:
        if feature_set & keys:
            return theme
    return "good_move" if positive else "bad_move"


def theme_for_intent(intent: "FeedbackIntent") -> str:
    """Public helper: the dominant comfort/task theme of an intent."""

    return _theme(intent.referenced_features, positive=intent.polarity > 0)


def render_templates(intent: "FeedbackIntent", context: "SubgoalContext") -> list[str]:
    """Return deterministic template sentences for one feedback intent."""

    verb, gerund, _noun = SUBGOAL_PHRASES.get(
        intent.subgoal, ("do that", "doing that", "that")
    )
    role = intent.role

    if role == "command_best":
        return [
            f"Please {verb}.",
            f"Go {verb}.",
            f"{verb.capitalize()}.",
            f"You should {verb} now.",
            f"Let's have you {verb}.",
        ]

    if role == "praise_best":
        theme = POSITIVE_THEME_PHRASES[_theme(intent.referenced_features, positive=True)]
        return [
            f"Nice, {gerund} is exactly right.",
            f"Good call {gerund}, {theme}.",
            f"Yes, {gerund} works well, {theme}.",
            f"Perfect, {theme}.",
        ]

    if role == "criticize_alt":
        theme = NEGATIVE_THEME_PHRASES[_theme(intent.referenced_features, positive=False)]
        return [
            f"No, don't {verb}, {theme}.",
            f"Stop {gerund}, {theme}.",
            f"Please avoid {gerund}, {theme}.",
            f"Not {gerund} please, {theme}.",
        ]

    if role == "describe_alt":
        theme = NEGATIVE_THEME_PHRASES[_theme(intent.referenced_features, positive=False)]
        return [
            f"You're {gerund}, {theme}.",
            f"When you keep {gerund}, {theme}.",
            f"Right now {gerund} means {theme}.",
        ]

    # Fallback for any unexpected role.
    return [f"About {gerund}: {'good' if intent.polarity > 0 else 'not good'}."]


def context_description(context: "SubgoalContext") -> str:
    """A compact English description of the state, for the LLM prompt."""

    pot = ", ".join(context.pot_ingredients) if context.pot_ingredients else "empty"
    holding = context.agent_holding or "nothing"
    human = context.human_target_resource or "nothing in particular"
    return (
        f"Recipe needs {', '.join(context.recipe)}. "
        f"The pot currently has: {pot} (status: {context.pot_status}). "
        f"You (the AI) are holding {holding}. "
        f"The human teammate is going for {human}."
    )
