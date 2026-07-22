"""Online featurizer: (state, candidate subgoal) -> shared reward features.

This is the glue that makes the Baseline B reward learner *subgoal-compatible*.
Training produces a weight belief `w` over the feature schema in
`data/overcooked_features.json`. To let `w` re-rank H0's subgoals at runtime we
need the SAME feature vector `phi(state, subgoal)` computed online, not just the
hand-authored `phi` baked into `probe_states.json`.

This first version is deterministic and rule-based. It derives:

- task features from recipe progress / pot status (does this subgoal add a
  needed ingredient, complete the recipe, serve a ready soup, ...);
- coordination/comfort features from the human's current target (does this
  subgoal duplicate/steal what the human is doing, block the serving route,
  or complement the human).

Path-geometry features that need a MotionPlanner (exact shortest-path blocking)
are approximated from the human's declared intent; a live version can refine
them once the featurizer runs inside the pygame loop.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

from .subgoal_schema import (
    INGREDIENT_PICKUP_SUBGOALS,
    INGREDIENT_POT_SUBGOALS,
    SERVING_SUBGOALS,
    SUBGOAL_RESOURCE,
)


# Human-intent strings (or held objects) that indicate which resource the human
# is currently competing for.
_INTENT_RESOURCE_KEYWORDS: tuple[tuple[tuple[str, ...], str], ...] = (
    (("dish", "plate"), "dish"),
    (("serve", "serving", "deliver"), "serving"),
    (("soup",), "soup"),
    (("tomato",), "tomato"),
    (("onion",), "onion"),
)


def _resource_from_text(text: str | None) -> str | None:
    if not text:
        return None
    lowered = str(text).lower()
    for keywords, resource in _INTENT_RESOURCE_KEYWORDS:
        if any(keyword in lowered for keyword in keywords):
            return resource
    return None


def _missing_ingredients(recipe: list[str], current: list[str]) -> list[str]:
    missing = Counter(recipe)
    missing.subtract(Counter(current))
    ordered: list[str] = []
    for ingredient in recipe:
        if missing[ingredient] > 0:
            ordered.append(ingredient)
            missing[ingredient] -= 1
    return ordered


@dataclass
class SubgoalContext:
    """Normalized state needed to featurize a candidate subgoal."""

    recipe: list[str] = field(default_factory=lambda: ["tomato", "tomato", "onion"])
    pot_ingredients: list[str] = field(default_factory=list)
    pot_status: str = "empty"  # empty | one/partial | cooking | ready
    agent_holding: str | None = None
    human_holding: str | None = None
    human_intent: str | None = None

    @property
    def human_target_resource(self) -> str | None:
        return _resource_from_text(self.human_intent) or _resource_from_text(
            self.human_holding
        )

    @property
    def soup_ready(self) -> bool:
        return self.pot_status == "ready"

    @property
    def soup_cooking(self) -> bool:
        return self.pot_status == "cooking"

    _NATIVE_FIELDS = frozenset(
        {
            "recipe",
            "pot_ingredients",
            "pot_status",
            "agent_holding",
            "human_holding",
            "human_intent",
        }
    )
    _STATE_FACT_KEYS = frozenset(
        {
            "pot_state",
            "pot",
            "current_orders",
            "ai_held_object",
            "human_held_object",
        }
    )

    @classmethod
    def coerce(cls, context: "SubgoalContext | dict") -> "SubgoalContext":
        """Accept a native-field dict, a raw state_facts dict, or an instance."""

        if isinstance(context, SubgoalContext):
            return context
        keys = set(context)
        if keys & cls._STATE_FACT_KEYS:
            return cls.from_state_facts(context)
        return cls(**{k: v for k, v in context.items() if k in cls._NATIVE_FIELDS})

    @classmethod
    def from_state_facts(cls, facts: dict) -> "SubgoalContext":
        """Best-effort normalizer for probe `state` blocks or session state_facts."""

        recipe = ["tomato", "tomato", "onion"]
        orders = facts.get("current_orders")
        if isinstance(orders, list) and orders:
            first = orders[0]
            if isinstance(first, dict):
                recipe = list(first.get("ingredients", recipe))
            elif isinstance(first, list):
                recipe = list(first)

        pot = facts.get("pot_state") or facts.get("pot") or {}
        pot_ingredients: list[str] = []
        pot_status = "empty"
        if isinstance(pot, dict):
            pot_ingredients = list(pot.get("ingredients", []))
            pot_status = str(pot.get("status", "empty") or "empty")

        def _held(value) -> str | None:
            if isinstance(value, dict):
                return value.get("name")
            if isinstance(value, str) and value not in ("", "none", "None"):
                return value
            return None

        return cls(
            recipe=recipe,
            pot_ingredients=pot_ingredients,
            pot_status=pot_status,
            agent_holding=_held(facts.get("agent_holding") or facts.get("ai_held_object")),
            human_holding=_held(facts.get("human_holding") or facts.get("human_held_object")),
            human_intent=facts.get("human_intent"),
        )


def _task_features(context: SubgoalContext, subgoal: str, features: dict[str, float]) -> None:
    missing = _missing_ingredients(context.recipe, context.pot_ingredients)

    if subgoal in ("GET_TOMATO", "PUT_TOMATO_IN_POT"):
        features["ingredient_tomato"] = 1.0
        if subgoal == "GET_TOMATO":
            features["pick_tomato"] = 1.0
        if "tomato" in missing:
            features["adds_needed_tomato"] = 1.0
            features["moves_toward_needed_object"] = 1.0
            features["matches_current_order"] = 1.0
        else:
            features["adds_extra_tomato"] = 1.0
            features["wrong_ingredient"] = 1.0
            features["breaks_recipe"] = 1.0

    elif subgoal in ("GET_ONION", "PUT_ONION_IN_POT"):
        features["ingredient_onion"] = 1.0
        if subgoal == "GET_ONION":
            features["pick_onion"] = 1.0
        if "onion" in missing:
            features["adds_needed_onion"] = 1.0
            features["recipe_needs_onion"] = 1.0
            features["moves_toward_needed_object"] = 1.0
            features["matches_current_order"] = 1.0
        else:
            features["adds_extra_onion"] = 1.0
            features["wrong_ingredient"] = 1.0
            features["breaks_recipe"] = 1.0

    elif subgoal == "GET_DISH":
        features["pick_dish"] = 1.0
        if context.soup_ready or context.soup_cooking:
            features["dish_needed_for_ready_soup"] = 1.0
            features["supports_serving"] = 1.0
        else:
            features["time_cost"] = 1.0
            features["delays_serving"] = 1.0

    elif subgoal == "PICKUP_SOUP":
        features["pick_ready_soup"] = 1.0
        if context.soup_ready:
            features["supports_serving"] = 1.0
        else:
            features["time_cost"] = 1.0

    elif subgoal == "SERVE_SOUP":
        if context.agent_holding == "soup" or context.soup_ready:
            features["serve_ready_soup"] = 1.0
            features["supports_serving"] = 1.0
            features["completes_recipe"] = 1.0
        else:
            features["time_cost"] = 1.0

    elif subgoal == "WAIT":
        features["time_cost"] = 1.0
        if context.soup_ready or missing:
            features["delays_serving"] = 1.0


def _coordination_features(
    context: SubgoalContext, subgoal: str, features: dict[str, float]
) -> None:
    human_target = context.human_target_resource
    if human_target is None:
        return

    subgoal_resource = SUBGOAL_RESOURCE.get(subgoal)

    # Human and agent are both going for the same resource -> duplication.
    same_resource = subgoal_resource is not None and subgoal_resource == human_target
    # Treat dish/soup/serving as one shared "serving chain" resource.
    serving_chain = {"dish", "soup", "serving"}
    same_serving_chain = (
        subgoal_resource in serving_chain and human_target in serving_chain
    )

    if same_resource or same_serving_chain:
        features["duplicate_human_task"] = 1.0
        features["crowds_human_target"] = 1.0
        features["frustrates_human"] = 1.0
        # Picking up the very object the human wants steals it.
        if subgoal in ("GET_DISH", "GET_TOMATO", "GET_ONION", "PICKUP_SOUP"):
            features["steals_human_target"] = 1.0
        # Moving onto the serving route while the human needs it blocks them.
        if subgoal in SERVING_SUBGOALS:
            features["blocks_serving_route"] = 1.0
            features["blocks_human_path"] = 1.0
            features["human_wait_cost"] = 1.0
            features["cuts_in_front_of_human"] = 1.0
    elif subgoal_resource is not None:
        # Agent works on a different, still-useful resource: complementary.
        features["complementary_to_human"] = 1.0
        features["respects_human_intent"] = 1.0
        features["avoids_duplicate_human_task"] = 1.0
        if subgoal not in SERVING_SUBGOALS and human_target in serving_chain:
            # Agent stays off the serving route the human is using.
            features["clears_serving_access"] = 1.0
            features["clears_human_path"] = 1.0
    elif subgoal == "WAIT":
        # Waiting yields the contested resource to the human without interfering.
        features["respects_human_intent"] = 1.0
        features["avoids_duplicate_human_task"] = 1.0
        features["clears_human_path"] = 1.0


def featurize_subgoal(
    context: SubgoalContext | dict, subgoal: str
) -> dict[str, float]:
    """Return the shared-schema feature vector for one candidate subgoal."""

    context = SubgoalContext.coerce(context)

    features: dict[str, float] = {}
    _task_features(context, subgoal, features)
    _coordination_features(context, subgoal, features)
    return {feature: value for feature, value in features.items() if value != 0}
