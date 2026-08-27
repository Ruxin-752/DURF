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


# Only these reward dimensions can change the ordering of live H0 candidates.
# Keeping this contract explicit prevents a language update from appearing
# successful when its feature is constant across every feasible subgoal (or is
# not observable at all in the current live adapter).
LIVE_DECISION_FEATURES: frozenset[str] = frozenset(
    {
        "adds_needed_onion",
        "adds_needed_tomato",
        "avoids_human_shortest_path",
        "avoids_duplicate_human_task",
        "blocks_human_path",
        "blocks_serving_route",
        "clears_human_path",
        "clears_human_shortest_path",
        "clears_serving_access",
        "complementary_to_human",
        "completes_recipe",
        "crowds_human_target",
        "cuts_in_front_of_human",
        "delays_serving",
        "dish_needed_for_ready_soup",
        "distance_cost",
        "duplicate_human_task",
        "frustrates_human",
        "human_wait_cost",
        "ingredient_onion",
        "ingredient_tomato",
        "matches_current_order",
        "moves_toward_needed_object",
        "pick_dish",
        "pick_onion",
        "pick_ready_soup",
        "pick_tomato",
        "recipe_needs_onion",
        "respects_human_intent",
        "serve_ready_soup",
        "steals_human_target",
        "supports_serving",
        "time_cost",
    }
)

# These state facts are observable, but they are shared by all candidates in a
# decision and therefore cancel out of argmax(w dot phi).
LIVE_CONTEXT_ONLY_FEATURES: frozenset[str] = frozenset(
    {
        "pot_cooking",
        "pot_empty",
        "pot_has_one_tomato",
        "pot_has_two_tomatoes",
        "pot_has_two_tomatoes_one_onion",
        "soup_ready",
    }
)


def candidate_varying_features(
    candidate_vectors: list[dict[str, float]],
) -> frozenset[str]:
    """Features whose value differs between candidates in one decision."""

    if not candidate_vectors:
        return frozenset()
    names = {str(feature) for vector in candidate_vectors for feature in vector}
    return frozenset(
        feature
        for feature in names
        if len({float(vector.get(feature, 0.0)) for vector in candidate_vectors}) > 1
    )


def audit_live_feature_coverage(
    schema_features: list[str] | tuple[str, ...] | set[str],
    *,
    candidate_sets: list[list[dict[str, float]]] | None = None,
) -> dict:
    """Partition the full schema and optionally verify candidate variation.

    Unsupported dimensions are deliberately named rather than silently kept in
    the live learner.  When representative candidate sets are supplied, the
    audit also proves that every dimension declared live-decision-relevant
    varies in at least one real candidate set.
    """

    schema = {str(feature) for feature in schema_features}
    decision = schema & set(LIVE_DECISION_FEATURES)
    context_only = schema & set(LIVE_CONTEXT_ONLY_FEATURES)
    unsupported = schema - decision - context_only
    varying = set()
    observed = set()
    for candidates in candidate_sets or []:
        varying.update(candidate_varying_features(candidates))
        observed.update(str(feature) for vector in candidates for feature in vector)
    checked = candidate_sets is not None
    missing_decision = decision - varying if checked else set()
    return {
        "schema_size": len(schema),
        "decision_feature_count": len(decision),
        "context_only_feature_count": len(context_only),
        "unsupported_feature_count": len(unsupported),
        "decision_features": sorted(decision),
        "context_only_features": sorted(context_only),
        "unsupported_features": sorted(unsupported),
        "decision_null_features": sorted(context_only | unsupported),
        "observed_features": sorted(observed),
        "varying_features": sorted(varying),
        "missing_declared_decision_features": sorted(missing_decision),
        "undeclared_varying_features": sorted(varying - decision),
        "candidate_variation_checked": checked,
        "complete": not checked or (not missing_decision and not (varying - decision)),
    }


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
    # Live geometry supplied by ``durf.baseline.comfort_reward``.  Keeping the
    # facts in the context (rather than guessing from a subgoal name) lets the
    # offline scorer remain independent of the Overcooked runtime.
    human_committed_units: int | None = None
    candidate_path_effects: dict[str, str] = field(default_factory=dict)
    candidate_target_overlaps_human: dict[str, bool] = field(default_factory=dict)
    # Optional full live-pot view. The singular fields above remain the
    # backwards-compatible focus pot used by legacy probes and feedback text.
    pot_snapshots: list[dict] | None = None
    # Resources already placed on ordinary counters.  These are completed
    # staging work, not fresh pickup targets, while every pot is closed.
    staged_inventory: dict[str, int] = field(default_factory=dict)

    @property
    def human_target_resource(self) -> str | None:
        return _resource_from_text(self.human_intent) or _resource_from_text(
            self.human_holding
        )

    @property
    def soup_ready(self) -> bool:
        return self.ready_soup_count > 0

    @property
    def soup_cooking(self) -> bool:
        return any(
            snapshot["status"] == "cooking"
            for snapshot in self.normalized_pot_snapshots
        )

    @property
    def normalized_pot_snapshots(self) -> list[dict]:
        """All pots, falling back exactly to the legacy singular snapshot."""

        normalized: list[dict] = []
        for raw in self.pot_snapshots or []:
            if isinstance(raw, dict):
                ingredients = list(raw.get("ingredients") or [])
                status = str(raw.get("status", "empty") or "empty").lower()
                snapshot = {"ingredients": ingredients, "status": status}
                if "position" in raw:
                    snapshot["position"] = raw["position"]
                normalized.append(snapshot)
            elif isinstance(raw, (list, tuple)) and len(raw) >= 2:
                normalized.append(
                    {
                        "ingredients": list(raw[0] or []),
                        "status": str(raw[1] or "empty").lower(),
                    }
                )
        if normalized:
            return normalized
        return [
            {
                "ingredients": list(self.pot_ingredients),
                "status": str(self.pot_status or "empty").lower(),
            }
        ]

    @property
    def ready_soup_count(self) -> int:
        return sum(
            snapshot["status"] == "ready"
            for snapshot in self.normalized_pot_snapshots
        )

    @property
    def cooking_soup_count(self) -> int:
        return sum(
            snapshot["status"] == "cooking"
            for snapshot in self.normalized_pot_snapshots
        )

    def staged_units(self, resource: str) -> int:
        """Non-negative count of a resource already staged on counters."""

        try:
            return max(0, int((self.staged_inventory or {}).get(resource, 0)))
        except (TypeError, ValueError):
            return 0

    @property
    def open_missing_ingredients(self) -> list[str]:
        """Recipe units still accepted by any non-cooking, non-ready pot."""

        missing: list[str] = []
        for snapshot in self.normalized_pot_snapshots:
            if snapshot["status"] in {"ready", "cooking"}:
                continue
            missing.extend(
                _missing_ingredients(self.recipe, snapshot["ingredients"])
            )
        return missing

    @property
    def has_open_recipe_work(self) -> bool:
        return bool(self.open_missing_ingredients)

    _NATIVE_FIELDS = frozenset(
        {
            "recipe",
            "pot_ingredients",
            "pot_status",
            "agent_holding",
            "human_holding",
            "human_intent",
            "human_committed_units",
            "candidate_path_effects",
            "candidate_target_overlaps_human",
            "pot_snapshots",
            "staged_inventory",
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
            human_committed_units=facts.get("human_committed_units"),
            candidate_path_effects=dict(facts.get("candidate_path_effects") or {}),
            candidate_target_overlaps_human={
                str(key): bool(value)
                for key, value in (
                    facts.get("candidate_target_overlaps_human") or {}
                ).items()
            },
            pot_snapshots=(
                list(facts.get("pot_snapshots") or [])
                if facts.get("pot_snapshots") is not None
                else None
            ),
            staged_inventory={
                str(resource): int(count)
                for resource, count in (facts.get("staged_inventory") or {}).items()
            },
        )


def _state_features(context: SubgoalContext, features: dict[str, float]) -> None:
    """State facts shared by every candidate; they cancel out in the argmax."""

    for snapshot in context.normalized_pot_snapshots:
        ingredients = snapshot["ingredients"]
        counts = Counter(ingredients)
        if not ingredients:
            features["pot_empty"] = 1.0
        if counts["tomato"] >= 1:
            features["pot_has_one_tomato"] = 1.0
        if counts["tomato"] >= 2:
            features["pot_has_two_tomatoes"] = 1.0
        if counts["tomato"] >= 2 and counts["onion"] >= 1:
            features["pot_has_two_tomatoes_one_onion"] = 1.0
    if context.soup_cooking:
        features["pot_cooking"] = 1.0
    if context.soup_ready:
        features["soup_ready"] = 1.0


def _task_features(context: SubgoalContext, subgoal: str, features: dict[str, float]) -> None:
    missing = context.open_missing_ingredients

    if subgoal in ("GET_TOMATO", "PUT_TOMATO_IN_POT"):
        features["ingredient_tomato"] = 1.0
        if subgoal == "GET_TOMATO":
            features["pick_tomato"] = 1.0
        if "tomato" in missing:
            features["adds_needed_tomato"] = 1.0
            features["moves_toward_needed_object"] = 1.0
            features["matches_current_order"] = 1.0
        elif subgoal == "GET_TOMATO" and context.soup_cooking:
            # The repeated layout recipe makes this a factual next-round move;
            # whether that is desirable is represented only by its weight.
            features["moves_toward_needed_object"] = 1.0
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
        elif subgoal == "GET_ONION" and context.soup_cooking:
            features["moves_toward_needed_object"] = 1.0
        else:
            features["adds_extra_onion"] = 1.0
            features["wrong_ingredient"] = 1.0
            features["breaks_recipe"] = 1.0

    elif subgoal == "GET_DISH":
        features["pick_dish"] = 1.0
        if context.soup_ready:
            features["dish_needed_for_ready_soup"] = 1.0
            features["supports_serving"] = 1.0
        elif not context.soup_cooking:
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

    elif subgoal == "STASH_HELD_OBJECT":
        # Stashing frees a hand when the object cannot advance the task, or
        # when another actor has already committed all remaining demand. Once
        # soup is ready this is a factual prerequisite for plating; otherwise
        # it consumes one step. No hand-authored preference bonus is added.
        if context.soup_ready:
            features["supports_serving"] = 1.0
        else:
            features["time_cost"] = 1.0

    elif subgoal == "YIELD_PATH":
        # The live executor exposes this candidate only for a legal one-tile
        # move out of the teammate's explicitly faced next cell. These are
        # factual action costs; learned weights decide whether clearing wins.
        features["time_cost"] = 1.0
        features["distance_cost"] = 1.0

    elif subgoal == "WAIT":
        # Waiting while a complete soup cooks is a neutral preparation choice.
        # Its desirability comes from learned weights on the pickup alternatives.
        passive_cooking_only = (
            context.soup_cooking
            and not context.soup_ready
            and not context.has_open_recipe_work
        )
        if not passive_cooking_only or context.agent_holding is not None:
            features["time_cost"] = 1.0
        if context.soup_ready or missing:
            features["delays_serving"] = 1.0
        # Once the agent is holding a currently unusable object, waiting
        # preserves the blockage while stashing frees the hand. This is a
        # factual task-state distinction; the learned weight decides whether
        # its one-step cost is acceptable.
        if context.agent_holding is not None and subgoal == "WAIT":
            features["delays_serving"] = 1.0


def _coordination_features(
    context: SubgoalContext, subgoal: str, features: dict[str, float]
) -> None:
    human_target = context.human_target_resource
    path_effect = context.candidate_path_effects.get(subgoal)
    blocks_path = path_effect in {"blocks", "enters"}
    clears_path = path_effect == "clears"
    if human_target is None:
        # A directly faced adjacent cell is itself an explicit immediate path
        # intent, even if the human's eventual task resource is unknown.
        if subgoal == "YIELD_PATH" and clears_path:
            features["respects_human_intent"] = 1.0
            features["clears_human_path"] = 1.0
            features["clears_human_shortest_path"] = 1.0
            features["avoids_human_shortest_path"] = 1.0
        return

    subgoal_resource = SUBGOAL_RESOURCE.get(subgoal)
    serving_chain = {"dish", "soup", "serving"}
    missing = Counter(context.open_missing_ingredients)

    def remaining_units(resource: str) -> int:
        if resource in ("tomato", "onion"):
            units = int(missing[resource])
            # During cooking, ingredient pickup is preparation for the next
            # identical order rather than an extra ingredient for this pot.
            if units == 0 and context.soup_cooking:
                units = int(Counter(context.recipe)[resource])
            return units
        if resource in {"dish", "soup"}:
            return context.ready_soup_count
        if resource == "serving":
            held_soups = int(context.agent_holding == "soup") + int(
                context.human_holding == "soup"
            )
            return context.ready_soup_count + held_soups
        return 0

    committed_units = context.human_committed_units
    if committed_units is None:
        # A declared target means one unit of committed work.  Live contexts
        # set this explicitly from the held object.
        committed_units = 1
    same_resource = subgoal_resource is not None and subgoal_resource == human_target
    target_overlap = context.candidate_target_overlaps_human.get(
        subgoal,
        # Offline/probe contexts have no geometry. Exact resource identity is
        # the conservative fallback; live contexts always provide the map.
        same_resource,
    )
    same_work = same_resource or (
        target_overlap
        and human_target in serving_chain
        and subgoal_resource in serving_chain
    )
    # Once both soups are already held, they are distinct delivery units. A
    # shared serving counter can still create a geometric crowd/path conflict,
    # but serving either soup does not duplicate serving the other one.
    if (
        subgoal == "SERVE_SOUP"
        and context.agent_holding == "soup"
        and context.human_holding == "soup"
    ):
        same_work = False
    human_covers_work = (
        same_work
        and remaining_units(human_target) > 0
        and int(committed_units) >= remaining_units(human_target)
    )
    releases_covered_duplicate = (
        subgoal == "STASH_HELD_OBJECT"
        and context.agent_holding in ("tomato", "onion")
        and context.agent_holding == context.human_holding == human_target
        and remaining_units(human_target) > 0
        and int(committed_units) >= remaining_units(human_target)
    )

    if releases_covered_duplicate:
        # This is the state-level opposite of duplicating already-covered
        # ingredient work. Learned weights still choose between STASH, PUT and
        # WAIT; the feature does not force an action.
        features["respects_human_intent"] = 1.0
        features["avoids_duplicate_human_task"] = 1.0
    elif human_covers_work:
        features["duplicate_human_task"] = 1.0
        if target_overlap:
            features["crowds_human_target"] = 1.0
        # An object already held by the human cannot be stolen.  A pickup only
        # steals a target when both agents are approaching the same live target.
        if (
            target_overlap
            and context.human_holding is None
            and subgoal in ("GET_DISH", "GET_TOMATO", "GET_ONION", "PICKUP_SOUP")
        ):
            features["steals_human_target"] = 1.0
    elif subgoal_resource is not None:
        # Different work -- including another still-needed unit of the same
        # ingredient -- is complementary rather than duplicate.
        features["complementary_to_human"] = 1.0
        features["respects_human_intent"] = 1.0
        features["avoids_duplicate_human_task"] = 1.0
    elif subgoal == "WAIT":
        # Waiting is a yield only when the human has already covered all of the
        # target work and staying put does not physically block that work.
        target_units = remaining_units(human_target)
        if (
            target_units > 0
            and int(committed_units) >= target_units
            and not blocks_path
        ):
            features["respects_human_intent"] = 1.0
            features["avoids_duplicate_human_task"] = 1.0

    # Path/route features come only from live occupancy geometry.  A subgoal
    # name alone is not evidence that it blocks or clears anybody.
    if blocks_path:
        features["blocks_human_path"] = 1.0
        features["human_wait_cost"] = 1.0
        features["frustrates_human"] = 1.0
        if path_effect == "enters":
            features["cuts_in_front_of_human"] = 1.0
        if human_target in serving_chain:
            features["blocks_serving_route"] = 1.0
    elif clears_path:
        features["clears_human_path"] = 1.0
        features["clears_human_shortest_path"] = 1.0
        features["avoids_human_shortest_path"] = 1.0
        if subgoal == "YIELD_PATH":
            features["respects_human_intent"] = 1.0
        if human_target in serving_chain:
            features["clears_serving_access"] = 1.0


def featurize_subgoal(
    context: SubgoalContext | dict, subgoal: str
) -> dict[str, float]:
    """Return the shared-schema feature vector for one candidate subgoal."""

    context = SubgoalContext.coerce(context)

    features: dict[str, float] = {}
    _state_features(context, features)
    _task_features(context, subgoal, features)
    _coordination_features(context, subgoal, features)
    return {feature: value for feature, value in features.items() if value != 0}
