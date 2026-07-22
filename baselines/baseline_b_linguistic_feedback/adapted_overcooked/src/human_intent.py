"""Infer the human's current intent for the coordination features.

The comfort/coordination features in :mod:`subgoal_featurizer` all key off
``context.human_target_resource`` -- *which resource the human is competing
for*. Offline probes hand-authored ``human_intent``; a live agent has to infer
it from what it can actually observe:

- what the human is holding (a committed signal), and
- which relevant feature the human is heading toward (a directional signal the
  runtime computes from the human's position + orientation).

This module is a pure mapping so it stays offline-testable: the runtime resolves
a ``target_resource`` from geometry and passes it in; held objects take priority
because a held object is a stronger commitment than a heading.
"""

from __future__ import annotations


# Canonical intent strings. They are phrased so that
# ``subgoal_featurizer._resource_from_text`` resolves them back to the same
# resource, keeping the featurizer as the single source of truth.
RESOURCE_TO_INTENT: dict[str, str] = {
    "tomato": "get_tomato",
    "onion": "get_onion",
    "dish": "pick_dish_then_serve",
    "soup": "pickup_soup",
    "serving": "serve_soup",
}

# What a held object implies about the resource the human is working on.
_HELD_TO_RESOURCE: dict[str, str] = {
    "tomato": "tomato",
    "onion": "onion",
    "dish": "dish",
    "soup": "serving",  # holding soup -> committed to the serving chain
}


def _resource_from_holding(human_holding: str | None) -> str | None:
    if not human_holding:
        return None
    return _HELD_TO_RESOURCE.get(str(human_holding).lower())


def infer_human_intent(
    *,
    human_holding: str | None = None,
    target_resource: str | None = None,
) -> str | None:
    """Return a canonical ``human_intent`` string, or ``None`` if unknown.

    ``human_holding`` (e.g. ``"dish"``) is a stronger, already-committed signal
    and wins over ``target_resource`` (e.g. ``"onion"`` derived from the human
    heading toward the onion dispenser).
    """

    resource = _resource_from_holding(human_holding)
    if resource is None:
        resource = (target_resource or "").lower() or None
    if resource is None:
        return None
    return RESOURCE_TO_INTENT.get(resource)


def with_inferred_intent(
    context_fields: dict,
    *,
    human_holding: str | None = None,
    target_resource: str | None = None,
) -> dict:
    """Return a copy of ``context_fields`` with ``human_intent`` filled in.

    Only overwrites ``human_intent`` when it is not already set, so an explicit
    intent (from a probe or a caller) is never clobbered.
    """

    fields = dict(context_fields)
    if fields.get("human_intent"):
        return fields
    holding = human_holding if human_holding is not None else fields.get("human_holding")
    intent = infer_human_intent(human_holding=holding, target_resource=target_resource)
    if intent is not None:
        fields["human_intent"] = intent
    return fields
