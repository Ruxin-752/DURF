"""One serialisation of the game state, shared by every recorder.

``play_with_baseline`` (real-human sessions) and ``sim_session`` (scripted
personas) each used to carry their own copy of these helpers, and the copies
had drifted: the sim copy summarised a pot as ``{"name": "soup"}`` with no
ingredients or cooking tick.  Nothing downstream noticed until
``durf/evaluation/matched_replay.reconstruct_state`` rebuilt every sim pot as
an empty idle pot -- silently, on all 159 sim sessions.  One module, one
shape; the replay side now refuses a soup summary without ``ingredients``
rather than guessing.
"""

from __future__ import annotations


def to_jsonable(value):
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, tuple):
        return [to_jsonable(item) for item in value]
    if isinstance(value, list):
        return [to_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if hasattr(value, "tolist"):
        return value.tolist()
    if hasattr(value, "name"):
        return str(value.name)
    return str(value)


def object_summary(obj):
    if obj is None:
        return None
    summary = {
        "type": type(obj).__name__,
        "name": getattr(obj, "name", None),
        "position": to_jsonable(getattr(obj, "position", None)),
    }
    for attr in (
        "ingredients",
        "cooking_tick",
        "is_cooking",
        "is_ready",
        "is_idle",
        "is_full",
    ):
        if hasattr(obj, attr):
            value = getattr(obj, attr)
            summary[attr] = to_jsonable(value() if callable(value) else value)
    return {key: value for key, value in summary.items() if value is not None}


def player_summary(player):
    return {
        "position": to_jsonable(getattr(player, "position", None)),
        "orientation": to_jsonable(getattr(player, "orientation", None)),
        "held_object": object_summary(getattr(player, "held_object", None)),
    }


def terrain_rows(mdp) -> list[str]:
    rows = getattr(mdp, "terrain_mtx", [])
    return ["".join(row) for row in rows]


def pot_state_summary(mdp, state):
    if not hasattr(mdp, "get_pot_states"):
        return None
    try:
        return to_jsonable(mdp.get_pot_states(state))
    except Exception as exc:
        return {"error": f"get_pot_states failed: {exc}"}


def state_facts(env) -> dict:
    state = env.base_env.state
    mdp = env.base_env.mdp
    players = list(getattr(state, "players", []))
    ai_player = players[0] if len(players) > 0 else None
    human_player = players[1] if len(players) > 1 else None
    objects = getattr(state, "objects", {})
    return {
        "ai_pos": to_jsonable(getattr(ai_player, "position", None)),
        "human_pos": to_jsonable(getattr(human_player, "position", None)),
        "ai_held_object": object_summary(getattr(ai_player, "held_object", None)),
        "human_held_object": object_summary(getattr(human_player, "held_object", None)),
        "players": [player_summary(player) for player in players],
        "objects": [
            {
                "position": to_jsonable(position),
                "object": object_summary(obj),
            }
            for position, obj in getattr(objects, "items", lambda: [])()
        ],
        "pot_states": pot_state_summary(mdp, state),
        "layout_features": {
            "layout_name": getattr(env, "layout_name", None),
            "terrain": terrain_rows(mdp),
        },
    }
