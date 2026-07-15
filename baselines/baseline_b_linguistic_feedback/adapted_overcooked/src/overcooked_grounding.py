"""Ground linguistic feedback to Overcooked feature vectors."""

from __future__ import annotations


KEYWORD_FEATURES: tuple[tuple[tuple[str, ...], dict[str, float]], ...] = (
    (
        ("block", "blocking", "in my way", "堵", "挡", "卡", "路"),
        {
            "blocks_human_path": 1,
            "human_wait_cost": 1,
            "frustrates_human": 1,
        },
    ),
    (
        ("move away", "step aside", "clear", "让开", "移开"),
        {
            "clears_human_path": 1,
            "respects_human_intent": 1,
        },
    ),
    (
        ("dish", "盘"),
        {
            "pick_dish": 1,
            "supports_serving": 1,
        },
    ),
    (
        ("tomato", "番茄"),
        {
            "ingredient_tomato": 1,
            "pick_tomato": 1,
        },
    ),
    (
        ("onion", "洋葱", "葱"),
        {
            "ingredient_onion": 1,
            "pick_onion": 1,
            "recipe_needs_onion": 1,
        },
    ),
    (
        ("duplicate", "repeat", "same task", "重复", "也去拿"),
        {
            "duplicate_human_task": 1,
            "crowds_human_target": 1,
            "frustrates_human": 1,
        },
    ),
    (
        ("shortest path", "cut in", "插", "最短路径"),
        {
            "cuts_in_front_of_human": 1,
            "blocks_human_path": 1,
            "human_wait_cost": 1,
        },
    ),
    (
        ("serve", "serving", "出餐", "上菜"),
        {
            "serve_ready_soup": 1,
            "supports_serving": 1,
            "blocks_serving_route": 1,
        },
    ),
)


def merge_features(*vectors: dict[str, float]) -> dict[str, float]:
    merged: dict[str, float] = {}
    for vector in vectors:
        for feature, value in vector.items():
            merged[feature] = merged.get(feature, 0.0) + float(value)
    return {feature: value for feature, value in merged.items() if value != 0}


def features_from_keywords(text: str | None) -> dict[str, float]:
    lowered = (text or "").lower()
    matched = []
    for keywords, features in KEYWORD_FEATURES:
        if any(keyword in lowered for keyword in keywords):
            matched.append(features)
    return merge_features(*matched)


def ground_feedback(
    feedback: dict,
    *,
    feedback_type: str,
    action_feature_library: dict[str, dict[str, float]] | None = None,
) -> dict:
    """Return target features and grounding metadata for one feedback example."""

    text = feedback.get("text") or feedback.get("feedback_text")
    action_feature_library = action_feature_library or {}

    explicit_target_features = feedback.get("target_features")
    if isinstance(explicit_target_features, dict) and explicit_target_features:
        return {
            "target_features": {
                str(feature): float(value)
                for feature, value in explicit_target_features.items()
            },
            "grounding_source": "target_features",
        }

    if feedback_type == "evaluative":
        target_features = feedback.get("trajectory_features") or {}
        return {
            "target_features": {str(k): float(v) for k, v in target_features.items()},
            "grounding_source": "trajectory_features",
        }

    target_action = feedback.get("target_action")
    if feedback_type == "imperative" and target_action in action_feature_library:
        return {
            "target_features": action_feature_library[target_action],
            "grounding_source": f"target_action:{target_action}",
        }

    target_features = features_from_keywords(text)
    return {
        "target_features": target_features,
        "grounding_source": "keyword_features",
    }
