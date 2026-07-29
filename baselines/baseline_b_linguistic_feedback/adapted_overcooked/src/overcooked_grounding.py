"""Ground linguistic feedback to Overcooked feature vectors."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_GROUNDING_MODEL_PATH = ROOT / "outputs" / "phrase_grounding" / "model.joblib"
OPPOSITE_FEATURES = (
    ("blocks_human_path", "clears_human_path"),
    ("blocks_human_path", "clears_human_shortest_path"),
    ("blocks_serving_route", "clears_serving_access"),
    ("duplicate_human_task", "avoids_duplicate_human_task"),
    ("moves_toward_needed_object", "moves_away_from_needed_object"),
    ("supports_serving", "delays_serving"),
)

KEYWORD_FEATURES: tuple[tuple[tuple[str, ...], dict[str, float]], ...] = (
    (
        ("block", "blocking", "in my way"),
        {
            "blocks_human_path": 1,
            "human_wait_cost": 1,
            "frustrates_human": 1,
        },
    ),
    (
        ("move away", "step aside", "clear"),
        {
            "clears_human_path": 1,
            "respects_human_intent": 1,
        },
    ),
    (
        ("dish",),
        {
            "pick_dish": 1,
            "supports_serving": 1,
        },
    ),
    (
        ("tomato",),
        {
            "ingredient_tomato": 1,
            "pick_tomato": 1,
        },
    ),
    (
        ("onion",),
        {
            "ingredient_onion": 1,
            "pick_onion": 1,
            "recipe_needs_onion": 1,
        },
    ),
    (
        ("duplicate", "repeat", "same task"),
        {
            "duplicate_human_task": 1,
            "crowds_human_target": 1,
            "frustrates_human": 1,
        },
    ),
    (
        ("shortest path", "cut in"),
        {
            "cuts_in_front_of_human": 1,
            "blocks_human_path": 1,
            "human_wait_cost": 1,
        },
    ),
    (
        ("serve", "serving"),
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
    return resolve_opposite_features(
        {feature: value for feature, value in merged.items() if value != 0}
    )


def resolve_opposite_features(vector: dict[str, float]) -> dict[str, float]:
    """Ensure contradictory binary reward features cannot be active together."""

    resolved = {str(feature): float(value) for feature, value in vector.items() if value}
    for left, right in OPPOSITE_FEATURES:
        if left not in resolved or right not in resolved:
            continue
        left_value, right_value = abs(resolved[left]), abs(resolved[right])
        if left_value == right_value:
            # Equal evidence is ambiguous and must not become two observations.
            resolved.pop(left, None)
            resolved.pop(right, None)
        elif left_value > right_value:
            resolved.pop(right, None)
        else:
            resolved.pop(left, None)
    return resolved


def features_from_keywords(text: str | None) -> dict[str, float]:
    lowered = (text or "").lower()
    matched = []
    for keywords, features in KEYWORD_FEATURES:
        if any(keyword in lowered for keyword in keywords):
            matched.append(features)
    return merge_features(*matched)


@lru_cache(maxsize=4)
def _load_grounding_artifact(path: str):
    from joblib import load

    return load(path)


def _processed(text: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", text.lower()))


def build_grounding_rows(examples: list[dict]) -> list[dict]:
    """Build phrase-level multilabel rows, preferring explicit annotations."""

    from .text_analysis import limited_punc_tokenization

    rows: list[dict] = []
    for example in examples:
        annotations = example.get("phrase_annotations")
        annotated = []
        if isinstance(annotations, list):
            for annotation in annotations:
                if not isinstance(annotation, dict):
                    continue
                phrase = annotation.get("phrase") or annotation.get("text")
                features = annotation.get("target_features") or annotation.get("features")
                if isinstance(phrase, str) and isinstance(features, dict) and features:
                    annotated.append((phrase, features))
        pairs = annotated or [
            (phrase, example.get("target_features") or {})
            for phrase in limited_punc_tokenization(example.get("text") or "")
        ]
        for phrase, features in pairs:
            active = sorted(str(key) for key, value in features.items() if float(value) != 0)
            if active:
                rows.append(
                    {
                        "text": phrase,
                        "processed": _processed(phrase),
                        "labels": active,
                        "annotation_source": (
                            "phrase_annotations" if annotated else "utterance_target_features"
                        ),
                    }
                )
    return rows


def train_phrase_grounding(
    examples: list[dict],
    *,
    min_df: int = 2,
    seed: int = 1,
) -> dict:
    """Train word TF-IDF + OneVsRest LogisticRegression multilabel grounding."""

    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.multiclass import OneVsRestClassifier
    from sklearn.preprocessing import MultiLabelBinarizer

    rows = build_grounding_rows(examples)
    if not rows:
        raise ValueError("no phrase grounding labels found")
    vectorizer = TfidfVectorizer(
        sublinear_tf=True, min_df=min_df, ngram_range=(1, 2), stop_words="english"
    )
    matrix = vectorizer.fit_transform([row["processed"] for row in rows])
    label_binarizer = MultiLabelBinarizer()
    targets = label_binarizer.fit_transform([row["labels"] for row in rows])
    classifier = OneVsRestClassifier(
        LogisticRegression(max_iter=2000, class_weight="balanced", random_state=seed)
    )
    classifier.fit(matrix, targets)
    prevalence = targets.mean(axis=0)
    thresholds = {
        str(label): float(min(0.7, max(0.35, 0.5 - 0.2 * rate)))
        for label, rate in zip(label_binarizer.classes_, prevalence)
    }
    return {
        "vectorizer": vectorizer,
        "classifier": classifier,
        "label_binarizer": label_binarizer,
        "feature_thresholds": thresholds,
        "minimum_confidence": 0.35,
        "training_rows": len(rows),
    }


def predict_grounded_features(
    text: str,
    *,
    model_path: str | Path = DEFAULT_GROUNDING_MODEL_PATH,
    feasible_features: set[str] | None = None,
) -> dict:
    """Load the multilabel model and apply a context-feasibility mask."""

    path = Path(model_path)
    if not path.exists():
        return {
            "target_features": {},
            "grounding_confidence": 0.0,
            "grounding_source": "model_unavailable",
            "abstained": True,
        }
    artifact = _load_grounding_artifact(str(path.resolve()))
    matrix = artifact["vectorizer"].transform([_processed(text)])
    probabilities = artifact["classifier"].predict_proba(matrix)[0]
    labels = artifact["label_binarizer"].classes_
    thresholds = artifact.get("feature_thresholds") or {}
    selected = {
        str(label): float(probability)
        for label, probability in zip(labels, probabilities)
        if probability >= float(thresholds.get(str(label), 0.5))
        and (feasible_features is None or str(label) in feasible_features)
    }
    selected = resolve_opposite_features(selected)
    confidence = max(selected.values(), default=0.0)
    minimum = float(artifact.get("minimum_confidence", 0.35))
    return {
        "target_features": {feature: 1.0 for feature in selected},
        "feature_probabilities": selected,
        "grounding_confidence": confidence,
        "grounding_source": "tfidf_ovr_logistic_regression",
        "abstained": not selected or confidence < minimum,
        "model_path": str(path),
    }


def _recent_event_grounding(feedback: dict) -> dict[str, float]:
    """Select the nearest recent event, avoiding indiscriminate trajectory sums."""

    events = feedback.get("recent_events")
    if not isinstance(events, list):
        return {}
    candidates = []
    current_step = int(feedback.get("total_step") or 0)
    for order, event in enumerate(events):
        if not isinstance(event, dict):
            continue
        features = event.get("features") or event.get("trajectory_features")
        if not isinstance(features, dict) or not features:
            continue
        step = int(event.get("total_step", current_step - (len(events) - order)))
        candidates.append((abs(current_step - step), -step, features))
    if not candidates:
        return {}
    return resolve_opposite_features(min(candidates, key=lambda item: item[:2])[2])


def ground_feedback(
    feedback: dict,
    *,
    feedback_type: str | None = None,
    reference_type: str | None = None,
    action_feature_library: dict[str, dict[str, float]] | None = None,
    grounding_model_path: str | Path = DEFAULT_GROUNDING_MODEL_PATH,
) -> dict:
    """Return target features and grounding metadata for one feedback example."""

    text = feedback.get("text") or feedback.get("feedback_text")
    action_feature_library = action_feature_library or {}

    explicit_target_features = feedback.get("target_features")
    if isinstance(explicit_target_features, dict) and explicit_target_features:
        return {
            "target_features": resolve_opposite_features({
                str(feature): float(value)
                for feature, value in explicit_target_features.items()
            }),
            "grounding_source": "target_features",
            "grounding_confidence": 1.0,
        }

    if reference_type == "other":
        return {
            "target_features": {},
            "grounding_source": "other",
            "grounding_confidence": 1.0,
            "abstained": True,
        }

    if reference_type in {"trajectory", "action_behavioral"} or (
        reference_type is None and feedback_type == "evaluative"
    ):
        target_features = (
            _recent_event_grounding(feedback)
            or feedback.get("event_features")
            or feedback.get("trajectory_features")
            or {}
        )
        return {
            "target_features": resolve_opposite_features(
                {str(k): float(v) for k, v in target_features.items()}
            ),
            "grounding_source": (
                "recent_event_features"
                if feedback.get("recent_events") and target_features
                else "trajectory_features"
            ),
            "grounding_confidence": 0.95 if target_features else 0.0,
        }

    target_action = feedback.get("target_action")
    if (
        reference_type == "action_spatial"
        or (reference_type is None and feedback_type == "imperative")
    ) and target_action in action_feature_library:
        return {
            "target_features": resolve_opposite_features(action_feature_library[target_action]),
            "grounding_source": f"target_action:{target_action}",
            "grounding_confidence": 0.9,
        }

    feasible_features = {
        str(feature)
        for vector in action_feature_library.values()
        for feature, value in vector.items()
        if float(value) != 0
    } or None
    learned = predict_grounded_features(
        str(text or ""),
        model_path=grounding_model_path,
        feasible_features=feasible_features,
    )
    if not learned["abstained"]:
        return learned
    target_features = features_from_keywords(text)
    if feasible_features is not None:
        target_features = {
            feature: value
            for feature, value in target_features.items()
            if feature in feasible_features
        }
    return {
        "target_features": target_features,
        "grounding_source": "keyword_features",
        "grounding_confidence": 0.65 if target_features else 0.0,
        "abstained": not bool(target_features),
        "model_abstained": True,
    }
