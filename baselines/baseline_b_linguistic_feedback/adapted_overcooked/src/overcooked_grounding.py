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
    from sklearn.pipeline import FeatureUnion
    from sklearn.linear_model import LogisticRegression
    from sklearn.multiclass import OneVsRestClassifier
    from sklearn.preprocessing import MultiLabelBinarizer

    rows = build_grounding_rows(examples)
    if not rows:
        raise ValueError("no phrase grounding labels found")
    # Raw word and character features preserve short function words (for
    # example "you", "again", and negation) that carry grounding scope.  The
    # configuration is fixed using dev only; test is never used for selection.
    vectorizer = FeatureUnion(
        [
            (
                "word",
                TfidfVectorizer(
                    sublinear_tf=True,
                    min_df=min_df,
                    ngram_range=(1, 2),
                    strip_accents="unicode",
                ),
            ),
            (
                "char",
                TfidfVectorizer(
                    analyzer="char_wb",
                    sublinear_tf=True,
                    min_df=min_df,
                    ngram_range=(3, 5),
                    strip_accents="unicode",
                ),
            ),
        ]
    )
    matrix = vectorizer.fit_transform([row["text"] for row in rows])
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
        "input_mode": "raw_text",
        "feature_config": {
            "word_ngram_range": [1, 2],
            "char_wb_ngram_range": [3, 5],
            "min_df": min_df,
            "sublinear_tf": True,
            "strip_accents": "unicode",
        },
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
    model_input = text if artifact.get("input_mode") == "raw_text" else _processed(text)
    matrix = artifact["vectorizer"].transform([model_input])
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


def _valid_recent_events(feedback: dict) -> list[dict]:
    """Return ordered, feature-bearing events that do not occur in the future."""

    events = feedback.get("recent_events")
    if not isinstance(events, list):
        return []
    explicit_current = feedback.get("total_step") is not None
    parsed = []
    for order, event in enumerate(events):
        if not isinstance(event, dict):
            continue
        features = event.get("features") or event.get("trajectory_features")
        if not isinstance(features, dict) or not features:
            continue
        step = int(event.get("total_step", order + 1))
        parsed.append(
            {
                "total_step": step,
                "order": order,
                "features": {
                    str(feature): float(value)
                    for feature, value in features.items()
                    if float(value) != 0
                },
            }
        )
    if not parsed:
        return []
    current_step = (
        int(feedback["total_step"])
        if explicit_current
        else max(event["total_step"] for event in parsed)
    )
    return [event for event in parsed if event["total_step"] <= current_step]


def _recent_event_grounding(feedback: dict) -> dict:
    """Select one nearest prior event for a single-action reference."""

    events = _valid_recent_events(feedback)
    if not events:
        return {}
    current_step = int(
        feedback.get("total_step")
        if feedback.get("total_step") is not None
        else max(event["total_step"] for event in events)
    )
    event = min(
        events,
        key=lambda item: (current_step - item["total_step"], -item["order"]),
    )
    return {
        "target_features": resolve_opposite_features(event["features"]),
        "target_event_step": event["total_step"],
        "event_distance": current_step - event["total_step"],
        "recent_event_count": len(events),
    }


def _repeated_event_grounding(feedback: dict) -> dict:
    """Ground a behavioral reference only to features repeated across events."""

    events = _valid_recent_events(feedback)
    occurrence: dict[str, int] = {}
    totals: dict[str, float] = {}
    steps: dict[str, list[int]] = {}
    for event in events:
        for feature, value in event["features"].items():
            if not value:
                continue
            occurrence[feature] = occurrence.get(feature, 0) + 1
            totals[feature] = totals.get(feature, 0.0) + float(value)
            steps.setdefault(feature, []).append(int(event["total_step"]))
    repeated = {
        feature: totals[feature]
        for feature, count in occurrence.items()
        if count >= 2
    }
    if not repeated:
        return {}

    # Textual evidence narrows the repeated pattern when it can.  Falling back
    # to the repeated context remains faithful to the paper's behavior-frequency
    # reference and is explicitly lower confidence.
    text = str(feedback.get("text") or feedback.get("feedback_text") or "")
    named = features_from_keywords(text)
    named_repeated = {
        feature: value for feature, value in repeated.items() if feature in named
    }
    selected = named_repeated or repeated
    selected_steps = sorted(
        {step for feature in selected for step in steps.get(feature, [])}
    )
    return {
        "target_features": resolve_opposite_features(selected),
        "target_event_steps": selected_steps,
        "target_event_span": (
            [selected_steps[0], selected_steps[-1]] if selected_steps else None
        ),
        "repeat_count": min(occurrence[feature] for feature in selected),
        "recent_event_count": len(events),
        "text_narrowed": bool(named_repeated),
    }


def _apply_live_feature_mask(result: dict, feature_mask: set[str] | None) -> dict:
    """Make decision-null grounding explicit instead of silently updating it."""

    if feature_mask is None:
        return result
    result = dict(result)
    original = dict(result.get("target_features") or {})
    kept = {
        feature: value for feature, value in original.items() if feature in feature_mask
    }
    masked = sorted(set(original) - set(kept))
    result["target_features"] = kept
    result["live_feature_mask_applied"] = True
    result["masked_target_features"] = masked
    if original and not kept:
        result["abstained"] = True
        result["grounding_confidence"] = 0.0
        result["abstention_reason"] = "decision_null_or_unsupported_features"
    return result


_FORM_REFERENCE_DEFAULT = {
    "evaluative": "trajectory",
    "imperative": "action_spatial",
    "descriptive": "feature",
}
_FORM_REFERENCE_SUBTYPES = {
    # ``action_behavioral`` still denotes evidence in the executed trajectory;
    # it may refine temporal credit, but it must not turn an Evaluative phrase
    # into an action or feature update.
    "evaluative": frozenset({"trajectory", "action_behavioral"}),
    "imperative": frozenset({"action_spatial"}),
    "descriptive": frozenset({"feature"}),
}


def constrain_reference_type(
    feedback_type: str | None,
    reference_type: str | None,
) -> dict:
    """Keep a fine reference subtype inside the paper's coarse ``f_G`` form.

    The three-way feedback form owns credit assignment.  The five-way
    reference classifier may refine a compatible branch, while a cross-form
    prediction is logged and projected back to the coarse form's default.
    ``other`` remains a safety abstention rather than being forced into a
    learnable branch.
    """

    coarse = str(feedback_type or "").lower()
    requested = str(reference_type or "").lower() or None
    default = _FORM_REFERENCE_DEFAULT.get(coarse)
    if default is None:
        return {
            "feedback_form": coarse or "unknown",
            "requested_reference_type": requested,
            "effective_reference_type": "other",
            "reference_conflict": False,
            "reference_rejected": True,
            "reference_rejection_reason": "unknown_feedback_form",
        }
    if requested == "other":
        return {
            "feedback_form": coarse,
            "requested_reference_type": requested,
            "effective_reference_type": "other",
            "reference_conflict": False,
            "reference_rejected": True,
            "reference_rejection_reason": "reference_type_other",
        }
    compatible = requested in _FORM_REFERENCE_SUBTYPES[coarse]
    return {
        "feedback_form": coarse,
        "requested_reference_type": requested,
        "effective_reference_type": requested if compatible else default,
        "reference_conflict": bool(requested and not compatible),
        "reference_rejected": False,
        "reference_rejection_reason": None,
    }


def ground_feedback(
    feedback: dict,
    *,
    feedback_type: str | None = None,
    reference_type: str | None = None,
    action_feature_library: dict[str, dict[str, float]] | None = None,
    grounding_model_path: str | Path = DEFAULT_GROUNDING_MODEL_PATH,
    live_feature_mask: set[str] | frozenset[str] | None = None,
    enforce_feedback_form: bool = False,
) -> dict:
    """Return target features and grounding metadata for one feedback example."""

    text = feedback.get("text") or feedback.get("feedback_text")
    action_feature_library = action_feature_library or {}
    feature_mask = set(live_feature_mask) if live_feature_mask is not None else None

    reference_constraint = (
        constrain_reference_type(feedback_type, reference_type)
        if enforce_feedback_form
        else {
            "feedback_form": str(feedback_type or "").lower() or None,
            "requested_reference_type": reference_type,
            "effective_reference_type": reference_type,
            "reference_conflict": False,
            "reference_rejected": False,
            "reference_rejection_reason": None,
        }
    )
    reference_type = reference_constraint["effective_reference_type"]

    def finalize(result: dict) -> dict:
        masked = _apply_live_feature_mask(result, feature_mask)
        return {**masked, **reference_constraint}

    explicit_target_features = feedback.get("target_features")
    if isinstance(explicit_target_features, dict) and explicit_target_features:
        return finalize({
            "target_features": resolve_opposite_features({
                str(feature): float(value)
                for feature, value in explicit_target_features.items()
            }),
            "grounding_source": "target_features",
            "grounding_confidence": 1.0,
            "temporal_credit_mode": "explicit_reference",
        })

    if reference_type == "other" or reference_constraint["reference_rejected"]:
        return finalize({
            "target_features": {},
            "grounding_source": "other",
            "grounding_confidence": 1.0,
            "abstained": True,
            "temporal_credit_mode": "reject_other",
            "abstention_reason": reference_constraint["reference_rejection_reason"],
        })

    if reference_type == "trajectory" or (
        reference_type is None and feedback_type == "evaluative"
    ):
        target_features = (
            feedback.get("trajectory_features")
            or feedback.get("event_features")
            or {}
        )
        events = _valid_recent_events(feedback)
        result = {
            "target_features": resolve_opposite_features(
                {str(k): float(v) for k, v in target_features.items()}
            ),
            "grounding_source": "trajectory_features",
            "grounding_confidence": 0.95 if target_features else 0.0,
            "abstained": not bool(target_features),
            "temporal_credit_mode": "trajectory_window",
            "recent_event_count": len(events),
            "target_event_span": (
                [events[0]["total_step"], events[-1]["total_step"]]
                if events
                else None
            ),
        }
        return finalize(result)

    if reference_type == "action_behavioral":
        repeated = _repeated_event_grounding(feedback)
        if repeated:
            result = {
                **repeated,
                "grounding_source": "repeated_event_features",
                "grounding_confidence": 0.85 if repeated["text_narrowed"] else 0.7,
                "abstained": False,
                "temporal_credit_mode": "behavior_repetition",
            }
        else:
            legacy = feedback.get("trajectory_features") or {}
            result = {
                "target_features": {
                    str(feature): float(value) for feature, value in legacy.items()
                },
                "grounding_source": "trajectory_features",
                "grounding_confidence": 0.6 if legacy else 0.0,
                "abstained": not bool(legacy),
                "temporal_credit_mode": "behavior_window_legacy",
                "recent_event_count": len(_valid_recent_events(feedback)),
            }
        return finalize(result)

    target_action = feedback.get("target_action")
    if (
        reference_type == "action_spatial"
        or (reference_type is None and feedback_type == "imperative")
    ) and target_action in action_feature_library:
        return finalize({
            "target_features": resolve_opposite_features(action_feature_library[target_action]),
            "grounding_source": f"target_action:{target_action}",
            "grounding_confidence": 0.9,
            "temporal_credit_mode": "commanded_action",
        })

    if enforce_feedback_form and reference_type == "action_spatial":
        return finalize({
            "target_features": {},
            "grounding_source": "unresolved_target_action",
            "grounding_confidence": 0.0,
            "abstained": True,
            "abstention_reason": "imperative_target_action_unresolved",
            "temporal_credit_mode": "commanded_action",
        })

    if reference_type == "action_spatial":
        recent = _recent_event_grounding(feedback)
        if recent:
            return finalize(
                {
                    **recent,
                    "grounding_source": "recent_event_features",
                    "grounding_confidence": 0.9,
                    "abstained": False,
                    "temporal_credit_mode": "single_recent_event",
                },
            )

    feasible_features = feature_mask or ({
        str(feature)
        for vector in action_feature_library.values()
        for feature, value in vector.items()
        if float(value) != 0
    } or None)
    learned = predict_grounded_features(
        str(text or ""),
        model_path=grounding_model_path,
        feasible_features=feasible_features,
    )
    if not learned["abstained"]:
        return finalize({**learned, "temporal_credit_mode": "feature_semantic"})
    target_features = features_from_keywords(text)
    if feasible_features is not None:
        target_features = {
            feature: value
            for feature, value in target_features.items()
            if feature in feasible_features
        }
    return finalize({
        "target_features": target_features,
        "grounding_source": "keyword_features",
        "grounding_confidence": 0.65 if target_features else 0.0,
        "abstained": not bool(target_features),
        "model_abstained": True,
        "temporal_credit_mode": "feature_semantic",
    })
