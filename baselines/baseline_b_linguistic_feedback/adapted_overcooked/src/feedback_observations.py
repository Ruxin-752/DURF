"""Shared text -> grounded observation processing for Route 1 and Route 2.

The paper's feedback-form distinction changes what an utterance refers to:

* evaluative feedback refers to the recently observed trajectory/subgoal;
* imperative feedback refers to a desired action/subgoal;
* descriptive feedback is grounded from the behavior named in the text.

Both the offline Route 1 pipeline and the live subgoal agent use this module so
the classification/grounding stage cannot silently disappear at runtime.
"""

from __future__ import annotations

from .overcooked_grounding import features_from_keywords, ground_feedback
from .phrase_reference_classifier import classify_utterance
from .sentiment_extractor import (
    desired_action_sentiment,
    extract_sentiment,
    modified_vader_observation,
)
from .text_analysis import limited_punc_tokenization


PRIVILEGED_FEEDBACK_FIELDS = frozenset(
    {
        "expected_feedback_type",
        "target_features",
        "attributed_sentiment_score",
    }
)

PROHIBITIVE_MARKERS = (
    "don't ",
    "do not ",
    "stop ",
    "never ",
    "shouldn't ",
    "should not ",
    "quit ",
    "avoid ",
)
HARD_NEGATIVE_MARKERS = (
    "not good",
    "not helpful",
    "not right",
    "not what",
    "anything but",
    "hardly helpful",
    "barely helped",
    "failed to",
    "worse than",
    "no thanks",
)


def valence_with_safety_gate(text: str, score: float) -> tuple[float, float, str]:
    """Block the paper's positive neutral default on synthetic hard negatives."""

    lowered = text.lower()
    if any(marker in lowered for marker in PROHIBITIVE_MARKERS):
        return -1.0, 1.0, "prohibitive"
    if any(marker in lowered for marker in HARD_NEGATIVE_MARKERS):
        return min(float(score), -0.5), 0.9, "hard_negative"
    # Lexically neutral phrases retain modified VADER's paper baseline, but
    # their valence is less trustworthy than an explicit polar expression.
    confidence = 0.65 if float(score) == 0.5 else 0.9
    return float(score), confidence, "modified_vader"


def without_privileged_labels(feedback: dict) -> dict:
    """Copy feedback without gold type/grounding/valence annotations."""

    return {
        key: value
        for key, value in feedback.items()
        if key not in PRIVILEGED_FEEDBACK_FIELDS
    }


def effective_sentiment_score(
    *,
    feedback_type: str,
    sentiment: dict,
    target_features: dict[str, float],
    feedback: dict,
) -> float:
    """Resolve valence, preferring explicit labels only in oracle mode."""

    if feedback.get("attributed_sentiment_score") is not None:
        return float(feedback["attributed_sentiment_score"])
    if feedback_type == "imperative":
        text = str(feedback.get("text") or feedback.get("feedback_text") or "").lower()
        if any(marker in text for marker in PROHIBITIVE_MARKERS):
            return -1.0
        if feedback.get("target_action"):
            return desired_action_sentiment(feedback_type, target_features)
    score, _confidence, _source = valence_with_safety_gate(
        str(feedback.get("text") or feedback.get("feedback_text") or ""),
        float(sentiment["sentiment_score"]),
    )
    return score


def build_feedback_observations(
    feedback: dict,
    *,
    feedback_type: str,
    action_feature_library: dict[str, dict[str, float]],
    prefer_explicit_reference: bool = True,
) -> list[dict]:
    """Decompose one utterance into grounded, valenced Route 1 observations."""

    text = feedback.get("text") or feedback.get("feedback_text") or ""
    if prefer_explicit_reference:
        # Oracle/offline compatibility path: annotations and the known speech
        # act select grounding exactly as in the previous paper pipeline.
        grounding = ground_feedback(
            feedback,
            feedback_type=feedback_type,
            action_feature_library=action_feature_library,
        )
        if grounding["grounding_source"] != "keyword_features":
            explicit_valence = feedback.get("attributed_sentiment_score")
            sentiment_is_implied = feedback_type == "imperative" and bool(
                feedback.get("target_action")
            )
            sentiment = (
                {"sentiment_score": float(explicit_valence or 0.0)}
                if explicit_valence is not None or sentiment_is_implied
                else extract_sentiment(text)
            )
            valence = effective_sentiment_score(
                feedback_type=feedback_type,
                sentiment=sentiment,
                target_features=grounding["target_features"],
                feedback=feedback,
            )
            reference_type = (
                feedback.get("reference_type")
                or (
                    "trajectory"
                    if grounding["grounding_source"] == "trajectory_features"
                    else "action_spatial"
                    if grounding["grounding_source"].startswith("target_action:")
                    else "feature"
                )
            )
            return [
                {
                    "target_features": grounding["target_features"],
                    "valence": valence,
                    "phrase": text,
                    "reference_type": reference_type,
                    "reference_confidence": 1.0,
                    "grounding_source": grounding["grounding_source"],
                    "grounding_confidence": grounding.get("grounding_confidence", 1.0),
                    "grounding_abstained": grounding.get("abstained", False),
                    "reference_abstained": False,
                    "valence_confidence": 1.0 if explicit_valence is not None else 0.9,
                }
            ]

        sub_observations: list[dict] = []
        for phrase in limited_punc_tokenization(text):
            phrase_features = features_from_keywords(phrase)
            if phrase_features:
                phrase_feedback = dict(feedback)
                phrase_feedback["text"] = phrase
                sentiment = {"sentiment_score": modified_vader_observation(phrase)}
                safe_valence, valence_confidence, valence_source = valence_with_safety_gate(
                    phrase, sentiment["sentiment_score"]
                )
                sub_observations.append(
                    {
                        "target_features": phrase_features,
                        "valence": effective_sentiment_score(
                            feedback_type=feedback_type,
                            sentiment={"sentiment_score": safe_valence},
                            target_features=phrase_features,
                            feedback=phrase_feedback,
                        ),
                        "phrase": phrase,
                        "reference_type": "feature",
                        "reference_confidence": 1.0,
                        "grounding_source": "keyword_features",
                        "grounding_confidence": 0.7,
                        "grounding_abstained": False,
                        "reference_abstained": False,
                        "valence_confidence": valence_confidence,
                        "valence_source": valence_source,
                    }
                )
        if not sub_observations:
            fallback_features = features_from_keywords(text)
            sentiment = {"sentiment_score": modified_vader_observation(text)}
            safe_valence, valence_confidence, valence_source = valence_with_safety_gate(
                text, sentiment["sentiment_score"]
            )
            sub_observations.append(
                {
                    "target_features": fallback_features,
                    "valence": effective_sentiment_score(
                        feedback_type=feedback_type,
                        sentiment={"sentiment_score": safe_valence},
                        target_features=fallback_features,
                        feedback=feedback,
                    ),
                    "phrase": text,
                    "reference_type": "feature",
                    "reference_confidence": 1.0,
                    "grounding_source": "keyword_features",
                    "grounding_confidence": 0.7 if fallback_features else 0.0,
                    "grounding_abstained": not bool(fallback_features),
                    "reference_abstained": False,
                    "valence_confidence": valence_confidence,
                    "valence_source": valence_source,
                }
            )
        return sub_observations

    predictions = classify_utterance(text)
    sub_observations: list[dict] = []
    for prediction in predictions:
        phrase = prediction["phrase"]
        phrase_feedback = dict(feedback)
        phrase_feedback["text"] = phrase
        grounding = ground_feedback(
            phrase_feedback,
            feedback_type=feedback_type,
            reference_type=prediction["reference_type"],
            action_feature_library=action_feature_library,
        )
        sentiment = {"sentiment_score": modified_vader_observation(phrase)}
        safe_valence, valence_confidence, valence_source = valence_with_safety_gate(
            phrase, sentiment["sentiment_score"]
        )
        sub_observations.append(
            {
                "target_features": grounding["target_features"],
                "valence": effective_sentiment_score(
                    feedback_type=feedback_type,
                    sentiment={"sentiment_score": safe_valence},
                    target_features=grounding["target_features"],
                    feedback=phrase_feedback,
                ),
                "phrase": phrase,
                "reference_type": prediction["reference_type"],
                "reference_confidence": prediction["confidence"],
                "reference_probabilities": prediction["probabilities"],
                "reference_classifier": prediction.get("classifier"),
                "reference_abstained": prediction.get("abstained", False),
                "reference_top2_margin": prediction.get("top2_margin"),
                "grounding_source": grounding["grounding_source"],
                "grounding_confidence": grounding.get("grounding_confidence", 1.0),
                "grounding_abstained": grounding.get("abstained", False),
                "valence_confidence": valence_confidence,
                "valence_source": valence_source,
            }
        )
    return sub_observations
