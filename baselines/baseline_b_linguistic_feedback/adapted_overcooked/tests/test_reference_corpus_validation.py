from __future__ import annotations

import sys
import unittest
from types import SimpleNamespace
from unittest import mock
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.generate_synthetic_feedback import (  # noqa: E402
    GENERATOR_VERSION,
    _generation_messages,
)
from scripts.validate_feedback_corpus import (  # noqa: E402
    _cross_label_exact_conflicts,
    _reference_semantic_check,
    evaluate_example,
    validate_corpus,
)


class ReferencePromptTests(unittest.TestCase):
    def test_v8_prompt_contains_game_and_contrastive_reference_ontology(self) -> None:
        intent = SimpleNamespace(
            polarity=1.0,
            subgoal="GET_ONION",
            feedback_type="evaluative",
            reference_type="trajectory",
            referenced_features=["ingredient_onion"],
        )
        messages = _generation_messages(
            "The pot is empty.", intent, n=3, prompt_variant="contrastive"
        )
        prompt = "\n".join(message["content"] for message in messages)

        self.assertEqual(GENERATOR_VERSION, "deepseek-observer-v8")
        self.assertIn("Supported objects", prompt)
        self.assertIn("Do not invent chopping", prompt)
        self.assertIn("five reference labels are mutually exclusive", prompt)
        self.assertIn("Contrastive examples", prompt)
        for label in (
            "trajectory",
            "feature",
            "action_spatial",
            "action_behavioral",
            "other",
        ):
            self.assertIn(f"- {label}:", prompt)


class IndependentReferenceValidationTests(unittest.TestCase):
    def test_unambiguous_reference_cues_pass(self) -> None:
        cases = {
            "trajectory": "Overall, that whole onion sequence was the right call.",
            "feature": "Duplicating my onion task wastes effort.",
            "action_spatial": "Put the onion in the pot right now.",
            "action_behavioral": "You keep taking the onion every time.",
            "other": "How many onions does this soup need?",
        }
        for label, text in cases.items():
            with self.subTest(label=label):
                verdict = _reference_semantic_check(
                    {"text": text, "reference_type": label}
                )
                self.assertTrue(verdict["reference_semantic_ok"], verdict)

    def test_mixed_scope_cues_are_ambiguous(self) -> None:
        verdict = _reference_semantic_check(
            {
                "text": "You keep grabbing the onion right there.",
                "reference_type": "action_behavioral",
            }
        )
        self.assertFalse(verdict["reference_semantic_ok"])
        self.assertTrue(verdict["reference_ambiguous"])
        self.assertEqual(
            set(verdict["reference_candidates"]),
            {"action_spatial", "action_behavioral"},
        )

    def test_classifier_agreement_does_not_override_missing_cue(self) -> None:
        example = {
            "text": "Grab the onion.",
            "reference_type": "action_spatial",
            "expected_feedback_type": "imperative",
            "referenced_subgoal": "GET_ONION",
            "attributed_sentiment_score": 1.0,
        }
        with (
            mock.patch(
                "scripts.validate_feedback_corpus.predict_reference_type",
                return_value={
                    "reference_type": "action_spatial",
                    "confidence": 0.99,
                },
            ),
            mock.patch(
                "scripts.validate_feedback_corpus.classify_feedback",
                return_value="imperative",
            ),
            mock.patch(
                "scripts.validate_feedback_corpus._surface_sentiment",
                return_value=0.5,
            ),
            mock.patch(
                "scripts.validate_feedback_corpus._structure_errors",
                return_value=[],
            ),
            mock.patch(
                "scripts.validate_feedback_corpus._feature_errors",
                return_value=[],
            ),
        ):
            verdict = evaluate_example(example, weights={})

        self.assertTrue(verdict["reference_ok"])
        self.assertFalse(verdict["reference_semantic_ok"])
        self.assertFalse(verdict["passed"])
        self.assertIn("missing_reference_cue", verdict["reasons"])

    def test_cross_label_exact_conflicts_are_dropped_and_reported(self) -> None:
        rows = [
            {
                "feedback_id": "spatial-1",
                "text": "Put the onion in the pot right now!",
                "reference_type": "action_spatial",
                "source": "llm",
                "split": "train",
                "attributed_sentiment_score": 1.0,
            },
            {
                "feedback_id": "feature-1",
                "text": "put the onion in the pot right now.",
                "reference_type": "feature",
                "source": "llm",
                "split": "train",
                "attributed_sentiment_score": 1.0,
            },
        ]
        conflicts = _cross_label_exact_conflicts(rows)
        self.assertEqual(len(conflicts), 1)
        self.assertEqual(set(conflicts[0]["labels"]), {"action_spatial", "feature"})

        def accepted_verdict(example: dict, *, weights: dict) -> dict:
            return {
                "type_ok": True,
                "sentiment_ok": True,
                "reference_ok": True,
                "expected_reference_type": example["reference_type"],
                "reference_cue_ok": True,
                "reference_ambiguous": False,
                "reference_semantic_ok": True,
                "passed": True,
                "reasons": [],
            }

        with mock.patch(
            "scripts.validate_feedback_corpus.evaluate_example",
            side_effect=accepted_verdict,
        ):
            kept, report = validate_corpus(rows, near_duplicate_threshold=1.0)

        self.assertEqual(kept, [])
        self.assertEqual(
            report["drop_reasons"]["cross_label_exact_conflict"], 2
        )
        reference_report = report["reference_validation"]
        self.assertEqual(reference_report["cross_label_exact_conflict_groups"], 1)
        self.assertEqual(reference_report["cross_label_exact_conflict_rows"], 2)
        self.assertFalse(
            report["quality_checks"]["max_cross_label_exact_conflict_groups"]
        )


if __name__ == "__main__":
    unittest.main()
