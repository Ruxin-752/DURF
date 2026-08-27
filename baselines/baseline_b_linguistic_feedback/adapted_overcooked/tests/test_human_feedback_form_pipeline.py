from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TEST_TMP = ROOT / "outputs" / "_feedback_form_test_tmp"
sys.path.insert(0, str(ROOT))

from scripts.augment_human_feedback_form_train import (  # noqa: E402
    LABEL_SOURCE as AUGMENTATION_LABEL_SOURCE,
    SOURCE as AUGMENTATION_SOURCE,
    augment_training_rows,
)
from scripts.evaluate_human_feedback_form_test import evaluate  # noqa: E402
from scripts.prepare_human_feedback_form_annotations import (  # noqa: E402
    CANONICAL_LABELS,
    prepare_annotations,
    stable_text_split,
    template_family_signature,
)
from scripts.train_feedback_form_classifier import (  # noqa: E402
    load_training_rows,
    train_feedback_form_classifier,
)
from src.feedback_form_classifier import (  # noqa: E402
    FEEDBACK_TYPES,
    classify_feedback,
    predict_feedback_form,
)


def _language_for(split: str) -> str:
    for index in range(100000):
        language = f"Stable human feedback phrase number {index}."
        if stable_text_split(language.lower().rstrip(".")) == split:
            return language
    raise AssertionError(f"unable to find deterministic fixture for {split}")


def _model_row(text: str, label: str, split: str, source: str = "synthetic") -> dict:
    return {
        "text": text,
        "normalized_text": text.lower(),
        "label": label,
        "split": split,
        "source": source,
        "group_id": f"{source}:{split}:{text}",
    }


def _tiny_rows() -> tuple[list[dict], list[dict]]:
    train_phrases = {
        "evaluative": [
            "That was a good move",
            "Your last action was bad",
            "Nice work on that step",
            "That choice was wrong",
        ],
        "imperative": [
            "Please fetch an onion",
            "Move away from the pot",
            "Do not take that dish",
            "Serve the soup now",
        ],
        "descriptive": [
            "You keep blocking the path",
            "The counter is crowded",
            "You are standing by the pot",
            "Your repeated trips delay serving",
        ],
    }
    dev_phrases = {
        "evaluative": ["That previous step was great"],
        "imperative": ["Please grab a tomato"],
        "descriptive": ["You are blocking my route"],
    }
    train = [
        _model_row(text, label, "train")
        for label, texts in train_phrases.items()
        for text in texts
    ]
    dev = [
        _model_row(text, label, "dev")
        for label, texts in dev_phrases.items()
        for text in texts
    ]
    return train, dev


def _prepared_human_row(
    feedback_id: str,
    text: str,
    classification_label: str,
) -> dict:
    normalized = text.lower().rstrip(".")
    return {
        "feedback_id": feedback_id,
        "text": text,
        "normalized_text": normalized,
        "classification_label": classification_label,
        "expected_feedback_type": classification_label.lower(),
        "template_family": template_family_signature(normalized),
        "split": "train",
        "source": "human",
        "label_source": "human_explicit",
    }


class HumanFeedbackFormPreparationTests(unittest.TestCase):
    def test_only_exact_public_schema_and_canonical_labels_are_accepted(self) -> None:
        rows = [
            {"language": f"Unique {label} example", "classification_label": label}
            for label in CANONICAL_LABELS
        ]
        splits, report = prepare_annotations(rows)
        released = [row for values in splits.values() for row in values]
        self.assertEqual(len(released), 3)
        self.assertEqual(
            {row["expected_feedback_type"] for row in released}, set(FEEDBACK_TYPES)
        )
        self.assertFalse(report["model_predictions_used_as_labels"])
        self.assertFalse(
            report["split_policy"]["teacher_session_identity_available"]
        )

        with self.assertRaisesRegex(ValueError, "classification_label"):
            prepare_annotations(
                [{"language": "lowercase is invalid", "classification_label": "evaluative"}]
            )
        with self.assertRaisesRegex(ValueError, "schema"):
            prepare_annotations(
                [
                    {
                        "language": "extra metadata is invalid",
                        "classification_label": "Evaluative",
                        "teacher_id": "not_allowed",
                    }
                ]
            )

    def test_stable_text_split_and_exact_text_conflict_guard(self) -> None:
        rows = [
            {"language": _language_for(split), "classification_label": "Descriptive"}
            for split in ("train", "dev", "test")
        ]
        first, _ = prepare_annotations(rows)
        second, _ = prepare_annotations(list(reversed(rows)))
        self.assertEqual(
            {
                row["feedback_id"]: row["split"]
                for values in first.values()
                for row in values
            },
            {
                row["feedback_id"]: row["split"]
                for values in second.values()
                for row in values
            },
        )
        self.assertTrue(all(len(first[split]) == 1 for split in first))
        with self.assertRaisesRegex(ValueError, "conflicting explicit labels"):
            prepare_annotations(
                [
                    {"language": "Same normalized phrase!", "classification_label": "Evaluative"},
                    {"language": "same normalized phrase", "classification_label": "Imperative"},
                ]
            )

    def test_near_clone_templates_are_kept_in_one_partition(self) -> None:
        onion = "Please fetch an onion next."
        tomato = "Please fetch a tomato next."
        good = "That move was good."
        really_bad = "That move was really bad."
        self.assertEqual(
            template_family_signature(onion), template_family_signature(tomato)
        )
        self.assertEqual(stable_text_split(onion), stable_text_split(tomato))
        self.assertEqual(
            template_family_signature(good), template_family_signature(really_bad)
        )
        self.assertEqual(stable_text_split(good), stable_text_split(really_bad))

        splits, report = prepare_annotations(
            [
                {"language": onion, "classification_label": "Imperative"},
                {"language": tomato, "classification_label": "Imperative"},
                {"language": good, "classification_label": "Evaluative"},
                {"language": really_bad, "classification_label": "Evaluative"},
            ]
        )
        released = [row for rows in splits.values() for row in rows]
        self.assertEqual(len(released), 4)
        self.assertTrue(
            all(
                value == 0
                for value in report["split_policy"]["template_family_overlap"].values()
            )
        )
        self.assertFalse(
            report["split_policy"]["template_family_uses_labels_or_model_predictions"]
        )


class HumanFeedbackFormAugmentationTests(unittest.TestCase):
    def test_train_only_variants_preserve_parent_labels_and_raw_rows(self) -> None:
        parents = [
            _prepared_human_row(
                "parent_eval", "That move was really good.", "Evaluative"
            ),
            _prepared_human_row(
                "parent_imp", "Please fetch a tomato next.", "Imperative"
            ),
            _prepared_human_row(
                "parent_desc", "These tomatos are unneccessary.", "Descriptive"
            ),
        ]
        untouched = json.loads(json.dumps(parents))
        augmented, report = augment_training_rows(
            parents, [], [], max_variants_per_parent=3
        )

        self.assertEqual(parents, untouched)
        self.assertTrue(augmented)
        self.assertLessEqual(len(augmented), len(parents) * 3)
        parent_by_id = {row["feedback_id"]: row for row in parents}
        required = {
            "parent_feedback_id",
            "augmentation_rule",
            "split",
            "source",
            "label_source",
        }
        for row in augmented:
            self.assertTrue(required <= set(row))
            self.assertEqual(row["split"], "train")
            self.assertEqual(row["source"], AUGMENTATION_SOURCE)
            self.assertEqual(row["label_source"], AUGMENTATION_LABEL_SOURCE)
            parent = parent_by_id[row["parent_feedback_id"]]
            self.assertEqual(
                row["classification_label"], parent["classification_label"]
            )
            self.assertEqual(
                row["expected_feedback_type"], parent["expected_feedback_type"]
            )
        self.assertEqual(
            len({row["normalized_text"] for row in augmented}), len(augmented)
        )
        emitted_text = {row["text"] for row in augmented}
        self.assertIn("Please fetch an onion next.", emitted_text)
        self.assertIn("These tomatoes are unnecessary.", emitted_text)
        self.assertFalse(report["raw_human_rows_modified"])
        self.assertFalse(report["holdout_guard"]["performance_metrics_computed"])

    def test_exact_template_and_near_holdout_leakage_are_rejected(self) -> None:
        parents = [
            _prepared_human_row(
                "parent_exact", "Please fetch a tomato next.", "Imperative"
            ),
            _prepared_human_row(
                "parent_template", "Please grab another tomato.", "Imperative"
            ),
            _prepared_human_row(
                "parent_near", "That move was good.", "Evaluative"
            ),
        ]
        # These rows are intentionally minimal.  Their labels and split sizes
        # are neither inspected nor included in the augmentation report.
        development_guard = [
            {"text": "Please fetch an onion next."},
            {"text": "Please grab another plate."},
            {"text": "That last move was good today."},
        ]
        augmented, report = augment_training_rows(
            parents,
            development_guard,
            [],
            near_duplicate_threshold=0.80,
        )

        rejected = report["rejections"]
        self.assertGreaterEqual(rejected["holdout_exact"], 1)
        self.assertGreaterEqual(rejected["holdout_template_family"], 1)
        self.assertGreaterEqual(rejected["holdout_near_duplicate"], 1)
        emitted = {row["normalized_text"] for row in augmented}
        self.assertNotIn("please fetch an onion next", emitted)
        serialized_report = json.dumps(report)
        for guard in development_guard:
            self.assertNotIn(guard["text"], serialized_report)
        self.assertFalse(report["holdout_guard"]["holdout_content_in_report"])
        self.assertFalse(
            report["holdout_guard"]["holdout_row_or_label_counts_in_report"]
        )

    def test_cap_and_output_are_deterministic(self) -> None:
        parents = [
            _prepared_human_row(
                "parent_many", "Please fetch a tomato next.", "Imperative"
            )
        ]
        first, first_report = augment_training_rows(
            parents, [], [], max_variants_per_parent=1
        )
        second, second_report = augment_training_rows(
            list(reversed(parents)), [], [], max_variants_per_parent=1
        )
        self.assertEqual(first, second)
        self.assertEqual(first_report, second_report)
        self.assertEqual(len(first), 1)
        self.assertGreater(first_report["rejections"]["per_parent_cap"], 0)


class FeedbackFormModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        TEST_TMP.mkdir(parents=True, exist_ok=True)

    @classmethod
    def tearDownClass(cls) -> None:
        for path in TEST_TMP.iterdir():
            if path.is_file():
                path.unlink()
        try:
            TEST_TMP.rmdir()
        except OSError:
            pass

    def test_model_first_prediction_and_read_only_empty_test(self) -> None:
        from joblib import dump

        train, dev = _tiny_rows()
        artifact, report = train_feedback_form_classifier(train, dev, min_df=1)
        artifact["minimum_model_confidence"] = 0.0
        self.assertIsNone(report["test_metrics"])
        self.assertEqual(
            report["selection_protocol"]["test_examples_evaluated_during_selection"], 0
        )
        model_path = TEST_TMP / "model.joblib"
        test_path = TEST_TMP / "human_test.json"
        dump(artifact, model_path)
        test_path.write_text("[]\n", encoding="utf-8")
        prediction = predict_feedback_form(
            "Please fetch an onion", model_path=model_path
        )
        self.assertEqual(prediction["classifier"], "tfidf_logistic_regression")
        self.assertIn(prediction["feedback_type"], FEEDBACK_TYPES)
        self.assertEqual(
            classify_feedback("Please fetch an onion", model_path=model_path),
            prediction["feedback_type"],
        )
        artifact["minimum_model_confidence"] = 1.01
        fallback_model_path = TEST_TMP / "forced_fallback_model.joblib"
        dump(artifact, fallback_model_path)
        uncertain = predict_feedback_form(
            "That phrasing is deliberately uncertain", model_path=fallback_model_path
        )
        self.assertEqual(
            uncertain["classifier"], "tfidf_logistic_regression_low_confidence"
        )
        self.assertTrue(uncertain["abstained"])
        self.assertEqual(
            uncertain["feedback_type"], uncertain["model_feedback_type"]
        )
        self.assertLess(
            uncertain["model_confidence"], uncertain["confidence_threshold"]
        )
        self.assertEqual(
            uncertain["probabilities"], uncertain["model_probabilities"]
        )
        evaluation = evaluate(model_path, test_path)
        self.assertEqual(evaluation["status"], "no_human_test_data")
        self.assertIsNone(evaluation["metrics"])
        self.assertFalse(evaluation["used_for_training_or_hyperparameter_selection"])

    def test_deployed_model_preserves_low_confidence_model_prediction(self) -> None:
        low = predict_feedback_form("Good job.")
        self.assertEqual(low["feedback_type"], "evaluative")
        self.assertIn(
            low["classifier"],
            {
                "tfidf_logistic_regression",
                "tfidf_logistic_regression_low_confidence",
            },
        )
        if low["classifier"] == "tfidf_logistic_regression_low_confidence":
            self.assertLess(low["model_confidence"], low["confidence_threshold"])
            self.assertEqual(low["probabilities"], low["model_probabilities"])
            self.assertTrue(low["abstained"])
        else:
            self.assertGreaterEqual(low["confidence"], low["confidence_threshold"])

        clear = predict_feedback_form("Please fetch an onion.")
        self.assertEqual(clear["feedback_type"], "imperative")
        self.assertIn(
            clear["classifier"],
            {
                "tfidf_logistic_regression",
                "tfidf_logistic_regression_low_confidence",
            },
        )
        if clear["classifier"] == "tfidf_logistic_regression_low_confidence":
            self.assertEqual(clear["model_feedback_type"], "imperative")
            self.assertLess(clear["model_confidence"], clear["confidence_threshold"])
            self.assertTrue(clear["abstained"])
        else:
            self.assertGreaterEqual(clear["confidence"], clear["confidence_threshold"])

    def test_loader_collapses_reference_labels_and_never_uses_human_test(self) -> None:
        reference_for_label = {
            "evaluative": "trajectory",
            "imperative": "action_spatial",
            "descriptive": "feature",
        }
        synthetic = [
            {
                "text": f"{split} {label} example",
                "reference_type": reference_for_label[label],
                # Deliberately contradictory: this generator cross-product
                # field must not become the synthetic target.
                "expected_feedback_type": "descriptive" if label != "descriptive" else "evaluative",
                "split": split,
                "group_id": f"{split}_{label}",
            }
            for split in ("train", "dev", "test")
            for label in FEEDBACK_TYPES
        ]
        synthetic.extend(
            {
                "text": f"{split} excluded {reference_type} example",
                "reference_type": reference_type,
                "expected_feedback_type": "imperative",
                "split": split,
                "group_id": f"{split}_{reference_type}",
            }
            for split in ("train", "dev")
            for reference_type in ("action_behavioral", "other")
        )
        synthetic_path = TEST_TMP / "synthetic.json"
        human_train = TEST_TMP / "human_train.json"
        human_dev = TEST_TMP / "human_dev.json"
        synthetic_path.write_text(json.dumps(synthetic), encoding="utf-8")
        human_train.write_text("[]", encoding="utf-8")
        human_dev.write_text("[]", encoding="utf-8")
        train, dev, audit = load_training_rows(
            synthetic_path, human_train, human_dev
        )
        self.assertEqual(len(train), 3)
        self.assertEqual(len(dev), 3)
        self.assertEqual(audit["synthetic_test_rows_skipped_without_label_use"], 3)
        self.assertEqual(audit["test_examples_evaluated_during_selection"], 0)
        self.assertEqual(
            audit["synthetic_excluded_reference_counts"],
            {
                "train": {"action_behavioral": 1, "other": 1},
                "dev": {"action_behavioral": 1, "other": 1},
            },
        )
        self.assertFalse(audit["synthetic_expected_feedback_type_used_as_target"])
        self.assertEqual({row["label"] for row in train}, set(FEEDBACK_TYPES))
        self.assertFalse(audit["labels_inferred_from_role_or_predictions"])

        synthetic[0].pop("reference_type")
        synthetic[0]["role"] = "trajectory"
        synthetic_path.write_text(json.dumps(synthetic), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "reference_type"):
            load_training_rows(synthetic_path, human_train, human_dev)

    def test_missing_model_uses_rules(self) -> None:
        missing = ROOT / "outputs" / "definitely_missing_feedback_form_model.joblib"
        self.assertEqual(
            classify_feedback("Please move away", model_path=missing), "imperative"
        )


if __name__ == "__main__":
    unittest.main()
