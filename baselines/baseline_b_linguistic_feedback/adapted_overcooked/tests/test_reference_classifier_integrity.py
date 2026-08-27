from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.train_phrase_reference_classifier import (  # noqa: E402
    active_model_paper_strict_benchmark,
    reproduce_original_paper_benchmark,
)


def _strict_row(split: str, index: int, text: str, label: str) -> dict:
    return {
        "feedback_id": f"strict-{split}-{index}",
        "paper_task_uuid": f"task-{split}-{index}",
        "group_id": f"paper_task_uuid:task-{split}-{index}",
        "split": split,
        "text": text,
        "reference_type": label,
    }


class ReferenceClassifierIntegrityTests(unittest.TestCase):
    def test_paper_protocol_reproduction_discloses_non_external_overlap(self) -> None:
        report = reproduce_original_paper_benchmark()
        self.assertEqual(
            report["benchmark_role"], "paper_protocol_reproduction_not_external"
        )
        self.assertGreater(report["normalized_text_overlap_test_rows"], 0)
        self.assertGreater(report["task_uuid_overlap_count"], 0)

    def test_strict_external_benchmark_fails_on_actual_text_overlap(self) -> None:
        strict_rows = [
            _strict_row("train", 0, "paper training phrase", "trajectory"),
            _strict_row("dev", 0, "paper development phrase", "feature"),
            _strict_row("test", 0, "sealed external phrase", "trajectory"),
        ]
        active_rows = [
            {
                "feedback_id": "different-id",
                "text": "Sealed external phrase!",
                "split": "train",
            }
        ]
        with self.assertRaisesRegex(ValueError, "texts=1"):
            active_model_paper_strict_benchmark(
                active_rows,
                None,
                None,
                temperature=1.0,
                class_thresholds={},
                strict_rows=strict_rows,
            )

    def test_strict_external_benchmark_fails_on_task_group_overlap(self) -> None:
        strict_rows = [
            _strict_row("train", 0, "paper training phrase", "trajectory"),
            _strict_row("dev", 0, "paper development phrase", "feature"),
            _strict_row("test", 0, "sealed external phrase", "trajectory"),
        ]
        active_rows = [
            {
                "feedback_id": "different-id",
                "text": "different phrase from the sealed task",
                "group_id": "paper_task_uuid:task-test-0",
                "split": "train",
            }
        ]
        with self.assertRaisesRegex(ValueError, "tasks=1"):
            active_model_paper_strict_benchmark(
                active_rows,
                None,
                None,
                temperature=1.0,
                class_thresholds={},
                strict_rows=strict_rows,
            )

    def test_strict_external_benchmark_reports_computed_zero_overlap(self) -> None:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression

        strict_rows = [
            _strict_row("train", 0, "paper training praise", "trajectory"),
            _strict_row("dev", 0, "paper development object", "feature"),
            _strict_row("test", 0, "sealed praise phrase", "trajectory"),
            _strict_row("test", 1, "sealed object phrase", "feature"),
        ]
        train_text = ["whole sequence praise", "object property detail"]
        train_labels = ["trajectory", "feature"]
        vectorizer = TfidfVectorizer().fit(train_text)
        classifier = LogisticRegression(random_state=1).fit(
            vectorizer.transform(train_text), train_labels
        )
        report = active_model_paper_strict_benchmark(
            [{"feedback_id": "active", "text": "unrelated active phrase"}],
            vectorizer,
            classifier,
            temperature=1.0,
            class_thresholds={},
            strict_rows=strict_rows,
        )
        self.assertEqual(report["active_id_overlap_count"], 0)
        self.assertEqual(report["active_normalized_text_overlap_count"], 0)
        self.assertEqual(report["active_task_uuid_overlap_count"], 0)
        self.assertEqual(report["test_rows"], 2)
        self.assertEqual(
            report["benchmark_role"],
            "paper_strict_exposed_regression_not_for_selection",
        )


if __name__ == "__main__":
    unittest.main()
