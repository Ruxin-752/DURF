from __future__ import annotations

from collections import Counter
import sys
import unittest
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.train_direct_fg_classifier_candidate import (  # noqa: E402
    DIRECT_BOUNDARY_LABEL_SOURCE,
    HUMAN_LABEL_SOURCE,
    _convert_boundary_rows,
    _explicit_target,
    build_direct_fg_corpus,
    train_direct_fg_candidate,
)
from src.feedback_form_classifier import FEEDBACK_TYPES  # noqa: E402


class DirectFgClassifierCandidateTests(unittest.TestCase):
    def test_explicit_target_ignores_five_class_reference_type(self) -> None:
        target, field = _explicit_target(
            {
                "expected_feedback_type": "imperative",
                "classification_label": "Imperative",
                "label_source": DIRECT_BOUNDARY_LABEL_SOURCE,
                "reference_type": "trajectory",
            },
            source="fixture",
            index=1,
        )
        self.assertEqual((target, field), ("imperative", "expected_feedback_type"))

    def test_conflicting_explicit_labels_fail_closed(self) -> None:
        with self.assertRaisesRegex(ValueError, "disagree"):
            _explicit_target(
                {
                    "expected_feedback_type": "imperative",
                    "classification_label": "Evaluative",
                    "label_source": DIRECT_BOUNDARY_LABEL_SOURCE,
                },
                source="fixture",
                index=1,
            )

    def test_reference_collapsed_boundary_rows_are_excluded(self) -> None:
        rows, audit = _convert_boundary_rows(
            ROOT / "data" / "feedback_form_boundary_train.v1.json"
        )
        self.assertTrue(rows)
        self.assertEqual(
            {row["label_source"] for row in rows},
            {DIRECT_BOUNDARY_LABEL_SOURCE},
        )
        self.assertEqual(
            audit["excluded_label_source_counts"],
            {"paper_mapping_contrastive_train_only": 144},
        )

    def test_corpus_is_balanced_family_and_group_disjoint(self) -> None:
        first, first_report = build_direct_fg_corpus()
        second, second_report = build_direct_fg_corpus()
        self.assertEqual(first, second)
        self.assertEqual(first_report, second_report)
        self.assertEqual(len(first), 548)
        self.assertEqual(first_report["split_rows"], {"dev": 126, "train": 422})
        self.assertEqual(
            first_report["label_source_counts"],
            {DIRECT_BOUNDARY_LABEL_SOURCE: 504, HUMAN_LABEL_SOURCE: 44},
        )
        self.assertTrue(
            all(value == 0 for value in first_report["split_integrity"].values())
        )
        for split in ("train", "dev"):
            labels = Counter(
                row["label"] for row in first if row["split"] == split
            )
            self.assertEqual(set(labels), set(FEEDBACK_TYPES))
        train_families = {
            row["split_family"] for row in first if row["split"] == "train"
        }
        dev_families = {
            row["split_family"] for row in first if row["split"] == "dev"
        }
        train_groups = {row["group_id"] for row in first if row["split"] == "train"}
        dev_groups = {row["group_id"] for row in first if row["split"] == "dev"}
        self.assertFalse(train_families & dev_families)
        self.assertFalse(train_groups & dev_groups)
        self.assertFalse(first_report["reference_type_policy"]["used_as_target"])

    def test_missing_admissible_label_fails_closed(self) -> None:
        with self.assertRaisesRegex(ValueError, "no admissible"):
            _explicit_target(
                {
                    "reference_type": "trajectory",
                    "label_source": "paper_reference_mapping",
                },
                source="fixture",
                index=1,
            )

    def test_candidate_clears_direct_development_gate_without_test_data(self) -> None:
        rows, _audit = build_direct_fg_corpus()
        artifact, report = train_direct_fg_candidate(rows)
        metrics = report["direct_fg_family_group_disjoint_dev"]["classification"]
        self.assertGreaterEqual(metrics["accuracy"], 0.87)
        self.assertGreaterEqual(metrics["macro_f1"], 0.87)
        self.assertTrue(artifact["candidate_only"])
        self.assertFalse(artifact["reference_type_used_as_target"])
        self.assertEqual(report["selection_protocol"]["frozen_or_test_rows_used"], 0)
        self.assertFalse(report["frozen_test"]["path_opened"])

    def test_fixture_with_only_mapping_rows_fails_closed(self) -> None:
        fixture_rows = [
            {
                "text": "Good move.",
                "expected_feedback_type": "evaluative",
                "classification_label": "Evaluative",
                "split": "train",
                "label_source": "paper_mapping_contrastive_train_only",
                "group_id": "g1",
                "template_family": "x:evaluative:f1",
            }
        ]
        with patch(
            "scripts.train_direct_fg_classifier_candidate._read_json_list",
            return_value=fixture_rows,
        ):
            with self.assertRaisesRegex(ValueError, "no direct-speech-act"):
                _convert_boundary_rows(Path("mapping_only.json"))


if __name__ == "__main__":
    unittest.main()
