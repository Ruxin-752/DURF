from __future__ import annotations

import hashlib
import unittest
from pathlib import Path
import sys
import uuid


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.generate_feedback_form_review_candidates import (  # noqa: E402
    BLIND_SHUFFLE_VERSION,
    CANONICAL_LABELS,
    MINIMUM_FAMILIES_PER_LABEL,
    REVIEW_STATUS,
    _write_jsonl,
    build_blind_labeling_rows,
    curate_candidate_pool,
)


class FeedbackFormReviewCandidateTests(unittest.TestCase):
    def test_writer_reports_the_actual_on_disk_sha256(self) -> None:
        output = ROOT / "outputs" / f"_candidate_hash_{uuid.uuid4().hex}.jsonl"
        try:
            reported = _write_jsonl(
                output,
                [{"language": "line one\nline two", "classification_label": ""}],
            )
            self.assertEqual(
                reported, hashlib.sha256(output.read_bytes()).hexdigest()
            )
        finally:
            if output.exists():
                output.unlink()

    def test_default_pool_has_diverse_pending_only_schema(self) -> None:
        rows, report = curate_candidate_pool({"empty_fixture": []})

        self.assertTrue(rows)
        self.assertEqual(
            {row["classification_label"] for row in rows},
            set(CANONICAL_LABELS),
        )
        for row in rows:
            self.assertEqual(
                set(row),
                {"language", "classification_label", "review_status"},
            )
            self.assertEqual(row["review_status"], REVIEW_STATUS)
        for label in CANONICAL_LABELS:
            self.assertGreaterEqual(
                report["counts_by_label"][label],
                MINIMUM_FAMILIES_PER_LABEL,
            )
        self.assertEqual(
            report["candidate_count"], report["distinct_surface_family_count"]
        )
        self.assertFalse(report["human_authorship_claimed"])
        self.assertFalse(report["training_eligible"])
        self.assertFalse(report["evaluation_eligible"])
        self.assertFalse(report["model_or_classifier_used_to_assign_labels"])

    def test_blind_template_hides_labels_and_has_stable_scrambled_order(self) -> None:
        rows, _ = curate_candidate_pool({"empty_fixture": []})
        first = build_blind_labeling_rows(rows)
        second = build_blind_labeling_rows(reversed(rows))

        self.assertEqual(first, second)
        self.assertNotEqual(
            [row["language"] for row in first],
            [row["language"] for row in rows],
        )
        self.assertTrue(BLIND_SHUFFLE_VERSION)
        self.assertEqual(
            {tuple(row) for row in first},
            {("language", "classification_label")},
        )
        self.assertTrue(all(row["classification_label"] == "" for row in first))

    def test_exact_template_and_near_duplicates_are_rejected(self) -> None:
        candidates = (
            ("exact", "Please take the onion.", "Imperative"),
            ("template", "Please take the tomato.", "Imperative"),
            (
                "near",
                "Please take the onion from that nearby counter.",
                "Imperative",
            ),
            ("novel", "Could you rotate around the lower island?", "Imperative"),
        )
        rows, report = curate_candidate_pool(
            {
                "fixture": [
                    "Please take the onion.",
                    "Please take the onion from that nearby counter now.",
                ]
            },
            candidates,
            minimum_families_per_label=0,
        )

        languages = {row["language"] for row in rows}
        self.assertNotIn("Please take the onion.", languages)
        self.assertNotIn("Please take the tomato.", languages)
        self.assertNotIn(
            "Please take the onion from that nearby counter.", languages
        )
        self.assertIn("Could you rotate around the lower island?", languages)
        self.assertGreaterEqual(sum(report["rejections"].values()), 3)
        self.assertEqual(
            report["accuracy_boundary"][
                "these_candidates_change_reported_human_accuracy"
            ],
            False,
        )


if __name__ == "__main__":
    unittest.main()
