from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.generate_feedback_form_boundary_train import (  # noqa: E402
    DEFAULT_BASE,
    DEFAULT_HOLDOUT_SPECS,
    DEFAULT_PAPER_MANIFEST,
    FEEDBACK_TYPES,
    SCHEMA_FIELDS,
    build_boundary_corpus,
    generate_boundary_rows,
)
from scripts.run_feedback_form_boundary_candidates import resolve_min_df  # noqa: E402
from src.evaluation_splits import normalize_text  # noqa: E402


class FeedbackFormBoundaryTrainTests(unittest.TestCase):
    def test_candidate_min_df_inherits_baseline_unless_explicitly_overridden(self) -> None:
        report = {"feature_config": {"min_df": 1}}
        self.assertEqual(resolve_min_df(report, None), (1, 1))
        self.assertEqual(resolve_min_df(report, 2), (1, 2))
        with self.assertRaisesRegex(ValueError, "at least 1"):
            resolve_min_df(report, 0)

    def test_generation_is_balanced_deterministic_and_train_only(self) -> None:
        first, first_report = build_boundary_corpus()
        second, second_report = build_boundary_corpus()
        self.assertEqual(first, second)
        self.assertEqual(first_report, second_report)
        counts = Counter(row["expected_feedback_type"] for row in first)
        self.assertEqual(set(counts), set(FEEDBACK_TYPES))
        self.assertEqual(len(set(counts.values())), 1)
        self.assertGreater(counts["evaluative"], 48)
        self.assertEqual({row["split"] for row in first}, {"train"})
        self.assertEqual({row["source"] for row in first}, {"feedback_form_hard_train"})
        self.assertTrue(all(set(row) == SCHEMA_FIELDS for row in first))
        self.assertEqual(first_report["holdout_leakage_gate"]["exact_overlap_count"], 0)
        self.assertEqual(first_report["holdout_leakage_gate"]["near_overlap_count"], 0)
        self.assertFalse(first_report["holdout_leakage_gate"]["frozen_file_opened"])
        self.assertFalse(first_report["holdout_leakage_gate"]["test_files_opened"])
        self.assertFalse(first_report["holdout_leakage_gate"]["test_text_used_for_leakage_filter"])
        self.assertEqual(first_report["template_authoring"], "ai_assisted_codex")
        self.assertFalse(first_report["independent_human_semantic_audit"])
        self.assertTrue(all("test" not in path.name for path, _ in DEFAULT_HOLDOUT_SPECS))

    def test_contrast_groups_keep_all_three_labels(self) -> None:
        groups = defaultdict(set)
        for row in generate_boundary_rows(DEFAULT_BASE):
            groups[row["group_id"]].add(row["expected_feedback_type"])
            self.assertEqual(row["normalized"], normalize_text(row["text"]))
        self.assertTrue(all(labels == set(FEEDBACK_TYPES) for labels in groups.values()))

    def test_short_examples_cover_past_request_and_state_boundaries(self) -> None:
        rows = generate_boundary_rows(DEFAULT_BASE)
        by_label = {
            label: " ".join(
                row["text"].lower()
                for row in rows
                if row["expected_feedback_type"] == label
            )
            for label in FEEDBACK_TYPES
        }
        self.assertIn("that action was genuinely good", by_label["evaluative"])
        self.assertNotIn("that move was very good", by_label["evaluative"])
        self.assertIn("should have", by_label["evaluative"])
        self.assertIn("could you", by_label["imperative"])
        self.assertIn("you need to", by_label["imperative"])
        self.assertIn("the pot is ready", by_label["descriptive"])
        self.assertIn("keeping the aisle clear saves time", by_label["descriptive"])

    def test_exact_holdout_collision_rejects_entire_contrast_group(self) -> None:
        rows = generate_boundary_rows(DEFAULT_BASE)
        secret = next(
            row for row in rows if row["group_id"].startswith("feedback-form-boundary")
        )
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            suffix=".json",
            dir=ROOT / "outputs",
            delete=False,
        ) as handle:
            fixture = Path(handle.name)
            json.dump([{"text": secret["text"]}], handle)
        try:
            accepted, report = build_boundary_corpus(
                DEFAULT_BASE,
                ((fixture, False),),
                DEFAULT_PAPER_MANIFEST,
            )
        finally:
            fixture.unlink(missing_ok=True)
        accepted_groups = {row["group_id"] for row in accepted}
        self.assertNotIn(secret["group_id"], accepted_groups)
        self.assertEqual(
            report["holdout_leakage_gate"]["groups_rejected"]["exact_text"], 1
        )
        self.assertNotIn(secret["text"], json.dumps(report))

    def test_near_holdout_collision_rejects_entire_contrast_group(self) -> None:
        rows = generate_boundary_rows(DEFAULT_BASE)
        secret = next(
            row
            for row in rows
            if row["group_id"].startswith("feedback-form-boundary")
            and len(row["text"]) > 70
        )
        near_text = f"{secret['text'].rstrip('.')} please."
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            suffix=".json",
            dir=ROOT / "outputs",
            delete=False,
        ) as handle:
            fixture = Path(handle.name)
            json.dump([{"text": near_text}], handle)
        try:
            accepted, report = build_boundary_corpus(
                DEFAULT_BASE,
                ((fixture, False),),
                DEFAULT_PAPER_MANIFEST,
                near_duplicate_threshold=0.9,
            )
        finally:
            fixture.unlink(missing_ok=True)
        self.assertNotIn(secret["group_id"], {row["group_id"] for row in accepted})
        self.assertEqual(
            report["holdout_leakage_gate"]["groups_rejected"]["near_text"], 1
        )

    def test_frozen_membership_hash_rejects_without_opening_frozen_file(self) -> None:
        rows = generate_boundary_rows(DEFAULT_BASE)
        secret = next(
            row for row in rows if row["group_id"].startswith("feedback-form-boundary")
        )
        digest = hashlib.sha256(
            normalize_text(secret["text"]).encode("utf-8")
        ).hexdigest()
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            suffix=".json",
            dir=ROOT / "outputs",
            delete=False,
        ) as handle:
            manifest = Path(handle.name)
            json.dump(
                {
                    "frozen_test_membership": {
                        "normalized_text_sha256": [digest],
                        "paper_task_uuid_sha256": ["opaque-group-hash"],
                    }
                },
                handle,
            )
        try:
            accepted, report = build_boundary_corpus(
                DEFAULT_BASE, (), manifest, near_duplicate_threshold=1.0
            )
        finally:
            manifest.unlink(missing_ok=True)
        self.assertNotIn(secret["group_id"], {row["group_id"] for row in accepted})
        self.assertFalse(report["holdout_leakage_gate"]["frozen_file_opened"])


if __name__ == "__main__":
    unittest.main()
