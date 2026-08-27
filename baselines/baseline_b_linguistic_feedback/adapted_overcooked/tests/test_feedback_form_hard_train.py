from __future__ import annotations

from collections import Counter, defaultdict
import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.generate_feedback_form_hard_train import (  # noqa: E402
    CANONICAL_LABELS,
    DEFAULT_REFERENCE,
    FEEDBACK_TYPES,
    SCHEMA_FIELDS,
    build_hard_train_corpus,
    generate_rows,
)
from src.evaluation_splits import normalize_text  # noqa: E402


class FeedbackFormHardTrainTests(unittest.TestCase):
    def test_generation_is_deterministic_balanced_and_train_only(self) -> None:
        first, first_report = build_hard_train_corpus(DEFAULT_REFERENCE)
        second, second_report = build_hard_train_corpus(DEFAULT_REFERENCE)
        self.assertEqual(first, second)
        self.assertEqual(first_report, second_report)
        self.assertEqual(len(first), 288)
        self.assertEqual(
            Counter(row["expected_feedback_type"] for row in first),
            Counter({label: 96 for label in FEEDBACK_TYPES}),
        )
        self.assertEqual({row["split"] for row in first}, {"train"})
        self.assertEqual({row["source"] for row in first}, {"feedback_form_hard_train"})
        self.assertEqual(
            {row["label_source"] for row in first},
            {"paper_mapping_contrastive_train_only"},
        )
        for row in first:
            self.assertEqual(set(row), SCHEMA_FIELDS)
            self.assertEqual(row["normalized"], normalize_text(row["text"]))
            self.assertEqual(
                row["classification_label"],
                CANONICAL_LABELS[row["expected_feedback_type"]],
            )
        self.assertEqual(first_report["normalized_duplicate_count"], 0)
        self.assertEqual(first_report["cross_label_exact_conflict_count"], 0)

    def test_twelve_families_per_class_and_complete_minimal_contrast_groups(self) -> None:
        rows = generate_rows()
        families = defaultdict(set)
        groups = defaultdict(set)
        for row in rows:
            families[row["expected_feedback_type"]].add(row["template_family"])
            groups[row["group_id"]].add(row["expected_feedback_type"])
        self.assertEqual(
            {label: len(values) for label, values in families.items()},
            {label: 12 for label in FEEDBACK_TYPES},
        )
        self.assertEqual(len(groups), 96)
        self.assertTrue(all(labels == set(FEEDBACK_TYPES) for labels in groups.values()))

    def test_human_like_modal_families_are_balanced_minimal_contrasts(self) -> None:
        rows = generate_rows()
        targeted = [
            row
            for row in rows
            if int(row["template_family"].rsplit("_", 1)[-1]) >= 6
        ]
        self.assertEqual(len(targeted), 144)
        self.assertEqual(
            Counter(row["expected_feedback_type"] for row in targeted),
            Counter({label: 48 for label in FEEDBACK_TYPES}),
        )
        imperative_text = " ".join(
            row["text"].lower()
            for row in targeted
            if row["expected_feedback_type"] == "imperative"
        )
        for cue in ("don't ", "should ", "need to ", "can you ", "could you ", "why don't you "):
            self.assertIn(cue, imperative_text)
        # Modal/action vocabulary also appears outside Imperative.  This makes
        # a cue-only shortcut insufficient to solve the contrastive corpus.
        non_imperative_text = " ".join(
            row["text"].lower()
            for row in targeted
            if row["expected_feedback_type"] != "imperative"
        )
        for cue in ("should ", "needs ", "can ", "could "):
            self.assertIn(cue, non_imperative_text)

    def test_real_v8_holdout_is_used_only_as_zero_overlap_gate(self) -> None:
        _rows, report = build_hard_train_corpus(DEFAULT_REFERENCE)
        gate = report["holdout_leakage_gate"]
        self.assertEqual(gate["exact_overlap_count"], 0)
        self.assertEqual(gate["group_overlap_count"], 0)
        self.assertFalse(gate["heldout_contents_output"])
        self.assertFalse(gate["heldout_labels_accessed"])
        self.assertFalse(gate["heldout_metrics_computed"])
        self.assertNotIn("metrics", gate)
        self.assertNotIn("rows", gate)

    def test_exact_text_and_group_leakage_are_rejected_without_echoing_content(self) -> None:
        generated = generate_rows()
        secret_text = generated[0]["text"]
        secret_group = generated[1]["group_id"]
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            suffix=".json",
            dir=ROOT / "outputs",
            delete=False,
        ) as handle:
            fixture = Path(handle.name)
            json.dump(
                [
                    {"split": "dev", "text": secret_text, "group_id": "safe"},
                    {"split": "test", "text": "unrelated", "group_id": secret_group},
                ],
                handle,
            )
        try:
            with self.assertRaisesRegex(ValueError, "exact_overlap_count=1") as caught:
                build_hard_train_corpus(fixture)
            message = str(caught.exception)
            self.assertIn("group_overlap_count=1", message)
            self.assertNotIn(secret_text, message)
            self.assertNotIn(secret_group, message)
        finally:
            fixture.unlink(missing_ok=True)


if __name__ == "__main__":
    unittest.main()
