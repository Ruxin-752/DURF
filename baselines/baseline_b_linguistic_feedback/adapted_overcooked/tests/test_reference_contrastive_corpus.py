from __future__ import annotations

import sys
import unittest
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.generate_reference_contrastive_corpus import (  # noqa: E402
    EXPLICIT_CUES,
    REFERENCE_TYPES,
    build_contrastive_corpus,
)
from src.evaluation_splits import normalize_text  # noqa: E402


class ReferenceContrastiveCorpusTests(unittest.TestCase):
    def test_generation_is_deterministic_balanced_and_disjoint(self) -> None:
        first, first_report = build_contrastive_corpus()
        second, second_report = build_contrastive_corpus()
        self.assertEqual(first, second)
        self.assertEqual(first_report, second_report)
        self.assertEqual(len(first), 375)
        self.assertEqual(
            Counter(row["reference_type"] for row in first),
            Counter({label: 75 for label in REFERENCE_TYPES}),
        )
        self.assertEqual(first_report["split_counts"], {"dev": 75, "test": 75, "train": 225})

        group_splits = defaultdict(set)
        family_splits = defaultdict(set)
        text_splits = defaultdict(set)
        for row in first:
            group_splits[row["group_id"]].add(row["split"])
            family_splits[row["paraphrase_family"]].add(row["split"])
            text_splits[normalize_text(row["text"])].add(row["split"])
        self.assertTrue(all(len(value) == 1 for value in group_splits.values()))
        self.assertTrue(all(len(value) == 1 for value in family_splits.values()))
        self.assertTrue(all(len(value) == 1 for value in text_splits.values()))

    def test_each_contrast_set_has_all_five_labels(self) -> None:
        rows, _report = build_contrastive_corpus()
        labels_by_set = defaultdict(set)
        for row in rows:
            labels_by_set[row["contrast_set_id"]].add(row["reference_type"])
            annotation = row["phrase_annotations"][0]
            self.assertEqual(annotation["text"], row["text"])
            self.assertEqual(annotation["reference_type"], row["reference_type"])
        self.assertTrue(labels_by_set)
        self.assertTrue(
            all(labels == set(REFERENCE_TYPES) for labels in labels_by_set.values())
        )

    def test_hard_classes_do_not_require_stock_scope_markers(self) -> None:
        rows, report = build_contrastive_corpus()
        for label, cues in EXPLICIT_CUES.items():
            cue_free = [
                row
                for row in rows
                if row["reference_type"] == label
                and not any(cue in row["text"].lower() for cue in cues)
            ]
            self.assertGreaterEqual(len(cue_free), 50)
            self.assertEqual(report["explicit_cue_free_counts"][label], len(cue_free))


if __name__ == "__main__":
    unittest.main()
