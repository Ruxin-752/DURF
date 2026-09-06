from __future__ import annotations

import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.train_paper_feedback_recipe import audit_split, load_rows, PAPER_LABEL_SOURCE


class PaperRecipeDataTests(unittest.TestCase):
    def read(self, rows, semantics="speech_act"):
        with patch.object(Path, "read_text", return_value=json.dumps(rows)):
            return load_rows(Path("train.json"), "train", semantics)

    def rows(self):
        return [
            {"text": text, "classification_label": label, "split": "train",
             "group_id": str(index), "label_source": "human_explicit"}
            for index, (text, label) in enumerate([
                ("Nice work.", "Evaluative"), ("Fetch a dish.", "Imperative"),
                ("Soup is ready.", "Descriptive"),
            ])
        ]

    def test_reference_metadata_cannot_replace_explicit_target(self):
        rows = self.rows()
        rows[0]["reference_type"] = "action_spatial"
        self.assertEqual(self.read(rows)[0]["label"], "evaluative")
        del rows[0]["classification_label"]
        with self.assertRaisesRegex(ValueError, "explicit text/label"):
            self.read(rows)

    def test_paper_and_speech_labels_require_distinct_declared_sources(self):
        rows = self.rows()
        for row in rows:
            row["label_source"] = PAPER_LABEL_SOURCE
        with self.assertRaisesRegex(ValueError, "reviewed human labels"):
            self.read(rows)
        self.assertEqual(len(self.read(rows, "paper_grounding")), 3)

    def test_rejects_frozen_rows_and_conflicting_explicit_targets(self):
        rows = self.rows()
        rows[0]["split"] = "test"
        with self.assertRaisesRegex(ValueError, "unexpected split"):
            self.read(rows)
        rows[0]["split"] = "train"
        rows[0]["expected_feedback_type"] = "descriptive"
        with self.assertRaisesRegex(ValueError, "conflicting explicit labels"):
            self.read(rows)

    def test_rejects_group_and_normalized_text_leakage(self):
        train = [{"text": "Good job!", "group": "teacher-a"}]
        with self.assertRaisesRegex(ValueError, "1 groups"):
            audit_split(train, [{"text": "Another sentence", "group": "teacher-a"}])
        with self.assertRaisesRegex(ValueError, "1 normalized texts"):
            audit_split(train, [{"text": "GOOD job.", "group": "teacher-b"}])
        self.assertEqual(audit_split(train, [
            {"text": "A separate sentence", "group": "teacher-b"},
        ])["normalized_text_overlap"], 0)


if __name__ == "__main__":
    unittest.main()
