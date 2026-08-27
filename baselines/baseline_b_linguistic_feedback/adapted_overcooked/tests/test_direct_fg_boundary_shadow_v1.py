from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts import generate_direct_fg_boundary_shadow_v1 as generator


def _canonical_hash(value) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class DirectFgBoundaryShadowDataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rows = generator.generate_train_rows()
        cls.audit = generator.audit_train_rows(cls.rows)

    def test_generation_is_deterministic_balanced_and_train_only(self):
        self.assertEqual(self.rows, generator.generate_train_rows())
        self.assertEqual(len(self.rows), 240)
        self.assertEqual(
            Counter(row["expected_feedback_type"] for row in self.rows),
            Counter({label: 80 for label in generator.LABELS}),
        )
        self.assertEqual(len({row["feedback_id"] for row in self.rows}), 240)
        self.assertEqual(len({row["normalized_text"] for row in self.rows}), 240)
        self.assertTrue(all(row["split"] == "train" for row in self.rows))
        self.assertEqual(self.audit["external_input_files_opened"], 0)
        self.assertEqual(self.audit["old_human_dev_test_frozen_rows_read"], 0)
        self.assertFalse(self.audit["reference_type_used_as_target"])

    def test_direct_three_class_current_ontology_contract(self):
        for row in self.rows:
            self.assertEqual(
                generator.detect_speech_acts(row["text"]),
                (row["expected_feedback_type"],),
            )
            self.assertIn(row["canonical_action"], generator.ALLOWED_ACTIONS)
            self.assertNotIn("reference_type", row)
            self.assertNotIn("reference_type", row["semantic_payload"])
            self.assertNotIn("expected_feedback_type", row["semantic_payload"])
            for compiled in generator.FORBIDDEN_RE:
                self.assertIsNone(compiled.search(row["text"]), row["text"])
            self.assertFalse(row["provenance"]["human_gold"])
            self.assertFalse(row["provenance"]["real_player_data"])

    def test_eid_triads_share_only_typed_payload(self):
        groups: dict[str, list[dict]] = defaultdict(list)
        for row in self.rows:
            groups[row["contrast_triad_id"]].append(row)
        self.assertEqual(len(groups), 80)
        for rows in groups.values():
            self.assertEqual(len(rows), 3)
            self.assertEqual(
                {row["expected_feedback_type"] for row in rows},
                set(generator.LABELS),
            )
            self.assertEqual(
                len({_canonical_hash(row["semantic_payload"]) for row in rows}), 1
            )

    def test_surface_bundles_support_all_labels_and_negative_commands(self):
        bundle_labels = Counter(
            (row["surface_bundle_id"], row["expected_feedback_type"])
            for row in self.rows
        )
        self.assertEqual(len(bundle_labels), 60)
        self.assertEqual(set(bundle_labels.values()), {4})
        negatives = [row for row in self.rows if row["stance"] == "negative_command"]
        self.assertEqual(len(negatives), 20)
        self.assertTrue(
            all(row["expected_feedback_type"] == "imperative" for row in negatives)
        )

    def test_known_regression_is_balanced_non_gate_and_never_train(self):
        known = generator.generate_known_regression_rows()
        self.assertEqual(len(known), 30)
        self.assertEqual(
            Counter(row["expected_feedback_type"] for row in known),
            Counter({label: 10 for label in generator.LABELS}),
        )
        self.assertTrue(
            all(row["role"] == "training_informed_regression_only" for row in known)
        )
        self.assertTrue(all(not row["independent_accuracy_gate"] for row in known))
        self.assertTrue(
            all(
                generator.detect_speech_acts(row["text"])
                == (row["expected_feedback_type"],)
                for row in known
            )
        )
        train_text = {row["normalized_text"] for row in self.rows}
        self.assertFalse(
            train_text
            & {generator.normalize_text(row["text"]) for row in known}
        )

    def test_right_now_distinguishes_command_from_fact(self):
        self.assertEqual(
            generator.detect_speech_acts("Right now, pick up the tomato."),
            ("imperative",),
        )
        self.assertEqual(
            generator.detect_speech_acts("Right now, the soup is cooking."),
            ("descriptive",),
        )

    def test_mixed_sentences_are_rejected_not_given_a_fourth_training_label(self):
        mixed = generator.generate_mixed_rejection_rows(self.rows)
        self.assertEqual(len(mixed), 24)
        self.assertTrue(
            all(row["expected_behavior"] == "reject_mixed_speech_act" for row in mixed)
        )
        for row in mixed:
            self.assertEqual(
                set(generator.detect_speech_acts(row["text"])),
                set(row["speech_acts"]),
            )
            self.assertEqual(len(row["speech_acts"]), 2)
        self.assertEqual(
            {row["expected_feedback_type"] for row in self.rows},
            set(generator.LABELS),
        )

    def test_generator_has_no_legacy_data_input_path(self):
        source = Path(generator.__file__).read_text(encoding="utf-8")
        self.assertNotIn("reference_classifier_feedback", source)
        self.assertNotRegex(source, r"(?:read_text|read_bytes|json\.load)\s*\(")
        self.assertNotRegex(source, r"direct_fg_clean_v[3456]")


if __name__ == "__main__":
    unittest.main()
