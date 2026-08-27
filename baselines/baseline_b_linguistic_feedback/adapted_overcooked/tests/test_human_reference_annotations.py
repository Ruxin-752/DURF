from __future__ import annotations

import json
import subprocess
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.prepare_human_reference_annotations import (  # noqa: E402
    REFERENCE_TYPES,
    prepare_human_annotations,
    stable_teacher_session_split,
)


def _group_for(split: str) -> tuple[str, str]:
    for index in range(10000):
        teacher = f"teacher_{split}_{index}"
        session = f"session_{index}"
        if stable_teacher_session_split(teacher, session) == split:
            return teacher, session
    raise AssertionError(f"could not find {split} hash fixture")


def _row(
    feedback_id: str,
    text: str,
    reference_type: str,
    split: str = "train",
    status: str = "single_annotated",
) -> dict:
    teacher, session = _group_for(split)
    return {
        "feedback_id": feedback_id,
        "text": text,
        "reference_type": reference_type,
        "teacher_id": teacher,
        "session_id": session,
        "scenario_id": "ring_onion_soup",
        "annotation_status": status,
        "annotation_revision": 1,
        "annotator_id": "annotator_a",
        "adjudicator_id": "annotator_b" if status == "adjudicated" else None,
    }


class HumanReferenceAnnotationTests(unittest.TestCase):
    def test_all_five_labels_are_accepted(self) -> None:
        rows = [
            _row(f"feedback_{index}", f"unique human phrase {index}", label)
            for index, label in enumerate(REFERENCE_TYPES)
        ]
        normalized, train, gold, report = prepare_human_annotations(rows)
        self.assertEqual({row["reference_type"] for row in normalized}, set(REFERENCE_TYPES))
        self.assertEqual(len(train), 5)
        self.assertEqual(gold, [])
        self.assertFalse(report["model_predictions_used_as_labels"])

    def test_single_annotation_is_never_dev_or_test_gold(self) -> None:
        rows = [
            _row("single_train", "train only phrase", "feature", "train"),
            _row("single_dev", "dev pending phrase", "trajectory", "dev"),
            _row("single_test", "test pending phrase", "other", "test"),
            _row(
                "gold_dev",
                "adjudicated dev phrase",
                "action_spatial",
                "dev",
                "adjudicated",
            ),
            _row(
                "gold_test",
                "adjudicated test phrase",
                "action_behavioral",
                "test",
                "adjudicated",
            ),
        ]
        normalized, train, gold, _ = prepare_human_annotations(rows)
        self.assertEqual([row["feedback_id"] for row in train], ["single_train"])
        self.assertEqual({row["feedback_id"] for row in gold}, {"gold_dev", "gold_test"})
        self.assertTrue(all(row["annotation_status"] == "adjudicated" for row in gold))
        releases = {row["feedback_id"]: row["release_status"] for row in normalized}
        self.assertEqual(releases["single_dev"], "withheld_dev_pending_adjudication")
        self.assertEqual(releases["single_test"], "withheld_test_pending_adjudication")

    def test_cross_label_exact_duplicate_is_rejected(self) -> None:
        rows = [
            _row("one", "Don't block THIS path!", "feature"),
            _row("two", "DON'T BLOCK this path.", "action_behavioral"),
        ]
        with self.assertRaisesRegex(ValueError, "cross-label conflicts"):
            prepare_human_annotations(rows)

    def test_required_provenance_status_and_label_are_validated(self) -> None:
        row = _row("bad", "ambiguous", "not_a_label")
        row.pop("scenario_id")
        row["annotation_status"] = "machine_predicted"
        with self.assertRaisesRegex(ValueError, "reference_type") as caught:
            prepare_human_annotations([row])
        message = str(caught.exception)
        self.assertIn("annotation_status", message)
        self.assertIn("scenario_id or group_id", message)

    def test_append_does_not_move_an_existing_teacher_session(self) -> None:
        original = _row("original", "first stable phrase", "feature", "dev", "adjudicated")
        first = prepare_human_annotations([original])[0][0]
        appended = _row("new", "later unrelated phrase", "other", "test", "adjudicated")
        second_rows = prepare_human_annotations([original, appended])[0]
        second = next(row for row in second_rows if row["feedback_id"] == "original")
        self.assertEqual(first["assigned_split"], "dev")
        self.assertEqual(second["assigned_split"], first["assigned_split"])

    def test_append_only_revision_can_be_adjudicated(self) -> None:
        first = _row("revision", "This whole run was wasteful", "trajectory", "test")
        second = dict(first)
        second.update(
            {
                "reference_type": "feature",
                "annotation_status": "adjudicated",
                "annotation_revision": 2,
                "adjudicator_id": "annotator_b",
            }
        )
        normalized, train, gold, report = prepare_human_annotations([first, second])
        self.assertEqual(train, [])
        self.assertEqual(len(normalized), 1)
        self.assertEqual(normalized[0]["reference_type"], "feature")
        self.assertEqual(gold[0]["feedback_id"], "revision")
        self.assertEqual(report["superseded_revisions"], 1)

    def test_missing_default_style_input_is_safe(self) -> None:
        script = ROOT / "scripts" / "prepare_human_reference_annotations.py"
        prefix = ROOT / "outputs" / "_human_reference_missing_test"
        paths = {
            name: prefix.with_name(f"{prefix.name}.{name}.json")
            for name in ("normalized", "train", "gold", "report")
        }
        missing = prefix.with_suffix(".missing.jsonl")
        for path in [missing, *paths.values()]:
            if path.exists():
                path.unlink()
            self.addCleanup(lambda target=path: target.unlink(missing_ok=True))
        command = [
            sys.executable,
            "-B",
            str(script),
            "--input",
            str(missing),
            "--normalized-output",
            str(paths["normalized"]),
            "--train-output",
            str(paths["train"]),
            "--gold-output",
            str(paths["gold"]),
            "--report",
            str(paths["report"]),
        ]
        completed = subprocess.run(command, capture_output=True, text=True, check=False)
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertEqual(json.loads(paths["normalized"].read_text("utf-8")), [])
        report = json.loads(paths["report"].read_text("utf-8"))
        self.assertFalse(report["input_exists"])
        self.assertEqual(report["input_rows"], 0)


if __name__ == "__main__":
    unittest.main()
