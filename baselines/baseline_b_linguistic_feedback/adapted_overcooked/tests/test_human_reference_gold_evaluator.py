from __future__ import annotations

import hashlib
import json
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.evaluate_human_reference_gold import evaluate_human_gold  # noqa: E402
from src.phrase_reference_classifier import REFERENCE_TYPES  # noqa: E402


def _gold(
    feedback_id: str,
    text: str,
    label: str,
    split: str,
    teacher: str = "heldout_teacher",
    session: str = "heldout_session",
) -> dict:
    return {
        "feedback_id": feedback_id,
        "text": text,
        "reference_type": label,
        "annotation_status": "adjudicated",
        "annotator_id": "annotator_a",
        "adjudicator_id": "annotator_b",
        "teacher_id": teacher,
        "session_id": session,
        "split": split,
        "assigned_split": split,
    }


class HumanReferenceGoldEvaluatorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        from joblib import dump
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression

        cls.source_path = ROOT / "outputs" / "_human_gold_eval_test_source.json"
        cls.model_path = ROOT / "outputs" / "_human_gold_eval_test_model.joblib"
        source_rows = [
            {
                "feedback_id": f"train_{index}",
                "text": f"training {label} marker {index}",
                "reference_type": label,
                "split": "train",
                "teacher_id": "training_teacher",
                "session_id": "training_session",
            }
            for index, label in enumerate(REFERENCE_TYPES)
        ]
        cls.source_path.write_text(
            json.dumps(source_rows, ensure_ascii=False), encoding="utf-8"
        )
        texts = [row["text"] for row in source_rows]
        labels = [row["reference_type"] for row in source_rows]
        vectorizer = TfidfVectorizer().fit(texts)
        classifier = LogisticRegression(random_state=1).fit(
            vectorizer.transform(texts), labels
        )
        source_sha = hashlib.sha256(cls.source_path.read_bytes()).hexdigest()
        dump(
            {
                "vectorizer": vectorizer,
                "classifier": classifier,
                "input_mode": "raw_phrase",
                "manifest": {
                    "input": str(cls.source_path),
                    "input_sha256": source_sha,
                    "augment": [],
                    "preprocessing": "raw_phrase",
                },
            },
            cls.model_path,
        )

    @classmethod
    def tearDownClass(cls) -> None:
        cls.model_path.unlink(missing_ok=True)
        cls.source_path.unlink(missing_ok=True)

    def test_empty_gold_is_unknown_and_has_no_accuracy_claim(self) -> None:
        report = evaluate_human_gold([], model_path=self.model_path)
        self.assertEqual(report["evaluation_status"], "unknown_no_human_gold")
        self.assertEqual(report["gold_rows"], 0)
        self.assertIsNone(report["dev"]["accuracy"])
        self.assertIsNone(report["test"]["macro_f1"])
        self.assertFalse(report["final_human_accuracy_claim_allowed"])
        self.assertTrue(report["model"]["unchanged"])

    def test_dev_and_test_are_reported_separately(self) -> None:
        rows = [
            _gold("dev_1", "heldout complete sequence comment", "trajectory", "dev"),
            _gold(
                "test_1",
                "heldout repeated behavior comment",
                "action_behavioral",
                "test",
                teacher="heldout_teacher_2",
                session="heldout_session_2",
            ),
        ]
        report = evaluate_human_gold(rows, model_path=self.model_path)
        self.assertEqual(report["dev"]["rows"], 1)
        self.assertEqual(report["test"]["rows"], 1)
        self.assertIsInstance(report["dev"]["accuracy"], float)
        self.assertIsInstance(report["test"]["macro_f1"], float)
        self.assertEqual(len(report["test"]["confusion_matrix"]), 5)
        self.assertTrue(report["final_human_accuracy_claim_allowed"])

    def test_normalized_text_overlap_fails_closed(self) -> None:
        row = _gold(
            "overlap_text",
            "TRAINING trajectory marker 0!",
            "trajectory",
            "test",
        )
        with self.assertRaisesRegex(ValueError, "normalized_text=1"):
            evaluate_human_gold([row], model_path=self.model_path)

    def test_teacher_session_overlap_fails_closed(self) -> None:
        row = _gold(
            "overlap_group",
            "otherwise unique heldout phrase",
            "feature",
            "test",
            teacher="training_teacher",
            session="training_session",
        )
        with self.assertRaisesRegex(ValueError, "teacher_session=1"):
            evaluate_human_gold([row], model_path=self.model_path)

    def test_non_adjudicated_or_train_rows_are_rejected(self) -> None:
        single = _gold("single", "unique single phrase", "other", "dev")
        single["annotation_status"] = "single_annotated"
        train = _gold("train", "unique train phrase", "feature", "train")
        with self.assertRaisesRegex(ValueError, "adjudicated only"):
            evaluate_human_gold([single], model_path=self.model_path)
        with self.assertRaisesRegex(ValueError, "dev or test"):
            evaluate_human_gold([train], model_path=self.model_path)


if __name__ == "__main__":
    unittest.main()
