from __future__ import annotations

import hashlib
import json
import shutil
import sys
import unittest
import uuid
from pathlib import Path

from joblib import dump


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
TEST_ROOT = ROOT / "outputs" / "_synthetic_form_eval_unit_tests"
TEST_ROOT.mkdir(parents=True, exist_ok=True)

from scripts.evaluate_synthetic_feedback_form_test import evaluate  # noqa: E402
from src.evaluation_splits import normalize_text  # noqa: E402
from src.feedback_form_classifier import FEEDBACK_TYPES  # noqa: E402


class _TextVectorizer:
    def transform(self, texts):
        return list(texts)


class _MarkerClassifier:
    classes_ = FEEDBACK_TYPES

    def predict(self, texts):
        predictions = []
        for text in texts:
            if "[eval]" in text:
                predictions.append("evaluative")
            elif "[imp]" in text:
                predictions.append("imperative")
            elif "[desc]" in text:
                predictions.append("descriptive")
            else:
                predictions.append("descriptive")
        return predictions


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _text_sha(text: str) -> str:
    return hashlib.sha256(normalize_text(text).encode("utf-8")).hexdigest()


class SyntheticFeedbackFormEvaluatorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = TEST_ROOT / f"case_{uuid.uuid4().hex}"
        self.directory.mkdir()

    def tearDown(self) -> None:
        resolved = self.directory.resolve()
        if resolved.parent != TEST_ROOT.resolve():
            raise RuntimeError(f"refusing to remove unexpected test path {resolved}")
        shutil.rmtree(resolved)

    def _fixture(
        self,
        *,
        text_overlap: bool = False,
        group_overlap: bool = False,
    ) -> tuple[Path, Path, Path]:
        reference_rows = (
            ("trajectory", "[eval]"),
            ("action_spatial", "[imp]"),
            ("feature", "[desc]"),
        )
        rows = []
        train_texts = []
        dev_texts = []
        for split, destination in (("train", train_texts), ("dev", dev_texts)):
            for index, (reference_type, marker) in enumerate(reference_rows):
                text = f"{split} {marker} example {index}"
                destination.append(text)
                rows.append(
                    {
                        "split": split,
                        "reference_type": reference_type,
                        "text": text,
                        "group_id": f"{split}_group_{index}",
                        "classifier_corpus_origin": "fixture_membership",
                    }
                )
        for origin in ("fixture_templates", "fixture_paraphrases"):
            for index, (reference_type, marker) in enumerate(reference_rows):
                text = f"test {origin} {marker} example {index}"
                if text_overlap and origin == "fixture_templates" and index == 0:
                    text = train_texts[0]
                group_id = f"test_{origin}_{index}"
                if group_overlap and origin == "fixture_templates" and index == 0:
                    group_id = "train_group_0"
                rows.append(
                    {
                        "split": "test",
                        "reference_type": reference_type,
                        "text": text,
                        "group_id": group_id,
                        "classifier_corpus_origin": origin,
                        # Deliberately unrelated: evaluator must ignore this target.
                        "expected_feedback_type": "descriptive",
                    }
                )
        rows.extend(
            [
                {
                    "split": "test",
                    "reference_type": "action_behavioral",
                    "text": "excluded behavioral row",
                    "group_id": "excluded_behavioral",
                },
                {
                    "split": "test",
                    "reference_type": "other",
                    "text": "excluded other row",
                    "group_id": "excluded_other",
                },
            ]
        )
        corpus = self.directory / "synthetic.json"
        corpus.write_text(json.dumps(rows), encoding="utf-8")
        manifest = {
            "synthetic_input": str(corpus),
            "synthetic_input_sha256": _sha256(corpus),
            "human_test_loaded": False,
        }
        artifact = {
            "model_type": "feedback_form_tfidf_logistic_regression",
            "model_version": 1,
            "input_mode": "raw_text",
            "labels": list(FEEDBACK_TYPES),
            "vectorizer": _TextVectorizer(),
            "classifier": _MarkerClassifier(),
            "data_membership": {
                "train_normalized_text_sha256": [_text_sha(text) for text in train_texts],
                "dev_normalized_text_sha256": [_text_sha(text) for text in dev_texts],
            },
            "manifest": manifest,
        }
        model = self.directory / "model.joblib"
        dump(artifact, model)
        training_report = self.directory / "model.report.json"
        training_report.write_text(
            json.dumps(
                {
                    "artifact_model_sha256": _sha256(model),
                    "artifact_manifest": manifest,
                }
            ),
            encoding="utf-8",
        )
        return model, training_report, corpus

    def test_evaluates_only_three_mapped_test_classes_by_origin(self) -> None:
        model, training_report, _corpus = self._fixture()
        before = _sha256(model)
        result = evaluate(model, training_report_path=training_report)
        self.assertEqual(result["test_rows"], 6)
        self.assertEqual(result["metrics"]["accuracy"], 1.0)
        self.assertEqual(result["metrics"]["balanced_accuracy"], 1.0)
        self.assertEqual(result["metrics"]["macro_f1"], 1.0)
        self.assertEqual(
            set(result["by_classifier_corpus_origin"]),
            {"fixture_templates", "fixture_paraphrases"},
        )
        self.assertEqual(
            result["corpus_audit"]["excluded_test_reference_counts"],
            {"action_behavioral": 1, "other": 1},
        )
        self.assertTrue(
            all(value == 0 for value in result["membership_overlap_audit"].values())
        )
        self.assertEqual(_sha256(model), before)
        self.assertTrue(result["model_sha256_unchanged"])

    def test_fails_closed_on_normalized_text_overlap(self) -> None:
        model, training_report, _corpus = self._fixture(text_overlap=True)
        with self.assertRaisesRegex(ValueError, "membership leakage"):
            evaluate(model, training_report_path=training_report)

    def test_fails_closed_on_group_overlap(self) -> None:
        model, training_report, _corpus = self._fixture(group_overlap=True)
        with self.assertRaisesRegex(ValueError, "membership leakage"):
            evaluate(model, training_report_path=training_report)

    def test_fails_closed_on_input_or_model_sha_mismatch(self) -> None:
        model, training_report, corpus = self._fixture()
        frozen_report = json.loads(training_report.read_text(encoding="utf-8"))
        frozen_report["artifact_model_sha256"] = "0" * 64
        training_report.write_text(json.dumps(frozen_report), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "frozen training report"):
            evaluate(model, training_report_path=training_report)
        frozen_report["artifact_model_sha256"] = _sha256(model)
        training_report.write_text(json.dumps(frozen_report), encoding="utf-8")

        corpus.write_text(corpus.read_text(encoding="utf-8") + " ", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "synthetic input SHA256"):
            evaluate(model, training_report_path=training_report)
        with self.assertRaisesRegex(ValueError, "frozen expected value"):
            evaluate(
                model,
                training_report_path=training_report,
                expected_model_sha256="0" * 64,
            )


if __name__ == "__main__":
    unittest.main()
