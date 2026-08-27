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
TEST_ROOT = ROOT / "outputs" / "_assistant_curated_eval_unit_tests"
TEST_ROOT.mkdir(parents=True, exist_ok=True)

from scripts.evaluate_assistant_curated_feedback_form_candidates import (  # noqa: E402
    evaluate,
)
from scripts.generate_feedback_form_review_candidates import (  # noqa: E402
    REVIEW_STATUS,
)
from src.evaluation_splits import normalize_text  # noqa: E402
from src.feedback_form_classifier import FEEDBACK_TYPES  # noqa: E402


class _TextVectorizer:
    def transform(self, texts):
        return list(texts)


class _MarkerClassifier:
    classes_ = FEEDBACK_TYPES

    def predict(self, texts):
        output = []
        for text in texts:
            if "[eval]" in text:
                output.append("evaluative")
            elif "[imp]" in text:
                output.append("imperative")
            else:
                output.append("descriptive")
        return output


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _text_sha(text: str) -> str:
    return hashlib.sha256(normalize_text(text).encode("utf-8")).hexdigest()


class AssistantCuratedRegressionEvaluatorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = TEST_ROOT / f"case_{uuid.uuid4().hex}"
        self.directory.mkdir()

    def tearDown(self) -> None:
        resolved = self.directory.resolve()
        if resolved.parent != TEST_ROOT.resolve():
            raise RuntimeError(f"refusing to remove unexpected test path {resolved}")
        shutil.rmtree(resolved)

    def _fixture(self, *, overlap: bool = False) -> tuple[Path, Path, Path, Path]:
        candidate_rows = [
            {
                "language": "[eval] A singular judgement of the exchange.",
                "classification_label": "Evaluative",
                "review_status": REVIEW_STATUS,
            },
            {
                "language": "[imp] Rotate around the lower kitchen island.",
                "classification_label": "Imperative",
                "review_status": REVIEW_STATUS,
            },
            {
                "language": "[desc] The upper corridor has no free tile.",
                "classification_label": "Descriptive",
                "review_status": REVIEW_STATUS,
            },
        ]
        candidates = self.directory / "candidates.jsonl"
        candidates.write_text(
            "".join(json.dumps(row) + "\n" for row in candidate_rows),
            encoding="utf-8",
        )
        counts = {"Evaluative": 1, "Imperative": 1, "Descriptive": 1}
        candidate_report = self.directory / "candidates.report.json"
        candidate_report.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "provenance": REVIEW_STATUS,
                    "human_authorship_claimed": False,
                    "human_review_completed": False,
                    "training_eligible": False,
                    "evaluation_eligible": False,
                    "model_or_classifier_used_to_assign_labels": False,
                    "candidate_schema": {
                        "exact_fields": [
                            "language",
                            "classification_label",
                            "review_status",
                        ],
                        "required_review_status": REVIEW_STATUS,
                    },
                    "output": str(candidates),
                    "output_sha256": _sha256(candidates),
                    "candidate_count": 3,
                    "distinct_surface_family_count": 3,
                    "counts_by_label": counts,
                    "novelty_guard": {
                        "exact_collision_in_emitted_pool": 0,
                        "template_family_collision_in_emitted_pool": 0,
                        "near_duplicate_in_emitted_pool": 0,
                    },
                    "suggested_label_boundary": {"never_valid_as": "human gold"},
                }
            ),
            encoding="utf-8",
        )

        train_text = candidate_rows[0]["language"] if overlap else "unrelated train"
        manifest = {"human_test_loaded": False, "seed": 7}
        artifact = {
            "model_type": "feedback_form_tfidf_logistic_regression",
            "input_mode": "raw_text",
            "labels": list(FEEDBACK_TYPES),
            "vectorizer": _TextVectorizer(),
            "classifier": _MarkerClassifier(),
            "manifest": manifest,
            "data_membership": {
                "train_normalized_text_sha256": [_text_sha(train_text)],
                "dev_normalized_text_sha256": [_text_sha("unrelated dev")],
            },
        }
        model = self.directory / "model.joblib"
        dump(artifact, model)
        training_report = self.directory / "model.report.json"
        training_report.write_text(
            json.dumps(
                {
                    "artifact_model_sha256": _sha256(model),
                    "artifact_manifest": manifest,
                    "test_metrics": None,
                }
            ),
            encoding="utf-8",
        )
        return model, training_report, candidates, candidate_report

    def test_model_only_regression_is_read_only_and_not_human_accuracy(self) -> None:
        model, training_report, candidates, candidate_report = self._fixture()
        before = {path: _sha256(path) for path in self._fixture_paths(
            model, training_report, candidates, candidate_report
        )}
        result = evaluate(
            model,
            candidates,
            candidate_report,
            training_report_path=training_report,
            expected_model_sha256=before[model],
            expected_training_report_sha256=before[training_report],
            expected_candidate_sha256=before[candidates],
            expected_candidate_report_sha256=before[candidate_report],
        )

        self.assertEqual(result["benchmark_role"], "assistant_curated_regression")
        self.assertFalse(result["human_accuracy_claim"])
        self.assertFalse(result["used_for_model_selection"])
        self.assertFalse(result["runtime_low_confidence_fallback_used"])
        self.assertEqual(result["metrics"]["accuracy"], 1.0)
        self.assertEqual(result["metrics"]["balanced_accuracy"], 1.0)
        self.assertEqual(result["metrics"]["macro_f1"], 1.0)
        self.assertTrue(
            all(value == 0 for value in result["membership_overlap_audit"].values())
        )
        self.assertTrue(result["model_sha256_unchanged"])
        for path, digest in before.items():
            self.assertEqual(_sha256(path), digest)

    def test_fails_closed_on_candidate_sha_or_review_status(self) -> None:
        model, training_report, candidates, candidate_report = self._fixture()
        rows = candidates.read_text(encoding="utf-8").splitlines()
        changed = json.loads(rows[0])
        changed["review_status"] = "human"
        rows[0] = json.dumps(changed)
        candidates.write_text("\n".join(rows) + "\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "candidate data SHA256"):
            evaluate(
                model,
                candidates,
                candidate_report,
                training_report_path=training_report,
            )
        report = json.loads(candidate_report.read_text(encoding="utf-8"))
        report["output_sha256"] = _sha256(candidates)
        candidate_report.write_text(json.dumps(report), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "review_status"):
            evaluate(
                model,
                candidates,
                candidate_report,
                training_report_path=training_report,
            )

    def test_fails_closed_on_training_membership_overlap(self) -> None:
        model, training_report, candidates, candidate_report = self._fixture(overlap=True)
        with self.assertRaisesRegex(ValueError, "membership leakage"):
            evaluate(
                model,
                candidates,
                candidate_report,
                training_report_path=training_report,
            )

    @staticmethod
    def _fixture_paths(*paths: Path) -> tuple[Path, ...]:
        return tuple(paths)


if __name__ == "__main__":
    unittest.main()
