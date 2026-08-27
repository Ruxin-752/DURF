from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TEST_TMP = ROOT / "outputs" / "_human_feedback_form_holdout_test_tmp"
sys.path.insert(0, str(ROOT))

from scripts.evaluate_human_feedback_form_test import evaluate  # noqa: E402
from scripts.prepare_human_feedback_form_annotations import (  # noqa: E402
    template_family_signature,
)
from scripts.prepare_human_feedback_form_holdout import (  # noqa: E402
    DEFAULT_INPUT,
    PROTOCOL_VERSION,
    prepare_holdout,
    sha256_file,
    text_digest,
    value_digest,
)
from src.evaluation_splits import normalize_text  # noqa: E402
from src.feedback_form_classifier import FEEDBACK_TYPES  # noqa: E402


def _artifact(*, train_text: str | None = None, train_family: str | None = None) -> dict:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression

    texts = [
        "That was helpful",
        "That was a bad choice",
        "Please fetch an onion",
        "Move away from the pot",
        "You are blocking the path",
        "The counter is crowded",
    ]
    labels = [
        "evaluative",
        "evaluative",
        "imperative",
        "imperative",
        "descriptive",
        "descriptive",
    ]
    vectorizer = TfidfVectorizer(ngram_range=(1, 2))
    matrix = vectorizer.fit_transform(texts)
    classifier = LogisticRegression(max_iter=500, random_state=1).fit(matrix, labels)
    return {
        "model_type": "feedback_form_tfidf_logistic_regression",
        "labels": list(FEEDBACK_TYPES),
        "vectorizer": vectorizer,
        "classifier": classifier,
        "data_membership": {
            "train_normalized_text_sha256": (
                [text_digest(train_text)] if train_text is not None else []
            ),
            "dev_normalized_text_sha256": [],
            "human_train_template_family_sha256": (
                [value_digest(train_family)] if train_family is not None else []
            ),
            "human_dev_template_family_sha256": [],
        },
    }


def _rows() -> list[dict]:
    return [
        {
            "language": "I disliked that last decision.",
            "classification_label": "Evaluative",
        },
        {
            "language": "Keep the center aisle clear next time.",
            "classification_label": "Imperative",
        },
        {
            "language": "Waiting beside the pot blocks the route.",
            "classification_label": "Descriptive",
        },
    ]


class FrozenHumanFeedbackFormHoldoutTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        TEST_TMP.mkdir(parents=True, exist_ok=True)

    @classmethod
    def tearDownClass(cls) -> None:
        for path in TEST_TMP.iterdir():
            if path.is_file():
                path.unlink()
        try:
            TEST_TMP.rmdir()
        except OSError:
            pass

    def _seal(
        self,
        rows: list[dict],
        *,
        text_origin: str,
        minimum: int,
        stem: str,
    ) -> tuple[Path, Path, Path]:
        from joblib import dump

        model_path = TEST_TMP / f"{stem}.model.joblib"
        input_path = TEST_TMP / f"{stem}.jsonl"
        test_path = TEST_TMP / f"{stem}.test.json"
        sidecar_path = TEST_TMP / f"{stem}.sidecar.json"
        artifact = _artifact()
        dump(artifact, model_path)
        input_path.write_text(
            "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
        )
        prepared, report = prepare_holdout(
            rows,
            artifact,
            text_origin=text_origin,
            minimum_families_per_label=minimum,
            labels_confirmed_by_human=True,
        )
        test_path.write_text(
            json.dumps(prepared, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        report.update(
            {
                "public_input": str(input_path.resolve()),
                "public_input_sha256": sha256_file(input_path),
                "frozen_model": str(model_path.resolve()),
                "frozen_model_sha256": sha256_file(model_path),
                "prepared_test": str(test_path.resolve()),
                "prepared_test_sha256": sha256_file(test_path),
                "sidecar": str(sidecar_path.resolve()),
            }
        )
        sidecar_path.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        return model_path, test_path, sidecar_path

    def test_empty_window_and_template_keep_the_two_field_contract(self) -> None:
        nonblank = [line for line in DEFAULT_INPUT.read_text(encoding="utf-8").splitlines() if line.strip()]
        self.assertEqual(nonblank, [])
        template = DEFAULT_INPUT.with_name("human_feedback_form_holdout.template.jsonl")
        values = [json.loads(line) for line in template.read_text(encoding="utf-8").splitlines() if line.strip()]
        self.assertEqual(len(values), 3)
        self.assertTrue(
            all(set(value) == {"language", "classification_label"} for value in values)
        )

    def test_prepare_is_all_test_and_requires_explicit_human_confirmation(self) -> None:
        with self.assertRaisesRegex(ValueError, "human label confirmation"):
            prepare_holdout(
                _rows(),
                _artifact(),
                text_origin="human_authored",
                minimum_families_per_label=1,
            )
        prepared, report = prepare_holdout(
            _rows(),
            _artifact(),
            text_origin="human_authored",
            minimum_families_per_label=1,
            labels_confirmed_by_human=True,
        )
        self.assertTrue(prepared)
        self.assertTrue(all(row["split"] == "test" for row in prepared))
        self.assertTrue(all(row["source"] == "human_holdout" for row in prepared))
        self.assertEqual(report["protocol_version"], PROTOCOL_VERSION)
        self.assertTrue(report["claim_ready"])
        self.assertTrue(report["human_language_claim_ready"])

        invalid = [dict(_rows()[0], reviewer="not allowed")]
        with self.assertRaisesRegex(ValueError, "schema"):
            prepare_holdout(
                invalid,
                _artifact(),
                text_origin="human_authored",
                minimum_families_per_label=1,
                labels_confirmed_by_human=True,
            )

    def test_model_text_and_family_exposure_fail_closed(self) -> None:
        rows = _rows()
        exposed_text = normalize_text(rows[0]["language"])
        with self.assertRaisesRegex(ValueError, "exposed"):
            prepare_holdout(
                rows,
                _artifact(train_text=exposed_text),
                text_origin="human_authored",
                minimum_families_per_label=1,
                labels_confirmed_by_human=True,
            )
        family = template_family_signature(normalize_text(rows[1]["language"]))
        with self.assertRaisesRegex(ValueError, "exposed"):
            prepare_holdout(
                rows,
                _artifact(train_family=family),
                text_origin="human_authored",
                minimum_families_per_label=1,
                labels_confirmed_by_human=True,
            )

    def test_insufficient_holdout_does_not_reveal_model_metrics(self) -> None:
        model, test, sidecar = self._seal(
            _rows()[:2], text_origin="human_authored", minimum=1, stem="insufficient"
        )
        result = evaluate(model, test, sidecar)
        self.assertEqual(result["status"], "insufficient_human_test_data")
        self.assertFalse(result["claim_ready"])
        self.assertIsNone(result["metrics"])
        self.assertEqual(result["missing_labels"], ["Descriptive"])

        quota_model, quota_test, quota_sidecar = self._seal(
            _rows(), text_origin="human_authored", minimum=2, stem="below_quota"
        )
        quota_result = evaluate(quota_model, quota_test, quota_sidecar)
        self.assertEqual(quota_result["status"], "insufficient_human_test_data")
        self.assertEqual(
            quota_result["labels_below_family_quota"],
            {"Evaluative": 1, "Imperative": 1, "Descriptive": 1},
        )
        self.assertIsNone(quota_result["metrics"])

    def test_ready_ai_candidate_scope_is_not_human_language(self) -> None:
        model, test, sidecar = self._seal(
            _rows(),
            text_origin="ai_candidate_human_confirmed",
            minimum=1,
            stem="ai_candidate",
        )
        result = evaluate(model, test, sidecar)
        self.assertEqual(result["status"], "evaluated")
        self.assertTrue(result["claim_ready"])
        self.assertFalse(result["human_language_claim_ready"])
        self.assertEqual(result["metric_scope"], "human_verified_ai_candidate_language")
        self.assertIsNotNone(result["metrics"])

    def test_evaluator_rejects_test_or_sidecar_tampering(self) -> None:
        model, test, sidecar = self._seal(
            _rows(), text_origin="human_authored", minimum=1, stem="tamper"
        )
        original = test.read_text(encoding="utf-8")
        test.write_text(original + " ", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "prepared-test hash mismatch"):
            evaluate(model, test, sidecar)
        test.write_text(original, encoding="utf-8")

        model_bytes = model.read_bytes()
        model.write_bytes(model_bytes + b"tampered")
        with self.assertRaisesRegex(ValueError, "frozen-model hash mismatch"):
            evaluate(model, test, sidecar)
        model.write_bytes(model_bytes)

        value = json.loads(sidecar.read_text(encoding="utf-8"))
        value["provenance"]["label_source"] = "human_confirmed"
        sidecar.write_text(json.dumps(value), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "label-source contract mismatch"):
            evaluate(model, test, sidecar)


if __name__ == "__main__":
    unittest.main()
