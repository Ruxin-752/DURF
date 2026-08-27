from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts import predict_direct_fg_boundary_shadow_v1 as predictor
from scripts import train_direct_fg_boundary_shadow_v1 as trainer


class BoundaryShadowTrainerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rows = trainer.load_locked_train_rows(trainer.DEFAULT_TRAIN_JSON)
        cls.output_dir = trainer.DEFAULT_OUTPUT_DIR
        cls.report = json.loads(
            (cls.output_dir / "training_report.json").read_text(encoding="utf-8")
        )
        cls.config = json.loads(
            (cls.output_dir / "frozen_config.json").read_text(encoding="utf-8")
        )
        cls.manifest = json.loads(
            (cls.output_dir / "manifest.json").read_text(encoding="utf-8")
        )
        cls.artifact = predictor.load_artifact(cls.output_dir / "model.joblib")

    def test_locked_canonical_hash_fails_before_fit(self):
        self.assertEqual(
            trainer.canonical_hash(self.rows),
            trainer.EXPECTED_TRAIN_CANONICAL_SHA256,
        )
        round_tripped = json.loads(
            json.dumps(self.rows, ensure_ascii=False, separators=(",", ":"))
        )
        self.assertEqual(
            trainer.canonical_hash(trainer._validate_locked_rows(round_tripped)),
            trainer.EXPECTED_TRAIN_CANONICAL_SHA256,
        )
        changed = json.loads(json.dumps(self.rows))
        changed[0]["text"] += " changed"
        with mock.patch.object(
            trainer,
            "build_vectorizer",
            side_effect=AssertionError("fit must not start"),
        ):
            with self.assertRaisesRegex(ValueError, "canonical SHA-256 mismatch"):
                trainer._validate_locked_rows(changed)

    def test_loader_reads_only_the_explicit_train_json(self):
        path = trainer.DEFAULT_TRAIN_JSON
        original = Path.read_text
        opened = []

        def guarded(candidate, *args, **kwargs):
            opened.append(candidate.resolve())
            return original(candidate, *args, **kwargs)

        with mock.patch.object(Path, "read_text", guarded):
            trainer.load_locked_train_rows(path)
        self.assertEqual(opened, [path.resolve()])
        source = Path(trainer.__file__).read_text(encoding="utf-8")
        self.assertNotRegex(source, r"\.(?:glob|rglob)\s*\(")
        self.assertNotIn("generate_known_regression_rows", source)
        self.assertNotIn("generate_mixed_rejection_rows", source)
        self.assertNotIn("import generate_direct_fg_boundary", source)

    def test_raw_text_is_the_only_model_feature(self):
        text = self.rows[0]["text"]
        left = {"text": text, "leaked_label": "imperative", "semantic_payload": {}}
        right = {"text": text, "leaked_label": "descriptive", "canonical_action": "X"}
        self.assertEqual(trainer._texts([left]), trainer._texts([right]))
        self.assertEqual(trainer.extract_model_input(text), text)
        with self.assertRaises(TypeError):
            trainer.extract_model_input(left)
        self.assertEqual(
            self.artifact["model_input_contract"],
            {"input_type": "raw_string_only", "field": "text", "metadata_features": []},
        )

    def test_nested_five_by_five_group_ledger_is_disjoint(self):
        nested = self.report["nested_group_oof"]
        self.assertEqual(nested["outer_splits"], 5)
        self.assertEqual(nested["inner_splits"], 5)
        self.assertTrue(nested["each_row_predicted_once_out_of_fold"])
        all_groups = set(row["surface_bundle_id"] for row in self.rows)
        seen_validation = set()
        for fold in nested["folds"]:
            train_groups = set(fold["outer_train_groups"])
            validation_groups = set(fold["outer_validation_groups"])
            self.assertFalse(train_groups & validation_groups)
            self.assertEqual(train_groups | validation_groups, all_groups)
            self.assertFalse(seen_validation & validation_groups)
            seen_validation |= validation_groups
            inner = fold["inner_selection"]
            self.assertEqual(inner["n_splits"], 5)
            self.assertEqual(inner["group_count"], len(train_groups))
            for candidate in inner["candidates"]:
                self.assertEqual(len(candidate["folds"]), 5)
                self.assertEqual(
                    sum(item["metrics"]["row_count"] for item in candidate["folds"]),
                    fold["outer_train_rows"],
                )
        self.assertEqual(seen_validation, all_groups)

    def test_temperature_is_nested_oof_only_and_argmax_preserving(self):
        calibration = self.report["temperature_calibration"]
        self.assertEqual(
            calibration["fit_scope"], "synthetic_train_nested_group_oof_only"
        )
        self.assertTrue(calibration["argmax_unchanged"])
        self.assertFalse(calibration["independently_validated"])
        self.assertGreater(calibration["temperature"], 0.0)
        self.assertEqual(
            self.report["nested_oof_temperature_scaled_metrics"]["row_count"], 240
        )
        self.assertEqual(
            self.report["nested_oof_uncalibrated_metrics"]["confusion_matrix"],
            self.report["nested_oof_temperature_scaled_metrics"]["confusion_matrix"],
        )

    def test_final_config_selection_is_train_group_cv_only(self):
        selection = self.report["final_train_only_selection"]
        self.assertEqual(selection["selection_scope"], "synthetic_train_group_cv_only")
        self.assertEqual(selection["n_splits"], 5)
        self.assertEqual(selection["group_count"], 20)
        self.assertEqual(
            selection["selected_config_id"],
            self.report["selected_hyperparameters"]["config_id"],
        )
        self.assertEqual(self.config["selected_hyperparameters"], self.report["selected_hyperparameters"])

    def test_claims_are_diagnostic_only_and_auxiliary_sets_are_excluded(self):
        self.assertTrue(self.report["diagnostic_only"])
        self.assertFalse(self.report["promotion_eligible"])
        self.assertFalse(self.report["production_promotion_eligible"])
        self.assertFalse(self.report["confidence_ready_for_player_ui"])
        self.assertIn("not independent", self.report["metric_claim"])
        self.assertEqual(
            Counter(self.report["train_input"]["label_counts"]),
            Counter({"descriptive": 80, "evaluative": 80, "imperative": 80}),
        )
        for value in self.report["excluded_from_trainer"].values():
            self.assertFalse(value["opened"])
            self.assertFalse(value.get("used", value.get("fit", False)))
        serialized = json.dumps(self.report["nested_oof_temperature_scaled_metrics"])
        self.assertNotIn("known", serialized.lower())
        self.assertNotIn("mixed", serialized.lower())

    def test_manifest_binds_model_config_predictor_and_train(self):
        artifacts = self.manifest["artifacts"]
        self.assertEqual(
            artifacts["model"]["sha256"],
            trainer.file_sha256(Path(artifacts["model"]["path"])),
        )
        self.assertEqual(
            artifacts["frozen_config"]["canonical_sha256"],
            trainer.canonical_hash(self.config),
        )
        self.assertEqual(
            artifacts["predictor"]["sha256"],
            trainer.file_sha256(Path(predictor.__file__)),
        )
        self.assertEqual(
            artifacts["locked_train_copy"]["canonical_sha256"],
            trainer.EXPECTED_TRAIN_CANONICAL_SHA256,
        )
        self.assertFalse(self.manifest["train_input"]["directory_glob_used"])
        self.assertFalse(self.manifest["private_diagnostic_opened"])
        self.assertFalse(self.manifest["known_or_mixed_opened_by_trainer"])
        content = bytearray(Path(artifacts["model"]["path"]).read_bytes())
        content[-1] ^= 1
        self.assertNotEqual(
            hashlib.sha256(content).hexdigest(), artifacts["model"]["sha256"]
        )

    def test_reload_prediction_contract_is_exact(self):
        model_path = self.output_dir / "model.joblib"
        first = predictor.load_artifact(model_path)
        second = predictor.load_artifact(model_path)
        fixtures = (
            "Please wait while the soup cooks.",
            "That last move was helpful.",
            "At present, one tomato is inside the pot.",
        )
        for text in fixtures:
            left = predictor.predict_text(first, text)
            right = predictor.predict_text(second, text)
            self.assertEqual(left, right)
            self.assertAlmostEqual(sum(left["probabilities"].values()), 1.0, places=12)
            self.assertEqual(
                left["label"], max(left["probabilities"], key=left["probabilities"].get)
            )
            self.assertFalse(left["promotion_eligible"])
        with self.assertRaises(TypeError):
            predictor.predict_text(first, {"text": fixtures[0]})

    def test_private_diagnostic_receipt_is_aggregate_only_and_hash_bound(self):
        receipt = json.loads(
            (self.output_dir / "private_diagnostic_v2_one_shot_receipt.json").read_text(
                encoding="utf-8"
            )
        )
        binding = receipt["artifact_binding"]
        self.assertEqual(binding["model_sha256"], self.manifest["artifacts"]["model"]["sha256"])
        self.assertEqual(
            binding["frozen_config_file_sha256"],
            self.manifest["artifacts"]["frozen_config"]["sha256"],
        )
        self.assertEqual(
            binding["predictor_sha256"],
            self.manifest["artifacts"]["predictor"]["sha256"],
        )
        self.assertFalse(receipt["diagnostic"]["private_text_or_row_predictions_stored_here"])
        self.assertEqual(receipt["execution_receipt"]["successful_full_batch_evaluations"], 1)
        self.assertEqual(receipt["execution_receipt"]["pre_inference_launcher_failure_rows_inferred"], 0)
        self.assertEqual(receipt["results"]["scored"]["correct"], 59)
        self.assertEqual(receipt["results"]["scored"]["rows"], 72)
        self.assertEqual(receipt["results"]["ambiguity_challenges"]["rows"], 9)
        self.assertEqual(
            receipt["results"]["ambiguity_challenges"][
                "forced_classification_at_or_above_0_55"
            ],
            9,
        )
        self.assertTrue(receipt["interpretation"]["failed_target"])
        self.assertFalse(receipt["interpretation"]["promotion_eligible"])
        self.assertFalse(
            receipt["interpretation"]["can_be_used_as_independent_tuning_data_again"]
        )

    def test_output_directory_is_isolated_and_allowlisted(self):
        self.assertEqual(
            {path.name for path in self.output_dir.iterdir()},
            {
                "frozen_config.json",
                "manifest.json",
                "model.joblib",
                "private_diagnostic_v2_one_shot_receipt.json",
                "train.locked.json",
                "training_report.json",
            },
        )
        self.assertNotIn("production", str(self.output_dir).lower())


if __name__ == "__main__":
    unittest.main()
