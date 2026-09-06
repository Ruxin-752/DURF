from __future__ import annotations

import argparse
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = ROOT / "scripts" / "export_web_models.py"
SPEC = importlib.util.spec_from_file_location("export_web_models_release_test", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
export_web_models = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(export_web_models)


def args(release: str) -> argparse.Namespace:
    return argparse.Namespace(
        classifier_release=release,
        classifier=None,
        classifier_report=None,
        classifier_release_contract=None,
        classifier_output_name=None,
        route2_manifest=export_web_models.DEFAULT_ROUTE2,
        output_dir=Path("virtual-browser-model-output"),
    )


class BrowserModelReleaseExportTest(unittest.TestCase):
    def temp_path(self, suffix: str) -> Path:
        descriptor, name = tempfile.mkstemp(dir=ROOT, suffix=suffix)
        os.close(descriptor)
        path = Path(name)
        self.addCleanup(path.unlink, missing_ok=True)
        return path

    @staticmethod
    def release_metadata(release: str) -> dict[str, object]:
        shadow = release == "shadow-preview"
        return {
            "channel": release,
            "status": "shadow_diagnostic_only" if shadow else "active_legacy_default",
            "default_eligible": not shadow,
            "promotion_eligible": False,
            "promotion_status": (
                "failed_target_keep_non_promotable_shadow"
                if shadow
                else "legacy_active_not_requalified"
            ),
            "promotion_reason": "bound test reason",
            "source_model_sha256": "a" * 64,
            "model_card_schema_version": "durf-feedback-form-model-card-v1",
        }

    def test_shadow_release_only_verifies_existing_route2(self) -> None:
        with (
            patch.object(export_web_models, "parse_args", return_value=args("shadow-preview")),
            patch.object(
                export_web_models,
                "export_classifier",
                return_value=export_web_models.SHADOW_BROWSER_SHA256,
            ),
            patch.object(export_web_models, "export_route2") as export_route2,
            patch.object(
                export_web_models,
                "sha256",
                return_value=export_web_models.ROUTE2_BROWSER_SHA256,
            ) as sha256,
            patch.object(
                export_web_models,
                "browser_release_metadata",
                return_value=self.release_metadata("shadow-preview"),
            ),
            patch.object(export_web_models, "write_json", return_value="manifest-hash") as write_json,
        ):
            export_web_models.main()

        export_route2.assert_not_called()
        sha256.assert_called_once_with(
            Path("virtual-browser-model-output") / "route2-v5.json"
        )
        manifest_path, manifest = write_json.call_args.args
        self.assertEqual(
            manifest_path,
            Path("virtual-browser-model-output")
            / export_web_models.SHADOW_MANIFEST_FILENAME,
        )
        self.assertEqual(
            manifest["models"]["route2"]["sha256"],
            export_web_models.ROUTE2_BROWSER_SHA256,
        )
        self.assertEqual(manifest["schema_version"], "durf-browser-model-manifest-v2")
        self.assertFalse(manifest["release"]["promotion_eligible"])

    def test_production_release_keeps_route2_export(self) -> None:
        with (
            patch.object(export_web_models, "parse_args", return_value=args("production")),
            patch.object(
                export_web_models,
                "export_classifier",
                return_value="a" * 64,
            ),
            patch.object(
                export_web_models,
                "export_route2",
                return_value=export_web_models.ROUTE2_BROWSER_SHA256,
            ) as export_route2,
            patch.object(
                export_web_models,
                "browser_release_metadata",
                return_value=self.release_metadata("production"),
            ),
            patch.object(export_web_models, "write_json", return_value="manifest-hash"),
        ):
            export_web_models.main()

        export_route2.assert_called_once_with(
            export_web_models.DEFAULT_ROUTE2.resolve(),
            Path("virtual-browser-model-output") / "route2-v5.json",
        )

    def test_active_production_export_builds_only_current_sha_bound_model_card(self) -> None:
        output = self.temp_path(".feedback.json")
        digest = export_web_models.export_classifier(
            export_web_models.DEFAULT_CLASSIFIER,
            export_web_models.DEFAULT_CLASSIFIER_REPORT,
            output,
            release="production",
            release_contract_path=(
                export_web_models.DEFAULT_CLASSIFIER_RELEASE_CONTRACT
            ),
        )
        payload = json.loads(output.read_text(encoding="utf-8"))
        actual_digest = hashlib.sha256(output.read_bytes()).hexdigest()

        self.assertEqual(digest, actual_digest)
        card = payload["model_card"]
        self.assertEqual(
            card["source_model_sha256"],
            "3ce488daf88afb611fa5923ed21407ea344fbbb79c10fbc8ab3c39bc174d6e31",
        )
        self.assertEqual(card["release"]["status"], "active_legacy_default")
        self.assertTrue(card["release"]["default_eligible"])
        self.assertFalse(card["release"]["promotion_eligible"])
        self.assertEqual(card["claims"]["independent_current_player_accuracy"], None)
        self.assertTrue(
            all(
                report["source_model_sha256"] == card["source_model_sha256"]
                for report in card["reports"]
            )
        )
        self.assertEqual(card["evidence"][0]["accuracy"], 0.9166666666666666)

    def test_historical_reports_are_not_rebranded_as_active_release_evidence(self) -> None:
        contract = json.loads(
            export_web_models.DEFAULT_CLASSIFIER_RELEASE_CONTRACT.read_text(
                encoding="utf-8"
            )
        )
        referenced = {report["file"] for report in contract["reports"]}
        self.assertTrue(
            referenced.isdisjoint(
                {
                    "synthetic_test.report.json",
                    "assistant_curated_regression.report.json",
                    "human_feedback_form_holdout.evaluation.json",
                }
            )
        )

    def test_invalid_production_status_fails_before_joblib_deserialization(self) -> None:
        contract = json.loads(
            export_web_models.DEFAULT_CLASSIFIER_RELEASE_CONTRACT.read_text(
                encoding="utf-8"
            )
        )
        contract["release"]["promotion_eligible"] = True
        contract_path = self.temp_path(".release.json")
        output = self.temp_path(".feedback.json")
        contract_path.write_text(json.dumps(contract), encoding="utf-8")
        with patch.object(export_web_models.joblib, "load") as joblib_load:
            with self.assertRaisesRegex(ValueError, "status or promotion"):
                export_web_models.export_classifier(
                    export_web_models.DEFAULT_CLASSIFIER,
                    export_web_models.DEFAULT_CLASSIFIER_REPORT,
                    output,
                    release="production",
                    release_contract_path=contract_path,
                )
        joblib_load.assert_not_called()

    def test_report_source_sha_mismatch_fails_before_joblib_deserialization(self) -> None:
        model_sha = export_web_models.sha256(export_web_models.DEFAULT_CLASSIFIER)
        release = {
            "channel": "production",
            "status": "active_legacy_default",
            "default_eligible": True,
            "promotion_eligible": False,
            "promotion_status": "legacy_active_not_requalified",
            "promotion_reason": "No independent current-player evaluation is bound to this model.",
        }
        training = {"artifact_model_sha256": model_sha}
        evidence = {
            "model_sha256_before": "0" * 64,
            "model_sha256_after": "0" * 64,
        }
        training_path = self.temp_path(".training.json")
        evidence_path = self.temp_path(".evidence.json")
        contract_path = self.temp_path(".release.json")
        output = self.temp_path(".feedback.json")
        training_path.write_text(json.dumps(training), encoding="utf-8")
        evidence_path.write_text(json.dumps(evidence), encoding="utf-8")
        contract = {
            "schema_version": "durf-feedback-form-release-contract-v1",
            "model_sha256": model_sha,
            "release": release,
            "training": {"trained": True, "frozen": True, "report_id": "training"},
            "claims": {
                "independent_current_player_accuracy": None,
                "calibration_independently_validated": False,
            },
            "display_evidence_id": "evidence",
            "reports": [
                {
                    "id": "training",
                    "role": "training",
                    "file": training_path.name,
                    "sha256": export_web_models.sha256(training_path),
                    "source_model_sha256_fields": ["artifact_model_sha256"],
                },
                {
                    "id": "evidence",
                    "role": "diagnostic",
                    "file": evidence_path.name,
                    "sha256": export_web_models.sha256(evidence_path),
                    "source_model_sha256_fields": [
                        "model_sha256_before",
                        "model_sha256_after",
                    ],
                },
            ],
        }
        contract_path.write_text(json.dumps(contract), encoding="utf-8")
        with patch.object(export_web_models.joblib, "load") as joblib_load:
            with self.assertRaisesRegex(ValueError, "source model SHA-256 mismatch"):
                export_web_models.export_classifier(
                    export_web_models.DEFAULT_CLASSIFIER,
                    training_path,
                    output,
                    release="production",
                    release_contract_path=contract_path,
                )
        joblib_load.assert_not_called()

    def test_shadow_model_card_reads_the_bound_one_shot_receipt(self) -> None:
        output = self.temp_path(".feedback.json")
        with patch.object(
            export_web_models,
            "read_bytes_with_sha256",
            wraps=export_web_models.read_bytes_with_sha256,
        ) as read_bytes:
            export_web_models.export_classifier(
                export_web_models.DEFAULT_SHADOW_CLASSIFIER,
                export_web_models.DEFAULT_SHADOW_CLASSIFIER_REPORT,
                output,
                release="shadow-preview",
            )
        card = json.loads(output.read_text(encoding="utf-8"))["model_card"]
        evidence = card["evidence"][0]
        self.assertEqual(evidence["accuracy"], 0.8194444444444444)
        self.assertEqual(evidence["rows"], 72)
        self.assertEqual(evidence["requested_accuracy_target"], 0.87)
        self.assertFalse(evidence["target_passed"])
        self.assertTrue(
            all(
                report["source_model_sha256"] == card["source_model_sha256"]
                for report in card["reports"]
            )
        )
        read_paths = [call.args[0].resolve() for call in read_bytes.call_args_list]
        for source in (
            export_web_models.DEFAULT_SHADOW_CLASSIFIER,
            export_web_models.DEFAULT_SHADOW_CLASSIFIER_REPORT,
            export_web_models.DEFAULT_SHADOW_FROZEN_CONFIG,
            export_web_models.DEFAULT_SHADOW_CANDIDATE_MANIFEST,
            export_web_models.DEFAULT_SHADOW_DIAGNOSTIC_RECEIPT,
            export_web_models.DEFAULT_SHADOW_PREDICTOR,
        ):
            self.assertEqual(read_paths.count(source.resolve()), 1)

    def test_default_route2_export_preserves_pinned_browser_bytes(self) -> None:
        output = self.temp_path(".route2.json")
        with patch.object(
            export_web_models,
            "read_bytes_with_sha256",
            wraps=export_web_models.read_bytes_with_sha256,
        ) as read_bytes:
            digest = export_web_models.export_route2(
                export_web_models.DEFAULT_ROUTE2,
                output,
            )
        artifact = json.loads(output.read_text(encoding="utf-8"))
        self.assertEqual(digest, export_web_models.ROUTE2_BROWSER_SHA256)
        self.assertEqual(
            artifact["source"]["ensemble_manifest_sha256"],
            export_web_models.DEFAULT_ROUTE2_MANIFEST_SHA256,
        )
        self.assertEqual(
            artifact["source"]["ensemble_identity"],
            export_web_models.DEFAULT_ROUTE2_ENSEMBLE_IDENTITY,
        )
        read_paths = [call.args[0].resolve() for call in read_bytes.call_args_list]
        self.assertEqual(read_paths.count(export_web_models.DEFAULT_ROUTE2.resolve()), 1)
        self.assertEqual(len(read_paths), 11)
        self.assertEqual(len(set(read_paths)), 11)

    def test_default_route2_manifest_identity_fails_before_checkpoint_load(self) -> None:
        manifest, _ = export_web_models.read_json_with_sha256(
            export_web_models.DEFAULT_ROUTE2
        )
        cases = (
            (manifest, "0" * 64),
            (
                {
                    **manifest,
                    "manifest_sha256": "0" * 64,
                },
                export_web_models.DEFAULT_ROUTE2_MANIFEST_SHA256,
            ),
        )
        for supplied_manifest, supplied_hash in cases:
            with self.subTest(supplied_hash=supplied_hash):
                with (
                    patch.object(
                        export_web_models,
                        "read_json_with_sha256",
                        return_value=(supplied_manifest, supplied_hash),
                    ) as read_manifest,
                    patch.object(export_web_models, "read_bytes_with_sha256") as read_bytes,
                    patch.object(export_web_models.torch, "load") as torch_load,
                ):
                    with self.assertRaisesRegex(ValueError, "ensemble identity changed"):
                        export_web_models.export_route2(
                            export_web_models.DEFAULT_ROUTE2,
                            self.temp_path(".route2.json"),
                        )
                read_manifest.assert_called_once_with(
                    export_web_models.DEFAULT_ROUTE2.resolve()
                )
                read_bytes.assert_not_called()
                torch_load.assert_not_called()

    def test_route2_deserializes_the_same_checkpoint_bytes_after_hash_validation(self) -> None:
        features = ["a"]
        checkpoint_payloads = {
            f"fold_{fold}.pt": f"checkpoint-{fold}".encode("ascii")
            for fold in range(10)
        }
        manifest = {
            "schema_version": "route2-ensemble-v1",
            "manifest_sha256": "custom-test-identity",
            "features": features,
            "members": [
                {
                    "fold": fold,
                    "checkpoint": filename,
                    "checkpoint_sha256": hashlib.sha256(payload).hexdigest(),
                }
                for fold, (filename, payload) in enumerate(checkpoint_payloads.items())
            ],
        }
        checkpoint = {
            "features": features,
            "vocab": {"<unk>": 0},
            "config": {"vocab_size": 1, "n_features": 1},
            "state_dict": {
                "embedding.weight": export_web_models.torch.zeros((1, 1)),
                "fc1.weight": export_web_models.torch.zeros((1, 2)),
                "fc1.bias": export_web_models.torch.zeros(1),
                "fc2.weight": export_web_models.torch.zeros((1, 1)),
                "fc2.bias": export_web_models.torch.zeros(1),
            },
        }
        seen_bytes: list[bytes] = []

        def read_checkpoint(path: Path) -> tuple[bytes, str]:
            payload = checkpoint_payloads[path.name]
            return payload, hashlib.sha256(payload).hexdigest()

        def load_checkpoint(
            source: io.BytesIO,
            *,
            map_location: str,
            weights_only: bool,
        ) -> dict[str, object]:
            self.assertIsInstance(source, io.BytesIO)
            self.assertEqual(map_location, "cpu")
            self.assertTrue(weights_only)
            seen_bytes.append(source.read())
            return checkpoint

        with (
            patch.object(
                export_web_models,
                "read_json_with_sha256",
                return_value=(manifest, "1" * 64),
            ) as read_manifest,
            patch.object(
                export_web_models,
                "read_bytes_with_sha256",
                side_effect=read_checkpoint,
            ) as read_bytes,
            patch.object(
                export_web_models.torch,
                "load",
                side_effect=load_checkpoint,
            ) as torch_load,
        ):
            export_web_models.export_route2(
                self.temp_path(".custom-manifest.json"),
                self.temp_path(".route2.json"),
            )

        self.assertEqual(read_manifest.call_count, 1)
        self.assertEqual(read_bytes.call_count, 10)
        self.assertEqual(torch_load.call_count, 10)
        self.assertEqual(seen_bytes, list(checkpoint_payloads.values()))


if __name__ == "__main__":
    unittest.main()
