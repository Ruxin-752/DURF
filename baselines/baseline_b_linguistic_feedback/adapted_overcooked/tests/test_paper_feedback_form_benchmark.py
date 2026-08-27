from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
TEST_TMP = ROOT / "outputs" / "_paper_feedback_form_test_tmp"
sys.path.insert(0, str(ROOT))

from scripts.evaluate_frozen_paper_feedback_form_test import evaluate_once  # noqa: E402
from scripts.prepare_paper_feedback_form_benchmark import (  # noqa: E402
    REFERENCE_TO_FEEDBACK,
    prepare_benchmark,
)
from scripts.train_feedback_form_classifier import (  # noqa: E402
    bind_frozen_benchmark_without_loading_test,
    load_training_rows,
    train_feedback_form_classifier,
)
from src.feature_schema import write_json  # noqa: E402
from src.feedback_form_classifier import predict_feedback_form  # noqa: E402


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _source_rows(repeats: int = 4) -> list[dict]:
    rows = []
    references = (*REFERENCE_TO_FEEDBACK, "other")
    for split_index, split in enumerate(("train", "dev", "test")):
        for reference_index, reference_type in enumerate(references):
            for repeat in range(repeats):
                source_index = split_index * 1000 + reference_index * 100 + repeat
                rows.append(
                    {
                        "feedback_id": f"paper_{source_index}",
                        "group_id": f"paper_task_uuid:task_{split}_{repeat}",
                        "paper_task_uuid": f"task_{split}_{repeat}",
                        "paper_source_index": source_index,
                        "reference_type": reference_type,
                        "split": split,
                        "text": f"{split} {reference_type} human phrase {repeat}",
                    }
                )
    return rows


def _write_manifest(directory: Path, converted: dict, report: dict) -> tuple[Path, dict]:
    paths = {
        "train": directory / "paper_train.json",
        "dev": directory / "paper_dev.json",
        "frozen_test": directory / "paper_test.json",
    }
    for name, split in (("train", "train"), ("dev", "dev"), ("frozen_test", "test")):
        write_json(paths[name], converted[split])
    manifest = {
        **report,
        "outputs": {
            name: {"path": str(path), "sha256": _sha256(path)}
            for name, path in paths.items()
        },
    }
    manifest_path = directory / "manifest.json"
    write_json(manifest_path, manifest)
    return manifest_path, paths


class PaperFeedbackFormBenchmarkTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        TEST_TMP.mkdir(parents=True, exist_ok=True)

    def test_main_experiment_mapping_and_split_boundaries(self) -> None:
        converted, report = prepare_benchmark(_source_rows())
        self.assertEqual(
            REFERENCE_TO_FEEDBACK["action_behavioral"], "evaluative"
        )
        self.assertTrue(
            all(row["reference_type"] != "other" for rows in converted.values() for row in rows)
        )
        self.assertEqual(report["group_overlap_count"], 0)
        self.assertEqual(report["normalized_text_overlap_count"], 0)
        self.assertTrue(report["frozen_test_membership"]["normalized_text_sha256"])
        self.assertEqual(
            {row["expected_feedback_type"] for row in converted["test"]},
            {"evaluative", "imperative", "descriptive"},
        )

    def test_training_binds_hashes_without_loading_test(self) -> None:
        with tempfile.TemporaryDirectory(dir=TEST_TMP) as temporary:
            directory = Path(temporary)
            converted, benchmark_report = prepare_benchmark(_source_rows())
            manifest_path, paths = _write_manifest(
                directory, converted, benchmark_report
            )
            synthetic = directory / "synthetic.json"
            human_train = directory / "human_train.json"
            human_dev = directory / "human_dev.json"
            write_json(synthetic, [])
            write_json(human_train, [])
            write_json(human_dev, [])
            train, dev, audit = load_training_rows(
                synthetic,
                human_train,
                human_dev,
                paper_human_train_path=paths["train"],
                paper_human_dev_path=paths["dev"],
            )
            behavior_rows = [
                row
                for row in train
                if "action_behavioral" in row["normalized_text"]
            ]
            self.assertTrue(behavior_rows)
            self.assertTrue(all(row["label"] == "evaluative" for row in behavior_rows))
            binding = bind_frozen_benchmark_without_loading_test(
                manifest_path, paths["train"], paths["dev"], train, dev
            )
            self.assertFalse(binding["frozen_test_loaded"])
            self.assertEqual(audit["test_examples_evaluated_during_selection"], 0)

    def test_calibrated_artifact_and_one_time_receipt(self) -> None:
        from joblib import dump

        with tempfile.TemporaryDirectory(dir=TEST_TMP) as temporary:
            directory = Path(temporary)
            converted, benchmark_report = prepare_benchmark(_source_rows(repeats=8))
            manifest_path, paths = _write_manifest(
                directory, converted, benchmark_report
            )
            synthetic = directory / "synthetic.json"
            human_train = directory / "human_train.json"
            human_dev = directory / "human_dev.json"
            write_json(synthetic, [])
            write_json(human_train, [])
            write_json(human_dev, [])
            train, dev, _audit = load_training_rows(
                synthetic,
                human_train,
                human_dev,
                paper_human_train_path=paths["train"],
                paper_human_dev_path=paths["dev"],
            )
            binding = bind_frozen_benchmark_without_loading_test(
                manifest_path, paths["train"], paths["dev"], train, dev
            )
            artifact, training_report = train_feedback_form_classifier(
                train, dev, min_df=1
            )
            artifact["frozen_benchmark_binding"] = binding
            artifact["manifest"] = {
                "frozen_test_loaded": False,
                "frozen_test_sha256_bound_not_loaded": binding[
                    "frozen_test_sha256_bound_not_opened"
                ],
                "paper_benchmark_manifest_sha256": _sha256(manifest_path),
            }
            model_path = directory / "model.joblib"
            dump(artifact, model_path)
            training_report["artifact_manifest"] = dict(artifact["manifest"])
            training_report["artifact_model_sha256"] = _sha256(model_path)
            training_report_path = directory / "model.report.json"
            write_json(training_report_path, training_report)

            prediction = predict_feedback_form(
                "dev trajectory human phrase 1", model_path=model_path
            )
            self.assertTrue(prediction["calibrated"])
            self.assertAlmostEqual(sum(prediction["probabilities"].values()), 1.0)

            report_path = directory / "frozen.report.json"
            receipt_path = directory / "frozen.receipt.json"
            result = evaluate_once(
                model_path=model_path,
                training_report_path=training_report_path,
                manifest_path=manifest_path,
                test_path=paths["frozen_test"],
                report_path=report_path,
                receipt_path=receipt_path,
            )
            self.assertTrue(receipt_path.exists())
            self.assertTrue(report_path.exists())
            self.assertFalse(result["independent_human_speech_form_gold"])
            with self.assertRaises(FileExistsError):
                evaluate_once(
                    model_path=model_path,
                    training_report_path=training_report_path,
                    manifest_path=manifest_path,
                    test_path=paths["frozen_test"],
                    report_path=report_path,
                    receipt_path=receipt_path,
                )


if __name__ == "__main__":
    unittest.main()
