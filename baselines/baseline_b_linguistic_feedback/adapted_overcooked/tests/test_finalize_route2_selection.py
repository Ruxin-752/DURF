from __future__ import annotations

import json
import shutil
import sys
import unittest
import uuid
from contextlib import contextmanager
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
TEST_TMP_PARENT = ROOT / "outputs" / "_finalize_unit_tests"
TEST_TMP_PARENT.mkdir(parents=True, exist_ok=True)

from scripts.finalize_route2_selection import finalize_selection  # noqa: E402
from src.evaluation_splits import canonical_sha256  # noqa: E402


@contextmanager
def _workspace_test_dir():
    path = TEST_TMP_PARENT / f"case_{uuid.uuid4().hex}"
    path.mkdir()
    try:
        yield path
    finally:
        shutil.rmtree(path)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _with_manifest_hash(payload: dict) -> dict:
    payload = dict(payload)
    payload["manifest_sha256"] = canonical_sha256(payload)
    return payload


def _make_completed_selection(model_dir: Path) -> None:
    corpus_sha = "a" * 64
    registry = {"candidates": [{"candidate_id": "paper_candidate"}]}
    registry_sha = canonical_sha256(registry)
    cv = _with_manifest_hash({"folds": [{"fold": fold} for fold in range(10)]})

    selection_folds = []
    members = []
    for fold in range(10):
        fold_dir = model_dir / f"fold_{fold:02d}"
        fold_dir.mkdir(parents=True)
        winner = {"candidate_id": "paper_candidate"}
        fold_report = {
            "fold": fold,
            "test_accessed": False,
            "test_examples_evaluated": 0,
            "candidates": [{"candidate_id": "paper_candidate"}],
            "winner": dict(winner),
        }
        selection_folds.append(dict(fold_report))
        _write_json(fold_dir / "dev_selection_report.json", fold_report)
        torch.save(
            {
                "extra": {
                    "fold": fold,
                    "selection_partition": "dev",
                    "test_accessed": False,
                    "selected_candidate": "paper_candidate",
                    "candidate_registry_sha256": registry_sha,
                    "cross_validation_manifest_sha256": cv["manifest_sha256"],
                    "corpus_sha256": corpus_sha,
                }
            },
            fold_dir / "model.pt",
        )
        members.append(
            {
                "fold": fold,
                "checkpoint": f"fold_{fold:02d}/model.pt",
                "selected_candidate": "paper_candidate",
                "test_metrics": None,
            }
        )

    selection = _with_manifest_hash(
        {
            "test_accessed": False,
            "partition": "dev",
            "test_examples_evaluated": 0,
            "corpus_sha256": corpus_sha,
            "cross_validation_manifest_sha256": cv["manifest_sha256"],
            "candidate_registry": registry,
            "candidate_registry_sha256": registry_sha,
            "winner_counts": {"paper_candidate": 10},
            "folds": selection_folds,
        }
    )
    ensemble = _with_manifest_hash(
        {
            "test_accessed": False,
            "corpus_sha256": corpus_sha,
            "cross_validation_manifest_sha256": cv["manifest_sha256"],
            "selection_manifest_sha256": selection["manifest_sha256"],
            "members": members,
        }
    )
    _write_json(model_dir / "cross_validation_manifest.json", cv)
    _write_json(model_dir / "selection_manifest.json", selection)
    _write_json(model_dir / "ensemble_manifest.json", ensemble)


class FinalizeRoute2SelectionTests(unittest.TestCase):
    def test_binds_all_checkpoints_and_is_idempotent(self) -> None:
        with _workspace_test_dir() as model_dir:
            _make_completed_selection(model_dir)
            first = finalize_selection(model_dir)
            second = finalize_selection(model_dir)

            self.assertEqual(first["folds_verified"], 10)
            self.assertEqual(
                first["selection_manifest_sha256"],
                second["selection_manifest_sha256"],
            )
            selection = json.loads(
                (model_dir / "selection_manifest.json").read_text(encoding="utf-8")
            )
            ensemble = json.loads(
                (model_dir / "ensemble_manifest.json").read_text(encoding="utf-8")
            )
            self.assertEqual(
                selection["manifest_sha256"],
                canonical_sha256(
                    {
                        key: value
                        for key, value in selection.items()
                        if key != "manifest_sha256"
                    }
                ),
            )
            for fold, row in enumerate(selection["folds"]):
                self.assertEqual(row["winner"]["checkpoint"], f"fold_{fold:02d}/model.pt")
                self.assertEqual(len(row["winner"]["checkpoint_sha256"]), 64)
                self.assertEqual(
                    ensemble["members"][fold]["checkpoint_sha256"],
                    row["winner"]["checkpoint_sha256"],
                )
            self.assertEqual(
                ensemble["selection_manifest_sha256"], selection["manifest_sha256"]
            )

    def test_rejects_checkpoint_metadata_mismatch(self) -> None:
        with _workspace_test_dir() as model_dir:
            _make_completed_selection(model_dir)
            checkpoint_path = model_dir / "fold_04" / "model.pt"
            checkpoint = torch.load(
                checkpoint_path, map_location="cpu", weights_only=False
            )
            checkpoint["extra"]["selected_candidate"] = "tampered"
            torch.save(checkpoint, checkpoint_path)
            with self.assertRaisesRegex(ValueError, "checkpoint metadata mismatch"):
                finalize_selection(model_dir)

    def test_rejects_duplicate_and_missing_fold(self) -> None:
        with _workspace_test_dir() as model_dir:
            _make_completed_selection(model_dir)
            selection_path = model_dir / "selection_manifest.json"
            ensemble_path = model_dir / "ensemble_manifest.json"
            selection = json.loads(selection_path.read_text(encoding="utf-8"))
            selection["folds"][-1] = dict(selection["folds"][0])
            selection.pop("manifest_sha256")
            selection["manifest_sha256"] = canonical_sha256(selection)
            ensemble = json.loads(ensemble_path.read_text(encoding="utf-8"))
            ensemble["selection_manifest_sha256"] = selection["manifest_sha256"]
            ensemble.pop("manifest_sha256")
            ensemble["manifest_sha256"] = canonical_sha256(ensemble)
            _write_json(selection_path, selection)
            _write_json(ensemble_path, ensemble)

            with self.assertRaisesRegex(ValueError, "repeats fold 0"):
                finalize_selection(model_dir)

    def test_rejects_existing_test_receipt_without_writing(self) -> None:
        with _workspace_test_dir() as model_dir:
            _make_completed_selection(model_dir)
            selection_path = model_dir / "selection_manifest.json"
            before = selection_path.read_bytes()
            _write_json(model_dir / "test_evaluation_receipt.json", {"test_accessed": True})
            with self.assertRaisesRegex(RuntimeError, "already exists"):
                finalize_selection(model_dir)
            self.assertEqual(selection_path.read_bytes(), before)


if __name__ == "__main__":
    unittest.main()
