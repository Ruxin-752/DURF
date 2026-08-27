"""Freeze a completed Route 2 dev selection without opening corpus/test data.

This command binds every selected fold to the exact ``model.pt`` bytes.  It
only reads selection metadata, CV metadata, ensemble metadata, dev reports,
and checkpoints; it never accepts or opens a feedback corpus.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.evaluation_splits import canonical_sha256  # noqa: E402


DEFAULT_MODEL_DIR = (
    ROOT / "outputs" / "route2" / "paper_aligned_v5_seed137_selected"
)
EXPECTED_FOLDS = tuple(range(10))
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
FOLD_DIR_RE = re.compile(r"^fold_(\d+)$")


def _read_json(path: Path, *, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"missing {label}: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read {label}: {path}") from error
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must be a JSON object")
    return payload


def _manifest_hash(payload: dict[str, Any]) -> str:
    return canonical_sha256(
        {key: value for key, value in payload.items() if key != "manifest_sha256"}
    )


def _verify_manifest_hash(payload: dict[str, Any], *, label: str) -> str:
    supplied = payload.get("manifest_sha256")
    expected = _manifest_hash(payload)
    if supplied != expected:
        raise ValueError(f"{label} manifest hash mismatch")
    return expected


def _require_sha256(value: Any, *, label: str) -> str:
    text = str(value or "")
    if not SHA256_RE.fullmatch(text):
        raise ValueError(f"{label} must be a lowercase SHA256 digest")
    return text


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _fold_map(
    rows: Any,
    *,
    label: str,
    expected_folds: tuple[int, ...],
) -> dict[int, dict[str, Any]]:
    if not isinstance(rows, list):
        raise ValueError(f"{label} must be a list")
    output: dict[int, dict[str, Any]] = {}
    for position, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ValueError(f"{label}[{position}] must be an object")
        fold_value = row.get("fold")
        if isinstance(fold_value, bool) or not isinstance(fold_value, int):
            raise ValueError(f"{label}[{position}].fold must be an integer")
        if fold_value in output:
            raise ValueError(f"{label} repeats fold {fold_value}")
        output[fold_value] = row
    expected = set(expected_folds)
    if set(output) != expected:
        raise ValueError(
            f"{label} fold set must be exactly {sorted(expected)}; "
            f"got {sorted(output)}"
        )
    return output


def _validate_fold_directories(
    model_dir: Path, *, expected_folds: tuple[int, ...]
) -> None:
    numeric_dirs: dict[int, str] = {}
    for child in model_dir.iterdir():
        if not child.is_dir():
            continue
        match = FOLD_DIR_RE.fullmatch(child.name)
        if match is None:
            continue
        fold = int(match.group(1))
        if fold in numeric_dirs:
            raise ValueError(
                f"duplicate fold directories for fold {fold}: "
                f"{numeric_dirs[fold]!r}, {child.name!r}"
            )
        numeric_dirs[fold] = child.name
    expected_names = {fold: f"fold_{fold:02d}" for fold in expected_folds}
    if numeric_dirs != expected_names:
        raise ValueError(
            "fold directories must be the completed canonical 10-fold set; "
            f"expected={expected_names}, got={numeric_dirs}"
        )


def _checkpoint_path(model_dir: Path, relative: str) -> Path:
    candidate = (model_dir / relative).resolve(strict=True)
    try:
        candidate.relative_to(model_dir.resolve(strict=True))
    except ValueError as error:
        raise ValueError(f"checkpoint escapes model directory: {relative!r}") from error
    if not candidate.is_file():
        raise FileNotFoundError(f"checkpoint is not a regular file: {candidate}")
    return candidate


def _assert_missing_or_equal(
    mapping: dict[str, Any], key: str, expected: Any, *, label: str
) -> None:
    actual = mapping.get(key)
    if actual not in (None, "") and actual != expected:
        raise ValueError(
            f"{label}.{key} conflicts with frozen selection: "
            f"expected={expected!r}, actual={actual!r}"
        )


def _atomic_json_pair(
    first_path: Path,
    first_payload: dict[str, Any],
    second_path: Path,
    second_payload: dict[str, Any],
) -> None:
    """Stage both JSON documents, then replace their originals.

    If the second replacement fails, the first document is restored from its
    original bytes before the error is propagated.
    """

    originals = {
        first_path: first_path.read_bytes(),
        second_path: second_path.read_bytes(),
    }

    def stage(path: Path, payload: dict[str, Any] | None = None, raw: bytes | None = None) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, name = tempfile.mkstemp(
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
        )
        temporary = Path(name)
        try:
            with os.fdopen(descriptor, "wb") as handle:
                data = raw
                if data is None:
                    data = (
                        json.dumps(
                            payload,
                            ensure_ascii=False,
                            indent=2,
                            sort_keys=True,
                        )
                        + "\n"
                    ).encode("utf-8")
                handle.write(data)
                handle.flush()
                os.fsync(handle.fileno())
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
        return temporary

    first_temp = stage(first_path, first_payload)
    try:
        second_temp = stage(second_path, second_payload)
    except BaseException:
        first_temp.unlink(missing_ok=True)
        raise
    first_replaced = False
    try:
        first_temp.replace(first_path)
        first_replaced = True
        second_temp.replace(second_path)
    except BaseException:
        if first_replaced:
            restore = stage(first_path, raw=originals[first_path])
            restore.replace(first_path)
        raise
    finally:
        first_temp.unlink(missing_ok=True)
        second_temp.unlink(missing_ok=True)


def finalize_selection(
    model_dir: Path,
    *,
    expected_folds: tuple[int, ...] = EXPECTED_FOLDS,
) -> dict[str, Any]:
    """Validate and cryptographically bind a completed dev-only selection."""

    model_dir = model_dir.resolve(strict=True)
    if not model_dir.is_dir():
        raise NotADirectoryError(model_dir)
    receipt_path = model_dir / "test_evaluation_receipt.json"
    if receipt_path.exists():
        raise RuntimeError(
            "test_evaluation_receipt.json already exists; refusing to alter a "
            "selection after test access"
        )
    if (model_dir / "evaluation_report.json").exists():
        raise RuntimeError(
            "evaluation_report.json already exists without a receipt; refusing "
            "to assume that held-out test data was not accessed"
        )

    selection_path = model_dir / "selection_manifest.json"
    ensemble_path = model_dir / "ensemble_manifest.json"
    cv_path = model_dir / "cross_validation_manifest.json"
    selection = _read_json(selection_path, label="selection manifest")
    ensemble = _read_json(ensemble_path, label="ensemble manifest")
    cv_manifest = _read_json(cv_path, label="cross-validation manifest")

    old_selection_hash = _verify_manifest_hash(selection, label="selection")
    _verify_manifest_hash(ensemble, label="ensemble")
    cv_hash = _verify_manifest_hash(cv_manifest, label="cross-validation")
    corpus_hash = _require_sha256(
        selection.get("corpus_sha256"), label="selection.corpus_sha256"
    )
    registry = selection.get("candidate_registry")
    if not isinstance(registry, dict):
        raise ValueError("selection.candidate_registry must be an object")
    registry_hash = _require_sha256(
        selection.get("candidate_registry_sha256"),
        label="selection.candidate_registry_sha256",
    )
    if canonical_sha256(registry) != registry_hash:
        raise ValueError("candidate registry hash mismatch")

    if selection.get("test_accessed") is not False:
        raise ValueError("selection is not marked dev-only/test_accessed=false")
    if selection.get("partition") != "dev":
        raise ValueError("selection.partition must be 'dev'")
    if int(selection.get("test_examples_evaluated", -1)) != 0:
        raise ValueError("selection reports top-level test evaluation")
    if ensemble.get("test_accessed") is not False:
        raise ValueError("ensemble is not marked test_accessed=false")
    if selection.get("cross_validation_manifest_sha256") != cv_hash:
        raise ValueError("selection and cross-validation manifest hashes differ")
    if ensemble.get("cross_validation_manifest_sha256") != cv_hash:
        raise ValueError("ensemble and cross-validation manifest hashes differ")
    if ensemble.get("corpus_sha256") != corpus_hash:
        raise ValueError("ensemble and selection corpus hashes differ")
    if ensemble.get("selection_manifest_sha256") != old_selection_hash:
        raise ValueError("ensemble is not bound to the current selection manifest")

    registry_rows = registry.get("candidates")
    if not isinstance(registry_rows, list) or not registry_rows:
        raise ValueError("candidate registry contains no candidates")
    registry_candidates: dict[str, dict[str, Any]] = {}
    for position, row in enumerate(registry_rows):
        if not isinstance(row, dict) or not str(row.get("candidate_id") or ""):
            raise ValueError(f"candidate_registry.candidates[{position}] is invalid")
        candidate_id = str(row["candidate_id"])
        if candidate_id in registry_candidates:
            raise ValueError(f"candidate registry repeats {candidate_id!r}")
        registry_candidates[candidate_id] = row

    _validate_fold_directories(model_dir, expected_folds=expected_folds)
    cv_folds = _fold_map(
        cv_manifest.get("folds"), label="cross-validation folds", expected_folds=expected_folds
    )
    selection_folds = _fold_map(
        selection.get("folds"), label="selection folds", expected_folds=expected_folds
    )
    ensemble_members = _fold_map(
        ensemble.get("members"), label="ensemble members", expected_folds=expected_folds
    )
    if set(cv_folds) != set(selection_folds) or set(cv_folds) != set(ensemble_members):
        raise ValueError("CV, selection, and ensemble fold sets differ")

    winner_counts: Counter[str] = Counter()
    verified: list[dict[str, Any]] = []
    for fold in expected_folds:
        fold_row = selection_folds[fold]
        if fold_row.get("test_accessed") is not False:
            raise ValueError(f"fold {fold}: selection row is not test_accessed=false")
        if int(fold_row.get("test_examples_evaluated", -1)) != 0:
            raise ValueError(f"fold {fold}: selection row reports test evaluation")
        winner = fold_row.get("winner")
        if not isinstance(winner, dict):
            raise ValueError(f"fold {fold}: missing winner")
        candidate_id = str(winner.get("candidate_id") or "")
        if candidate_id not in registry_candidates:
            raise ValueError(f"fold {fold}: winner is absent from candidate registry")
        winner_counts[candidate_id] += 1

        candidate_rows = fold_row.get("candidates")
        if not isinstance(candidate_rows, list):
            raise ValueError(f"fold {fold}: candidates must be a list")
        fold_candidate_ids = [
            str(row.get("candidate_id") or "")
            for row in candidate_rows
            if isinstance(row, dict)
        ]
        if len(fold_candidate_ids) != len(candidate_rows):
            raise ValueError(f"fold {fold}: invalid candidate report")
        if len(set(fold_candidate_ids)) != len(fold_candidate_ids):
            raise ValueError(f"fold {fold}: duplicate candidate report")
        if set(fold_candidate_ids) != set(registry_candidates):
            raise ValueError(f"fold {fold}: candidate reports do not match registry")
        for candidate_row in candidate_rows:
            candidate_id_in_row = str(candidate_row["candidate_id"])
            registry_row = registry_candidates[candidate_id_in_row]
            inconsistent_config = {
                key: {
                    "expected": value,
                    "actual": candidate_row.get(key),
                }
                for key, value in registry_row.items()
                if key != "candidate_id" and candidate_row.get(key) != value
            }
            if inconsistent_config:
                raise ValueError(
                    f"fold {fold}: candidate {candidate_id_in_row!r} differs "
                    f"from registry: {inconsistent_config}"
                )

        expected_winner_config = {
            key: value
            for key, value in registry_candidates[candidate_id].items()
            if key != "candidate_id"
        }
        if expected_winner_config and winner.get("config") != expected_winner_config:
            raise ValueError(f"fold {fold}: winner config differs from registry")

        report_path = model_dir / f"fold_{fold:02d}" / "dev_selection_report.json"
        report = _read_json(report_path, label=f"fold {fold} dev report")
        if report.get("fold") != fold:
            raise ValueError(f"fold {fold}: dev report fold metadata differs")
        report_winner = report.get("winner")
        if not isinstance(report_winner, dict):
            raise ValueError(f"fold {fold}: dev report is missing its winner")
        if report_winner.get("candidate_id") != candidate_id:
            raise ValueError(f"fold {fold}: dev report winner differs")
        if expected_winner_config and report_winner.get("config") != expected_winner_config:
            raise ValueError(f"fold {fold}: dev report winner config differs")
        if report.get("test_accessed") is not False or int(
            report.get("test_examples_evaluated", -1)
        ) != 0:
            raise ValueError(f"fold {fold}: dev report indicates test access")

        relative = f"fold_{fold:02d}/model.pt"
        _assert_missing_or_equal(winner, "checkpoint", relative, label=f"fold {fold} winner")
        _assert_missing_or_equal(
            report_winner, "checkpoint", relative, label=f"fold {fold} dev winner"
        )
        member = ensemble_members[fold]
        _assert_missing_or_equal(
            member, "checkpoint", relative, label=f"fold {fold} ensemble member"
        )
        _assert_missing_or_equal(
            member,
            "selected_candidate",
            candidate_id,
            label=f"fold {fold} ensemble member",
        )
        if member.get("test_metrics") is not None:
            raise ValueError(f"fold {fold}: ensemble member already contains test metrics")

        checkpoint_path = _checkpoint_path(model_dir, relative)
        checkpoint_sha = _file_sha256(checkpoint_path)
        _assert_missing_or_equal(
            winner,
            "checkpoint_sha256",
            checkpoint_sha,
            label=f"fold {fold} winner",
        )
        _assert_missing_or_equal(
            report_winner,
            "checkpoint_sha256",
            checkpoint_sha,
            label=f"fold {fold} dev winner",
        )
        _assert_missing_or_equal(
            member,
            "checkpoint_sha256",
            checkpoint_sha,
            label=f"fold {fold} ensemble member",
        )

        checkpoint = torch.load(
            checkpoint_path, map_location="cpu", weights_only=True
        )
        if not isinstance(checkpoint, dict) or not isinstance(checkpoint.get("extra"), dict):
            raise ValueError(f"fold {fold}: checkpoint is missing extra metadata")
        extra = checkpoint["extra"]
        expected_extra = {
            "fold": fold,
            "selection_partition": "dev",
            "test_accessed": False,
            "selected_candidate": candidate_id,
            "candidate_registry_sha256": registry_hash,
            "cross_validation_manifest_sha256": cv_hash,
            "corpus_sha256": corpus_hash,
        }
        mismatches = {
            key: {"expected": value, "actual": extra.get(key)}
            for key, value in expected_extra.items()
            if extra.get(key) != value
        }
        if mismatches:
            raise ValueError(f"fold {fold}: checkpoint metadata mismatch: {mismatches}")
        if _file_sha256(checkpoint_path) != checkpoint_sha:
            raise RuntimeError(f"fold {fold}: checkpoint changed while being verified")

        winner["checkpoint"] = relative
        winner["checkpoint_sha256"] = checkpoint_sha
        member["checkpoint"] = relative
        member["checkpoint_sha256"] = checkpoint_sha
        member["selected_candidate"] = candidate_id
        verified.append(
            {
                "fold": fold,
                "candidate_id": candidate_id,
                "checkpoint": relative,
                "checkpoint_sha256": checkpoint_sha,
            }
        )

    expected_counts = dict(sorted(winner_counts.items()))
    if selection.get("winner_counts") != expected_counts:
        raise ValueError(
            "selection.winner_counts does not match per-fold winners: "
            f"expected={expected_counts}, actual={selection.get('winner_counts')}"
        )

    selection.pop("manifest_sha256", None)
    selection_hash = canonical_sha256(selection)
    selection["manifest_sha256"] = selection_hash
    ensemble["selection_manifest_sha256"] = selection_hash
    ensemble["members"] = [ensemble_members[fold] for fold in expected_folds]
    ensemble.pop("manifest_sha256", None)
    ensemble_hash = canonical_sha256(ensemble)
    ensemble["manifest_sha256"] = ensemble_hash

    # Recheck the receipt immediately before replacing either manifest.
    if receipt_path.exists():
        raise RuntimeError(
            "test receipt appeared during finalization; refusing to write manifests"
        )
    _atomic_json_pair(selection_path, selection, ensemble_path, ensemble)
    return {
        "model_dir": str(model_dir),
        "folds_verified": len(verified),
        "selection_manifest_sha256": selection_hash,
        "ensemble_manifest_sha256": ensemble_hash,
        "members": verified,
        "test_accessed": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    args = parser.parse_args()
    result = finalize_selection(args.model_dir)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
