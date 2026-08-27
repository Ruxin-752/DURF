"""Train and evaluate the frozen Route 2 trajectory-required diagnostic.

Three identical paper-shaped networks are compared:

* ``full`` receives language and trajectory features;
* ``text_only`` receives language with a zero trajectory vector;
* ``trajectory_only`` receives trajectory features with a constant text token.

The formal test path creates an exclusive receipt before opening ``test.json``.
It never reads or evaluates the project's seed-137 formal Route 2 test.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.generate_route2_trajectory_required_benchmark import (  # noqa: E402
    DEFAULT_OUTPUT_DIR,
    SCHEMA_VERSION,
)
from scripts.train_route2 import train  # noqa: E402
from src.neural_inference import build_vocab, collate_batch, save_checkpoint  # noqa: E402
from src.text_analysis import nn_tokenize, normalize_reference_vector  # noqa: E402


EVALUATOR_VERSION = "route2-trajectory-required-evaluator-v1"
INPUT_MODES = ("full", "text_only", "trajectory_only")


def _read_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return value


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _resolve_file(manifest_path: Path, entry: dict) -> Path:
    path = Path(entry["path"])
    if not path.is_absolute():
        path = manifest_path.parent / path
    if not path.is_file():
        raise FileNotFoundError(path)
    observed = _sha256(path)
    if observed != entry["sha256"]:
        raise ValueError(f"frozen benchmark hash mismatch: {path}")
    return path


def _create_receipt(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    descriptor = os.open(path, flags)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
    except Exception:
        try:
            os.close(descriptor)
        except OSError:
            pass
        raise


def _validate_payload(payload: dict, features: list[str], *, allowed_splits: set[str]) -> list[dict]:
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("unexpected trajectory-required benchmark schema")
    if list(payload.get("features") or []) != features:
        raise ValueError("benchmark feature schema does not match manifest")
    rows = payload.get("examples")
    if not isinstance(rows, list) or not rows:
        raise ValueError("benchmark payload has no examples")
    if {str(row.get("split")) for row in rows} - allowed_splits:
        raise ValueError("benchmark payload contains an unexpected split")
    expected_features = set(features)
    for index, row in enumerate(rows):
        weights = row.get("teacher_reward_weights")
        if not isinstance(weights, dict) or set(weights) != expected_features:
            raise ValueError(f"row {index} does not contain a complete reward vector")
        trajectory = row.get("route2_trajectory_features")
        if not isinstance(trajectory, dict) or not trajectory:
            raise ValueError(f"row {index} has no trajectory features")
    return rows


def _model_example(row: dict, features: list[str], *, input_mode: str) -> dict:
    counts = np.asarray(
        [float(row["route2_trajectory_features"].get(feature, 0.0)) for feature in features],
        dtype=np.float32,
    )
    counts = normalize_reference_vector(counts).astype(np.float32)
    tokens = nn_tokenize(row["text"]) if input_mode != "trajectory_only" else [""]
    return {
        "feedback_id": row["feedback_id"],
        "tokens": tokens,
        "feature_counts": counts.tolist(),
        "target_reward": [float(row["teacher_reward_weights"][feature]) for feature in features],
        "reward_config_id": row["reward_config_id"],
        "context_id": row["context_id"],
        "polarity": row["polarity"],
    }


def _dataset(rows: list[dict], features: list[str], vocab: dict, *, input_mode: str) -> dict:
    examples = [_model_example(row, features, input_mode=input_mode) for row in rows]
    train_indices = [index for index, row in enumerate(rows) if row["split"] == "train"]
    dev_indices = [index for index, row in enumerate(rows) if row["split"] == "dev"]
    if not train_indices or not dev_indices:
        raise ValueError("training payload must contain nonempty train and dev splits")
    return {
        "target_mode": "full_teacher_reward",
        "use_feature_counts": input_mode != "text_only",
        "vocab": vocab,
        "vocab_size": len(vocab),
        "n_features": len(features),
        "examples": examples,
        "split": {
            "train_indices": train_indices,
            "dev_indices": dev_indices,
            "test_indices": [],
        },
    }


def _predict(model, examples: list[dict], vocab: dict, *, use_feature_counts: bool) -> np.ndarray:
    batch = collate_batch(examples, vocab)
    if not use_feature_counts:
        batch["feature_counts"] = torch.zeros_like(batch["feature_counts"])
    with torch.no_grad():
        values = model(batch["tokens"], batch["offsets"], batch["feature_counts"])
    return values.detach().cpu().numpy().astype(np.float64)


def _target_catalog(rows: list[dict], features: list[str]) -> np.ndarray:
    unique: dict[tuple[float, ...], None] = {}
    for row in rows:
        vector = tuple(float(row["teacher_reward_weights"][feature]) for feature in features)
        unique.setdefault(vector, None)
    return np.asarray(list(unique), dtype=np.float64)


def _metrics(
    predictions: np.ndarray,
    examples: list[dict],
    *,
    varying_indices: list[int],
    target_catalog: np.ndarray,
) -> dict:
    targets = np.asarray([row["target_reward"] for row in examples], dtype=np.float64)
    errors = predictions - targets
    full_mse = float(np.mean(errors**2))
    varying_errors = errors[:, varying_indices]
    varying_mse = float(np.mean(varying_errors**2))
    target_varying = targets[:, varying_indices]
    pred_varying = predictions[:, varying_indices]
    numerator = np.sum(pred_varying * target_varying, axis=1)
    denominator = np.linalg.norm(pred_varying, axis=1) * np.linalg.norm(target_varying, axis=1)
    cosine = np.divide(
        numerator,
        denominator,
        out=np.zeros_like(numerator),
        where=denominator > 1e-12,
    )
    active = np.abs(targets) > 1e-12
    sign_correct = (np.sign(predictions) == np.sign(targets)) & active
    catalog = target_catalog[:, varying_indices]
    distances = np.sum((pred_varying[:, None, :] - catalog[None, :, :]) ** 2, axis=2)
    true_distances = np.sum((pred_varying - target_varying) ** 2, axis=1)
    minimum = distances.min(axis=1)
    nearest_ties = np.isclose(
        distances, minimum[:, None], atol=1e-10, rtol=1e-10
    ).sum(axis=1)
    nearest_correct = (
        np.isclose(true_distances, minimum, atol=1e-10, rtol=1e-10)
        & (nearest_ties == 1)
    )
    return {
        "full_vector_mse": full_mse,
        "varying_dimension_mse": varying_mse,
        "mean_cosine_similarity": float(np.mean(cosine)),
        "active_sign_accuracy": float(sign_correct.sum() / active.sum()),
        "nearest_target_vector_accuracy": float(np.mean(nearest_correct)),
        "nearest_target_vector_correct": int(nearest_correct.sum()),
        "nearest_target_vector_ties": int(np.sum(nearest_ties > 1)),
        "examples": len(examples),
        "varying_dimensions": len(varying_indices),
        "active_target_count": int(active.sum()),
    }


def _aggregate(rows: list[dict]) -> dict:
    metric_names = (
        "full_vector_mse",
        "varying_dimension_mse",
        "mean_cosine_similarity",
        "active_sign_accuracy",
        "nearest_target_vector_accuracy",
    )
    result = {}
    for metric in metric_names:
        values = np.asarray([row["metrics"][metric] for row in rows], dtype=np.float64)
        result[metric] = {
            "mean": float(values.mean()),
            "std": float(values.std()),
            "values": values.tolist(),
        }
    result["examples_per_seed"] = rows[0]["metrics"]["examples"]
    result["seeds"] = [row["seed"] for row in rows]
    return result


def _relative_gain(baseline: float, full: float) -> float:
    return float((baseline - full) / baseline) if baseline > 0 else 0.0


def _validity_gate(aggregates: dict, manifest: dict) -> dict:
    thresholds = manifest["validity_gate"]
    full = aggregates["full"]
    text = aggregates["text_only"]
    trajectory = aggregates["trajectory_only"]
    full_mse = full["varying_dimension_mse"]["mean"]
    text_mse = text["varying_dimension_mse"]["mean"]
    trajectory_mse = trajectory["varying_dimension_mse"]["mean"]
    text_gain = _relative_gain(text_mse, full_mse)
    trajectory_gain = _relative_gain(trajectory_mse, full_mse)
    full_config = full["nearest_target_vector_accuracy"]["mean"]
    text_config = text["nearest_target_vector_accuracy"]["mean"]
    trajectory_config = trajectory["nearest_target_vector_accuracy"]["mean"]
    checks = {
        "full_relative_gain_over_text_only": (
            text_gain >= thresholds["minimum_full_relative_gain_over_text_only"]
        ),
        "full_relative_gain_over_trajectory_only": (
            trajectory_gain >= thresholds["minimum_full_relative_gain_over_trajectory_only"]
        ),
        "full_nearest_target_accuracy": (
            full_config >= thresholds["minimum_full_nearest_config_accuracy"]
        ),
        "full_active_sign_accuracy": (
            full["active_sign_accuracy"]["mean"]
            >= thresholds["minimum_full_active_sign_accuracy"]
        ),
        "config_gain_over_text_only": (
            full_config - text_config
            >= thresholds["minimum_config_accuracy_gain_over_each_ablation"]
        ),
        "config_gain_over_trajectory_only": (
            full_config - trajectory_config
            >= thresholds["minimum_config_accuracy_gain_over_each_ablation"]
        ),
    }
    return {
        "status": "passed" if all(checks.values()) else "failed",
        "checks": checks,
        "thresholds": thresholds,
        "full_relative_mse_gain_over_text_only": text_gain,
        "full_relative_mse_gain_over_trajectory_only": trajectory_gain,
        "nearest_target_accuracy_gain_over_text_only": full_config - text_config,
        "nearest_target_accuracy_gain_over_trajectory_only": full_config - trajectory_config,
        "claim_allowed": all(checks.values()),
        "claim_scope": "synthetic two-input identifiability only",
    }


def run_evaluation(
    manifest_path: str | Path,
    *,
    partition: str = "dev",
    output_dir: str | Path | None = None,
) -> dict:
    if partition not in {"dev", "test"}:
        raise ValueError("partition must be dev or test")
    manifest_path = Path(manifest_path).resolve()
    manifest = _read_json(manifest_path)
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("unexpected benchmark manifest schema")
    output_dir = Path(output_dir).resolve() if output_dir else manifest_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / (
        "evaluation_report.json" if partition == "test" else "dev_evaluation_report.json"
    )
    receipt_path = output_dir / "test_evaluation_receipt.json"
    if report_path.exists():
        raise FileExistsError(f"evaluation report already exists: {report_path}")
    if partition == "test" and receipt_path.exists():
        raise FileExistsError("formal trajectory-required test has already been opened")

    train_dev_path = _resolve_file(manifest_path, manifest["files"]["train_dev"])
    train_dev_payload = _read_json(train_dev_path)
    features = list(train_dev_payload.get("features") or [])
    train_dev_rows = _validate_payload(
        train_dev_payload, features, allowed_splits={"train", "dev"}
    )
    train_token_lists = [
        nn_tokenize(row["text"]) for row in train_dev_rows if row["split"] == "train"
    ]
    vocab = build_vocab(train_token_lists, min_freq=1)
    varying = list(manifest["varying_features"])
    varying_indices = [features.index(feature) for feature in varying]
    dimension_weights = [1.0 if feature in set(varying) else 0.0 for feature in features]
    early_stop_mask = [feature in set(varying) for feature in features]
    config = manifest["training_config"]

    trained: dict[str, list[dict]] = {mode: [] for mode in INPUT_MODES}
    for input_mode in INPUT_MODES:
        dataset = _dataset(train_dev_rows, features, vocab, input_mode=input_mode)
        for seed in config["seeds"]:
            result = train(
                dataset,
                epochs=int(config["epochs"]),
                lr=float(config["learning_rate"]),
                weight_decay=float(config["weight_decay"]),
                batch_size=int(config["batch_size"]),
                patience=int(config["patience"]),
                optimizer_name=str(config["optimizer"]),
                use_feature_counts=input_mode != "text_only",
                seed=int(seed),
                evaluate_test=False,
                dimension_weights=dimension_weights,
                early_stop_dimension_mask=early_stop_mask,
            )
            checkpoint = output_dir / "models" / f"{input_mode}_seed{seed}" / "model.pt"
            save_checkpoint(
                checkpoint,
                result["model"],
                vocab,
                features,
                use_feature_counts=input_mode != "text_only",
                extra={
                    "benchmark_schema": SCHEMA_VERSION,
                    "input_mode": input_mode,
                    "seed": int(seed),
                    "best_dev_loss": float(result["best_val_loss"]),
                },
            )
            trained[input_mode].append(
                {
                    "seed": int(seed),
                    "model": result["model"],
                    "best_dev_loss": float(result["best_val_loss"]),
                    "best_epoch": int(result["best_epoch"]),
                    "epochs_run": int(result["epochs_run"]),
                    "checkpoint": str(checkpoint),
                    "checkpoint_sha256": _sha256(checkpoint),
                }
            )

    if partition == "dev":
        evaluation_rows = [row for row in train_dev_rows if row["split"] == "dev"]
        test_sha256 = None
    else:
        _create_receipt(
            receipt_path,
            {
                "schema_version": 1,
                "status": "test_open_started",
                "benchmark_manifest_sha256": _sha256(manifest_path),
                "train_dev_sha256": _sha256(train_dev_path),
                "evaluator_version": EVALUATOR_VERSION,
            },
        )
        test_path = _resolve_file(manifest_path, manifest["files"]["test"])
        test_payload = _read_json(test_path)
        evaluation_rows = _validate_payload(test_payload, features, allowed_splits={"test"})
        test_sha256 = _sha256(test_path)

    target_catalog = _target_catalog(train_dev_rows, features)
    per_system: dict[str, list[dict]] = {}
    for input_mode in INPUT_MODES:
        examples = [
            _model_example(row, features, input_mode=input_mode)
            for row in evaluation_rows
        ]
        seed_rows = []
        for trained_row in trained[input_mode]:
            predictions = _predict(
                trained_row["model"],
                examples,
                vocab,
                use_feature_counts=input_mode != "text_only",
            )
            seed_rows.append(
                {
                    "seed": trained_row["seed"],
                    "best_dev_loss": trained_row["best_dev_loss"],
                    "best_epoch": trained_row["best_epoch"],
                    "epochs_run": trained_row["epochs_run"],
                    "checkpoint": trained_row["checkpoint"],
                    "checkpoint_sha256": trained_row["checkpoint_sha256"],
                    "metrics": _metrics(
                        predictions,
                        examples,
                        varying_indices=varying_indices,
                        target_catalog=target_catalog,
                    ),
                }
            )
        per_system[input_mode] = seed_rows
    aggregates = {mode: _aggregate(rows) for mode, rows in per_system.items()}
    gate = _validity_gate(aggregates, manifest)
    report = {
        "schema_version": 1,
        "evaluator_version": EVALUATOR_VERSION,
        "partition": partition,
        "claim_scope": manifest["claim_scope"],
        "benchmark_manifest": str(manifest_path),
        "benchmark_manifest_sha256": _sha256(manifest_path),
        "train_dev_sha256": _sha256(train_dev_path),
        "test_sha256": test_sha256,
        "formal_seed137_test_touched": False,
        "examples": len(evaluation_rows),
        "feature_count": len(features),
        "varying_feature_count": len(varying),
        "unique_target_vectors": int(len(target_catalog)),
        "training_config": config,
        "systems": per_system,
        "aggregate": aggregates,
        "validity_gate": gate,
    }
    _write_json(report_path, report)
    report["report_path"] = str(report_path)
    if partition == "test":
        receipt = _read_json(receipt_path)
        receipt.update(
            {
                "status": "completed",
                "test_sha256": test_sha256,
                "evaluation_report": str(report_path),
                "evaluation_report_sha256": _sha256(report_path),
                "validity_gate_status": gate["status"],
            }
        )
        temporary = receipt_path.with_suffix(".json.tmp")
        _write_json(temporary, receipt)
        os.replace(temporary, receipt_path)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest", type=Path, default=DEFAULT_OUTPUT_DIR / "benchmark_manifest.json"
    )
    parser.add_argument("--partition", choices=("dev", "test"), default="dev")
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    report = run_evaluation(
        args.manifest,
        partition=args.partition,
        output_dir=args.output_dir,
    )
    print(f"Partition: {report['partition']}")
    for mode in INPUT_MODES:
        aggregate = report["aggregate"][mode]
        print(
            f"{mode}: varying MSE={aggregate['varying_dimension_mse']['mean']:.6f}, "
            f"nearest={aggregate['nearest_target_vector_accuracy']['mean']:.2%}, "
            f"sign={aggregate['active_sign_accuracy']['mean']:.2%}"
        )
    gate = report["validity_gate"]
    print(
        "Full gain over text-only: "
        f"{gate['full_relative_mse_gain_over_text_only']:.2%}"
    )
    print(f"Validity gate: {gate['status']}")
    print(f"Report: {report['report_path']}")
    return 0 if gate["status"] == "passed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
