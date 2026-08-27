"""Select one Route 2 model per fold using only that fold's train/dev rows.

The selection view physically omits test examples.  Each chosen checkpoint is
frozen before a separate evaluator may open its corresponding test partition
once.  Cooking behavior is a tie-break only because each dev fold has few
preference-sensitive cooking units.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections import Counter
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.evaluate_route2_paper_crossval import (  # noqa: E402
    _evaluation_suite,
    _preference_sensitive_groups,
    _reward_catalog,
)
from scripts.prepare_route2_dataset import build_dataset  # noqa: E402
from scripts.train_route2 import train  # noqa: E402
from scripts.train_route2_paper_crossval import _fold_dimension_weights  # noqa: E402
from src.evaluation_splits import canonical_sha256  # noqa: E402
from src.feature_schema import load_features, write_json  # noqa: E402
from src.neural_inference import (  # noqa: E402
    ENSEMBLE_SCHEMA_VERSION,
    build_vocab,
    collate_batch,
    save_checkpoint,
)
from src.paper_cross_validation import make_paper_cross_validation_manifest  # noqa: E402
from src.probe_evaluator import DEFAULT_PROBE_STATES_PATH, load_probe_states  # noqa: E402


DEFAULT_FEEDBACK = ROOT / "data" / "route2_teacher_feedback.paper_v5.synthetic.json"
DEFAULT_OUTPUT = ROOT / "outputs" / "route2" / "paper_aligned_v5_seed137_selected"
PREDECLARED_CANDIDATES = (
    {
        "candidate_id": "sgd_paper_accumulated_mse",
        "optimizer": "sgd",
        "lr": 0.005,
        "varying_dimension_weight": 1.0,
        "batch_reduction": "sum_examples",
    },
    {
        "candidate_id": "sgd_batch_mean_mse",
        "optimizer": "sgd",
        "lr": 0.005,
        "varying_dimension_weight": 1.0,
        "batch_reduction": "mean",
    },
    {
        "candidate_id": "adam_fixed_budget_mse",
        "optimizer": "adam",
        "lr": 0.01,
        "varying_dimension_weight": 1.0,
        "batch_reduction": "mean",
    },
    {
        "candidate_id": "sgd_paper_accumulated_varying_weight_2",
        "optimizer": "sgd",
        "lr": 0.005,
        "varying_dimension_weight": 2.0,
        "batch_reduction": "sum_examples",
    },
)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _selection_view(dataset: dict, fold: dict, *, min_freq: int) -> dict:
    """Return a physically cropped train+dev dataset with no test row."""

    train_examples = [dataset["examples"][index] for index in fold["train_indices"]]
    dev_examples = [dataset["examples"][index] for index in fold["dev_indices"]]
    selected = dict(dataset)
    selected["examples"] = [*train_examples, *dev_examples]
    selected["split"] = {
        "train_indices": list(range(len(train_examples))),
        "dev_indices": list(
            range(len(train_examples), len(train_examples) + len(dev_examples))
        ),
        "test_indices": [],
    }
    selected["folds"] = []
    selected["vocab"] = build_vocab(
        [example["tokens"] for example in train_examples], min_freq=min_freq
    )
    selected["vocab_size"] = len(selected["vocab"])
    return selected


def _predict(model, examples: list[dict], vocab: dict) -> tuple[torch.Tensor, torch.Tensor]:
    batch = collate_batch(examples, vocab)
    with torch.no_grad():
        predictions = model(
            batch["tokens"], batch["offsets"], batch["feature_counts"]
        )
    return predictions, batch["targets"]


def _metric_totals(suite: dict) -> dict:
    varying = suite["varying_dimension_reward_metrics"]
    nearest = suite["nearest_reward_config"]
    sensitive = suite["behavior_metrics"]["preference_sensitive"]
    cooking = suite["behavior_metrics"]["preference_sensitive_empty_hand_cooking"]
    return {
        "varying_squared_error_sum": (
            float(varying["mse"])
            * int(varying["examples"])
            * int(varying["dimensions"])
        ),
        "varying_values": int(varying["examples"]) * int(varying["dimensions"]),
        "nearest_correct": int(nearest["correct"]),
        "nearest_total": int(nearest["total"]),
        "sensitive_correct": int(sensitive["per_reward_context_majority"]["correct"]),
        "sensitive_total": int(sensitive["per_reward_context_majority"]["total"]),
        "cooking_correct": int(cooking["per_reward_context_majority"]["correct"]),
        "cooking_total": int(cooking["per_reward_context_majority"]["total"]),
    }


def _rates(totals: dict) -> dict:
    return {
        "varying_dimension_mse": (
            totals["varying_squared_error_sum"] / totals["varying_values"]
        ),
        "nearest_reward_config_accuracy": (
            totals["nearest_correct"] / totals["nearest_total"]
        ),
        "preference_sensitive_behavior_accuracy": (
            totals["sensitive_correct"] / totals["sensitive_total"]
            if totals["sensitive_total"]
            else 0.0
        ),
        "preference_sensitive_cooking_accuracy": (
            totals["cooking_correct"] / totals["cooking_total"]
            if totals["cooking_total"]
            else 0.0
        ),
        "dev_examples": totals["nearest_total"],
        "cooking_reward_contexts": totals["cooking_total"],
    }


def _sum_totals(rows: list[dict]) -> dict:
    return {key: sum(row[key] for row in rows) for key in rows[0]}


def _choose_fold_winner(candidate_reports: list[dict]) -> tuple[dict, dict]:
    valid = [
        report
        for report in candidate_reports
        if math.isfinite(report["metrics"]["varying_dimension_mse"])
    ]
    if not valid:
        raise RuntimeError("every predeclared candidate produced non-finite dev loss")
    best_mse = min(report["metrics"]["varying_dimension_mse"] for report in valid)
    tolerance = max(1e-4, 0.01 * best_mse)
    shortlist = [
        report
        for report in valid
        if report["metrics"]["varying_dimension_mse"] <= best_mse + tolerance
    ]
    registry_priority = {
        candidate["candidate_id"]: index
        for index, candidate in enumerate(PREDECLARED_CANDIDATES)
    }
    shortlist.sort(
        key=lambda report: (
            -report["metrics"]["nearest_reward_config_accuracy"],
            -report["metrics"]["preference_sensitive_cooking_accuracy"],
            registry_priority[report["candidate_id"]],
        )
    )
    return shortlist[0], {
        "primary": "minimum dev varying-dimension MSE",
        "shortlist_tolerance": tolerance,
        "tie_breaks": [
            "higher dev nearest-reward-config accuracy",
            "higher dev preference-sensitive empty-hand cooking majority accuracy",
            "predeclared registry priority (paper SGD first)",
        ],
        "shortlisted_candidates": [row["candidate_id"] for row in shortlist],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feedback", type=Path, default=DEFAULT_FEEDBACK)
    parser.add_argument("--probe-states", type=Path, default=DEFAULT_PROBE_STATES_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--n-folds", type=int, default=10)
    parser.add_argument("--split-seed", type=int, default=137)
    parser.add_argument("--model-seed", type=int, default=137)
    parser.add_argument("--min-freq", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--patience", type=int, default=2)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=256)
    args = parser.parse_args()

    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(
            f"selection output must be new/empty so frozen models are not overwritten: {args.output_dir}"
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)

    raw = args.feedback.read_bytes()
    raw_sha256 = hashlib.sha256(raw).hexdigest()
    payload = json.loads(raw.decode("utf-8"))
    dataset = build_dataset(
        payload,
        load_probe_states(args.probe_states),
        load_features(),
        min_freq=args.min_freq,
        n_folds=args.n_folds,
        seed=args.split_seed,
        paper_cv_mode=True,
        source_corpus_sha256=raw_sha256,
    )
    cv_manifest = make_paper_cross_validation_manifest(
        dataset["examples"], n_folds=args.n_folds, seed=args.split_seed
    )
    if cv_manifest["teacher_reward_axis_graph"]["component_count"] != 1:
        raise ValueError("selection requires one connected teacher/reward graph")

    registry = {
        "candidates": list(PREDECLARED_CANDIDATES),
        "epochs": args.epochs,
        "patience": args.patience,
        "weight_decay": args.weight_decay,
        "batch_size": args.batch_size,
        "min_freq": args.min_freq,
        "early_stop_metric": "fold-train-derived varying-dimension dev MSE",
    }
    registry_sha256 = canonical_sha256(registry)
    canonical_config = next(
        config
        for config in payload["reward_configurations"]
        if str(config["reward_config_id"]) == "reward_00_gold"
    )
    canonical_vector = torch.tensor(
        [float(canonical_config["weights"][feature]) for feature in dataset["features"]],
        dtype=torch.float32,
    )

    fold_reports = []
    selected_totals = []
    ensemble_members = []
    for fold in cv_manifest["folds"]:
        fold_id = int(fold["fold"])
        print(f"Fold {fold_id:02d} dev-only candidate selection", flush=True)
        selection_data = _selection_view(dataset, fold, min_freq=args.min_freq)
        train_examples = [
            selection_data["examples"][index]
            for index in selection_data["split"]["train_indices"]
        ]
        dev_examples = [
            selection_data["examples"][index]
            for index in selection_data["split"]["dev_indices"]
        ]
        _, varying_dimensions, varying_mask_list = _fold_dimension_weights(
            dataset,
            fold["train_indices"],
            varying_dimension_weight=1.0,
        )
        if varying_dimensions != 26:
            raise ValueError(
                f"fold {fold_id}: expected 26 train-varying dimensions, got {varying_dimensions}"
            )
        varying_mask = torch.tensor(varying_mask_list, dtype=torch.bool)
        dev_catalog = _reward_catalog(dev_examples)
        sensitive_groups = _preference_sensitive_groups(dev_examples)

        train_targets = torch.tensor(
            [example["target_reward"] for example in train_examples], dtype=torch.float32
        )
        baselines = {}
        baseline_vectors = {
            "train_mean": train_targets.mean(dim=0).repeat(len(dev_examples), 1),
            "canonical": canonical_vector.repeat(len(dev_examples), 1),
        }
        dev_targets = torch.tensor(
            [example["target_reward"] for example in dev_examples], dtype=torch.float32
        )
        for name, vectors in baseline_vectors.items():
            suite, _ = _evaluation_suite(
                dev_examples,
                vectors,
                dev_targets,
                dataset["features"],
                dev_catalog,
                varying_mask,
                sensitive_groups,
            )
            baselines[name] = _rates(_metric_totals(suite))

        candidate_reports = []
        candidate_models = {}
        for candidate in PREDECLARED_CANDIDATES:
            dimension_weights, _, _ = _fold_dimension_weights(
                dataset,
                fold["train_indices"],
                varying_dimension_weight=float(candidate["varying_dimension_weight"]),
            )
            result = train(
                selection_data,
                epochs=args.epochs,
                lr=float(candidate["lr"]),
                weight_decay=args.weight_decay,
                batch_size=args.batch_size,
                patience=args.patience,
                optimizer_name=str(candidate["optimizer"]),
                use_feature_counts=True,
                seed=args.model_seed + fold_id,
                evaluate_test=False,
                dimension_weights=dimension_weights,
                batch_reduction=str(candidate["batch_reduction"]),
                early_stop_dimension_mask=varying_mask_list,
            )
            if (
                result["untouched_test_loss"] is not None
                or result["untouched_test_metrics"] is not None
                or result["split_sizes"]["test"] != 0
            ):
                raise RuntimeError("candidate selection accessed a test example")
            predictions, targets = _predict(
                result["model"], dev_examples, selection_data["vocab"]
            )
            suite, _ = _evaluation_suite(
                dev_examples,
                predictions,
                targets,
                dataset["features"],
                dev_catalog,
                varying_mask,
                sensitive_groups,
            )
            totals = _metric_totals(suite)
            metrics = _rates(totals)
            report = {
                **candidate,
                "partition": "dev",
                "test_examples_evaluated": 0,
                "best_epoch": result["best_epoch"],
                "epochs_run": result["epochs_run"],
                "metrics": metrics,
                "metric_totals": totals,
            }
            candidate_reports.append(report)
            candidate_models[candidate["candidate_id"]] = result["model"]
            print(
                f"  {candidate['candidate_id']}: mse={metrics['varying_dimension_mse']:.4f} "
                f"nearest={metrics['nearest_reward_config_accuracy']:.3f} "
                f"cooking={metrics['preference_sensitive_cooking_accuracy']:.3f}",
                flush=True,
            )

        winner, selection_rule = _choose_fold_winner(candidate_reports)
        selected_totals.append(winner["metric_totals"])
        model = candidate_models[winner["candidate_id"]]
        fold_dir = args.output_dir / f"fold_{fold_id:02d}"
        checkpoint = fold_dir / "model.pt"
        save_checkpoint(
            checkpoint,
            model,
            selection_data["vocab"],
            dataset["features"],
            use_feature_counts=True,
            extra={
                "fold": fold_id,
                "selection_partition": "dev",
                "test_accessed": False,
                "selected_candidate": winner["candidate_id"],
                "candidate_registry_sha256": registry_sha256,
                "cross_validation_manifest_sha256": cv_manifest["manifest_sha256"],
                "corpus_sha256": raw_sha256,
            },
        )
        checkpoint_relative = f"fold_{fold_id:02d}/model.pt"
        checkpoint_sha256 = _file_sha256(checkpoint)
        best_constant = {
            "varying_dimension_mse": min(
                row["varying_dimension_mse"] for row in baselines.values()
            ),
            "nearest_reward_config_accuracy": max(
                row["nearest_reward_config_accuracy"] for row in baselines.values()
            ),
            "preference_sensitive_cooking_accuracy": max(
                row["preference_sensitive_cooking_accuracy"]
                for row in baselines.values()
            ),
        }
        fold_report = {
            "fold": fold_id,
            "partition": "dev",
            "test_accessed": False,
            "train_examples": len(train_examples),
            "dev_examples": len(dev_examples),
            "test_examples_evaluated": 0,
            "varying_dimensions_from_train": varying_dimensions,
            "dev_reward_configs": sorted(dev_catalog),
            "dev_sensitive_groups": len(sensitive_groups),
            "constant_baselines": baselines,
            "candidates": candidate_reports,
            "selection_rule": selection_rule,
            "winner": {
                "candidate_id": winner["candidate_id"],
                "checkpoint": checkpoint_relative,
                "checkpoint_sha256": checkpoint_sha256,
                "config": {
                    key: winner[key]
                    for key in (
                        "optimizer",
                        "lr",
                        "varying_dimension_weight",
                        "batch_reduction",
                    )
                },
                "dev_metrics": winner["metrics"],
                "beats_constant_checks": {
                    "varying_dimension_mse": (
                        winner["metrics"]["varying_dimension_mse"]
                        < best_constant["varying_dimension_mse"]
                    ),
                    "nearest_reward_config_accuracy": (
                        winner["metrics"]["nearest_reward_config_accuracy"]
                        > best_constant["nearest_reward_config_accuracy"]
                    ),
                    "preference_sensitive_cooking_accuracy": (
                        winner["metrics"]["preference_sensitive_cooking_accuracy"]
                        > best_constant["preference_sensitive_cooking_accuracy"]
                    ),
                },
            },
        }
        write_json(fold_dir / "dev_selection_report.json", fold_report)
        fold_reports.append(fold_report)
        ensemble_members.append(
            {
                "fold": fold_id,
                "checkpoint": checkpoint_relative,
                "checkpoint_sha256": checkpoint_sha256,
                "selected_candidate": winner["candidate_id"],
                "dev_metrics": winner["metrics"],
                "test_metrics": None,
            }
        )
        print(f"  selected: {winner['candidate_id']}", flush=True)

    winner_counts = dict(
        sorted(Counter(row["winner"]["candidate_id"] for row in fold_reports).items())
    )
    selection_manifest = {
        "schema_version": "route2-per-fold-dev-selection-v1",
        "protocol": "per_fold_train_dev_candidate_selection",
        "partition": "dev",
        "test_accessed": False,
        "test_examples_evaluated": 0,
        "warning": (
            "seed 137 is a frozen repartition of the same synthetic corpus, not "
            "an independent new-language corpus"
        ),
        "corpus_sha256": raw_sha256,
        "split_seed": args.split_seed,
        "model_seed": args.model_seed,
        "cross_validation_manifest_sha256": cv_manifest["manifest_sha256"],
        "candidate_registry": registry,
        "candidate_registry_sha256": registry_sha256,
        "winner_counts": winner_counts,
        "selected_dev_micro_metrics": _rates(_sum_totals(selected_totals)),
        "folds": fold_reports,
        "next_step": (
            "Freeze these checkpoints, then run evaluate_route2_paper_crossval.py "
            "once on the corresponding seed-137 test partitions."
        ),
    }
    selection_manifest["manifest_sha256"] = canonical_sha256(selection_manifest)
    write_json(args.output_dir / "selection_manifest.json", selection_manifest)
    write_json(args.output_dir / "cross_validation_manifest.json", cv_manifest)

    ensemble = {
        "schema_version": ENSEMBLE_SCHEMA_VERSION,
        "purpose": "frozen_per_fold_dev_selected_models_before_one_time_test",
        "features": dataset["features"],
        "use_feature_counts": True,
        "target_mode": dataset["target_mode"],
        "corpus_sha256": raw_sha256,
        "cross_validation_manifest_sha256": cv_manifest["manifest_sha256"],
        "selection_manifest_sha256": selection_manifest["manifest_sha256"],
        "test_accessed": False,
        "members": ensemble_members,
    }
    ensemble["manifest_sha256"] = canonical_sha256(ensemble)
    write_json(args.output_dir / "ensemble_manifest.json", ensemble)
    print(json.dumps({"winner_counts": winner_counts}, indent=2), flush=True)
    print(f"Frozen selection: {args.output_dir / 'selection_manifest.json'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
