"""Train the paper-style 10-fold Route 2 ensemble.

Each fold holds out one teacher fold for validation and the following teacher
fold for testing; reward configurations rotate in exactly the same way.  A
fold-specific vocabulary is fit on training text only.  Cross-validation
metrics use only each model's own held-out test split, while the ten-model mean
is reserved for deployment on new human feedback.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.prepare_route2_dataset import build_dataset  # noqa: E402
from scripts.train_route2 import train  # noqa: E402
from src.evaluation_splits import canonical_sha256  # noqa: E402
from src.feature_schema import load_features, read_json  # noqa: E402
from src.neural_inference import (  # noqa: E402
    ENSEMBLE_SCHEMA_VERSION,
    build_vocab,
    save_checkpoint,
)
from src.paper_cross_validation import (  # noqa: E402
    aggregate_vector_metrics,
    make_paper_cross_validation_manifest,
)
from src.probe_evaluator import DEFAULT_PROBE_STATES_PATH, load_probe_states  # noqa: E402


DEFAULT_FEEDBACK = ROOT / "data" / "route2_teacher_feedback.paper_v5.synthetic.json"
DEFAULT_OUTPUT = ROOT / "outputs" / "route2" / "paper_aligned_v5_10fold"


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _fold_dataset(dataset: dict, fold: dict, *, min_freq: int) -> dict:
    selected = dict(dataset)
    selected["split"] = {
        "train_indices": list(fold["train_indices"]),
        "dev_indices": list(fold["dev_indices"]),
        "test_indices": list(fold["test_indices"]),
    }
    train_tokens = [dataset["examples"][index]["tokens"] for index in fold["train_indices"]]
    selected["vocab"] = build_vocab(train_tokens, min_freq=min_freq)
    selected["vocab_size"] = len(selected["vocab"])
    return selected


def _fold_dimension_weights(
    dataset: dict,
    train_indices: list[int],
    *,
    varying_dimension_weight: float,
) -> tuple[list[float] | None, int, list[bool]]:
    """Derive the varying mask from this fold's training targets only."""

    targets = torch.tensor(
        [dataset["examples"][index]["target_reward"] for index in train_indices],
        dtype=torch.float32,
    )
    varying = (targets.max(dim=0).values - targets.min(dim=0).values) > 1e-8
    if varying_dimension_weight == 1.0:
        return None, int(varying.sum()), varying.tolist()
    weights = torch.ones(dataset["n_features"], dtype=torch.float32)
    weights[varying] = float(varying_dimension_weight)
    return weights.tolist(), int(varying.sum()), varying.tolist()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feedback", type=Path, default=DEFAULT_FEEDBACK)
    parser.add_argument("--probe-states", type=Path, default=DEFAULT_PROBE_STATES_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--n-folds", type=int, default=10)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--model-seed", type=int, default=42)
    parser.add_argument("--min-freq", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--patience", type=int, default=2)
    parser.add_argument("--optimizer", choices=("adam", "sgd"), default="sgd")
    parser.add_argument("--lr", type=float, default=0.005)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument(
        "--batch-reduction",
        choices=("mean", "sum_examples"),
        default="sum_examples",
        help=(
            "sum_examples approximates the paper's one-example SGD gradient "
            "accumulation; dev/test losses remain per-example means."
        ),
    )
    parser.add_argument(
        "--varying-dimension-weight",
        type=float,
        default=1.0,
        help="Training-only weight for dimensions varying in that fold's train rewards.",
    )
    parser.add_argument("--text-only-ablation", action="store_true")
    parser.add_argument(
        "--allow-sparse-axis-components",
        action="store_true",
        help="Permit non-paper sparse teacher/reward graphs (diagnostic only).",
    )
    args = parser.parse_args()
    if args.varying_dimension_weight <= 0:
        raise ValueError("--varying-dimension-weight must be positive")

    raw = args.feedback.read_bytes()
    raw_sha256 = hashlib.sha256(raw).hexdigest()
    dataset = build_dataset(
        json.loads(raw.decode("utf-8")),
        load_probe_states(args.probe_states),
        load_features(),
        min_freq=args.min_freq,
        n_folds=args.n_folds,
        seed=args.split_seed,
        paper_cv_mode=True,
        source_corpus_sha256=raw_sha256,
    )
    if dataset["target_mode"] != "full_teacher_reward":
        raise ValueError("paper-style Route 2 cross-validation requires full teacher rewards")
    manifest = make_paper_cross_validation_manifest(
        dataset["examples"], n_folds=args.n_folds, seed=args.split_seed
    )
    component_count = manifest["teacher_reward_axis_graph"]["component_count"]
    if component_count != 1 and not args.allow_sparse_axis_components:
        raise ValueError(
            "paper-aligned augmentation requires one connected teacher/reward "
            f"bipartite graph, found {component_count}; regenerate every base "
            "teacher/game across every reward configuration"
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(args.output_dir / "cross_validation_manifest.json", manifest)

    use_feature_counts = bool(dataset["use_feature_counts"]) and not args.text_only_ablation
    training_config = {
        "architecture": "paper_embeddingbag_30_plus_trajectory_relu_128",
        "optimizer": args.optimizer,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "batch_size": args.batch_size,
        "batch_reduction": args.batch_reduction,
        "epochs": args.epochs,
        "patience": args.patience,
        "min_freq": args.min_freq,
        "split_seed": args.split_seed,
        "model_seed": args.model_seed,
        "n_folds": args.n_folds,
        "rotation": "validation=i,test=(i+1)%n,train=remaining",
        "holdout_axes": ["teacher_id", "reward_config_id"],
        "use_feature_counts": use_feature_counts,
        "target_mode": dataset["target_mode"],
        "supervision_identifiability": "same_input_conflicts_rejected",
        "varying_dimension_weight": args.varying_dimension_weight,
        "varying_mask_source": "fold_train_targets_only",
    }
    training_config["config_sha256"] = canonical_sha256(training_config)

    fold_reports = []
    ensemble_members = []
    for fold in manifest["folds"]:
        fold_id = int(fold["fold"])
        print(
            f"Fold {fold_id:02d}: train={len(fold['train_indices'])} "
            f"dev={len(fold['dev_indices'])} test={len(fold['test_indices'])}",
            flush=True,
        )
        fold_data = _fold_dataset(dataset, fold, min_freq=args.min_freq)
        dimension_weights, varying_dimensions, varying_mask = _fold_dimension_weights(
            dataset,
            fold["train_indices"],
            varying_dimension_weight=args.varying_dimension_weight,
        )
        result = train(
            fold_data,
            epochs=args.epochs,
            lr=args.lr,
            weight_decay=args.weight_decay,
            batch_size=args.batch_size,
            patience=args.patience,
            optimizer_name=args.optimizer,
            use_feature_counts=use_feature_counts,
            seed=args.model_seed + fold_id,
            evaluate_test=True,
            dimension_weights=dimension_weights,
            batch_reduction=args.batch_reduction,
            early_stop_dimension_mask=varying_mask,
        )
        fold_dir = args.output_dir / f"fold_{fold_id:02d}"
        checkpoint = fold_dir / "model.pt"
        save_checkpoint(
            checkpoint,
            result["model"],
            fold_data["vocab"],
            fold_data["features"],
            use_feature_counts=use_feature_counts,
            extra={
                "fold": fold_id,
                "best_val_loss": result["best_val_loss"],
                "best_epoch": result["best_epoch"],
                "test_metrics": result["untouched_test_metrics"],
                "cross_validation_manifest_sha256": manifest["manifest_sha256"],
                "corpus_sha256": raw_sha256,
                "training_config": training_config,
            },
        )
        fold_report = {
            "fold": fold_id,
            "seed": args.model_seed + fold_id,
            "checkpoint": str(checkpoint),
            "vocab_size": fold_data["vocab_size"],
            "best_dev_mse": result["best_val_loss"],
            "best_epoch": result["best_epoch"],
            "epochs_run": result["epochs_run"],
            "split_sizes": result["split_sizes"],
            "test_metrics": result["untouched_test_metrics"],
            "dev_teachers": fold["dev_teachers"],
            "test_teachers": fold["test_teachers"],
            "dev_rewards": fold["dev_rewards"],
            "test_rewards": fold["test_rewards"],
            "coverage": fold.get("coverage"),
            "heldout_overlap_audit": fold.get("heldout_overlap_audit"),
            "varying_dimensions_in_train": varying_dimensions,
            "varying_dimension_weight": args.varying_dimension_weight,
            "batch_reduction": args.batch_reduction,
        }
        _write_json(fold_dir / "training_report.json", fold_report)
        fold_reports.append(fold_report)
        ensemble_members.append(
            {
                "fold": fold_id,
                "checkpoint": f"fold_{fold_id:02d}/model.pt",
                "test_metrics": result["untouched_test_metrics"],
            }
        )

    aggregate = aggregate_vector_metrics(row["test_metrics"] for row in fold_reports)
    ensemble = {
        "schema_version": ENSEMBLE_SCHEMA_VERSION,
        "purpose": "deployment_on_unseen_human_feedback_only",
        "warning": "Do not score this ensemble on CV fold tests; nine members may have seen them.",
        "features": dataset["features"],
        "use_feature_counts": use_feature_counts,
        "target_mode": dataset["target_mode"],
        "corpus_sha256": raw_sha256,
        "cross_validation_manifest_sha256": manifest["manifest_sha256"],
        "members": ensemble_members,
    }
    ensemble["manifest_sha256"] = canonical_sha256(ensemble)
    _write_json(args.output_dir / "ensemble_manifest.json", ensemble)
    report = {
        "protocol": "paper_10fold_teacher_and_reward_holdout",
        "feedback": str(args.feedback),
        "corpus_sha256": raw_sha256,
        "cross_validation_manifest_sha256": manifest["manifest_sha256"],
        "training_config": training_config,
        "cross_validation_coverage": manifest.get("coverage"),
        "validity_requirement": (
            "Run evaluate_route2_paper_crossval.py; do not claim reward inference "
            "unless its validity gate beats canonical and train-mean constants."
        ),
        "folds": fold_reports,
        "aggregate_held_out_test_metrics": aggregate,
        "ensemble_manifest": str(args.output_dir / "ensemble_manifest.json"),
    }
    _write_json(args.output_dir / "cross_validation_report.json", report)
    print(json.dumps(aggregate, indent=2), flush=True)
    print(f"Ensemble: {args.output_dir / 'ensemble_manifest.json'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
