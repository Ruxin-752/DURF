"""Robust Route 2 selection on dev seeds; open untouched test once."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.prepare_route2_dataset import build_dataset  # noqa: E402
from scripts.train_route2 import train  # noqa: E402
from src.evaluation_splits import canonical_sha256, write_split_manifest  # noqa: E402
from src.feature_schema import load_features, read_json  # noqa: E402
from src.neural_inference import save_checkpoint  # noqa: E402
from src.probe_evaluator import DEFAULT_PROBE_STATES_PATH, load_probe_states  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feedback", type=Path, required=True)
    parser.add_argument("--probe-states", type=Path, default=DEFAULT_PROBE_STATES_PATH)
    parser.add_argument("--split-manifest", type=Path)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--learning-rates", nargs="+", type=float, default=[0.003, 0.01])
    parser.add_argument("--weight-decays", nargs="+", type=float, default=[0.0, 1e-4])
    parser.add_argument("--batch-sizes", nargs="+", type=int, default=[32, 64])
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "outputs" / "route2" / "multiseed",
    )
    args = parser.parse_args()

    feedback = read_json(args.feedback)
    dataset = build_dataset(
        feedback,
        load_probe_states(args.probe_states),
        load_features(),
        seed=0,
        split_manifest=args.split_manifest,
    )
    configurations = [
        {"lr": lr, "weight_decay": decay, "batch_size": batch}
        for lr in args.learning_rates
        for decay in args.weight_decays
        for batch in args.batch_sizes
    ]
    candidate_reports = []
    for config in configurations:
        seed_reports = []
        for seed in args.seeds:
            result = train(
                dataset,
                epochs=args.epochs,
                lr=config["lr"],
                weight_decay=config["weight_decay"],
                batch_size=config["batch_size"],
                patience=args.patience,
                seed=seed,
                evaluate_test=False,
            )
            seed_reports.append(
                {
                    "seed": seed,
                    "best_val_loss": result["best_val_loss"],
                    "best_epoch": result["best_epoch"],
                    "epochs_run": result["epochs_run"],
                }
            )
        candidate_reports.append(
            {
                "config": config,
                "seeds": seed_reports,
                "worst_seed_val_loss": max(
                    row["best_val_loss"] for row in seed_reports
                ),
                "mean_val_loss": mean(row["best_val_loss"] for row in seed_reports),
            }
        )
    selected = min(
        candidate_reports,
        key=lambda row: (row["worst_seed_val_loss"], row["mean_val_loss"]),
    )
    selected_seed = min(
        selected["seeds"], key=lambda row: row["best_val_loss"]
    )["seed"]
    final = train(
        dataset,
        epochs=args.epochs,
        lr=selected["config"]["lr"],
        weight_decay=selected["config"]["weight_decay"],
        batch_size=selected["config"]["batch_size"],
        patience=args.patience,
        seed=selected_seed,
        evaluate_test=True,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = args.output_dir / "model.pt"
    training_config = {
        **selected["config"],
        "seed": selected_seed,
        "epochs": args.epochs,
        "patience": args.patience,
        "selection": "minimize worst-seed dev loss, then mean dev loss",
    }
    save_checkpoint(
        checkpoint,
        final["model"],
        dataset["vocab"],
        dataset["features"],
        use_feature_counts=False,
        extra={
            "training_config": training_config,
            "corpus_sha256": dataset["corpus_sha256"],
            "split_sha256": dataset["split_sha256"],
            "dataset_config_sha256": dataset["dataset_config_sha256"],
        },
    )
    (args.output_dir / "vocab.json").write_text(
        json.dumps(dataset["vocab"], ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    write_split_manifest(args.output_dir / "split_manifest.json", dataset["split_manifest"])
    report = {
        "feedback": str(args.feedback),
        "corpus_sha256": dataset["corpus_sha256"],
        "split_sha256": dataset["split_sha256"],
        "search_sha256": canonical_sha256(candidate_reports),
        "candidates": candidate_reports,
        "selected_config": training_config,
        "selected_dev_loss": final["best_val_loss"],
        "untouched_test_loss": final["untouched_test_loss"],
        "checkpoint": str(checkpoint),
    }
    (args.output_dir / "training_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Selected: {training_config}")
    print(f"Untouched test MSE: {final['untouched_test_loss']:.6f}")
    print(f"Checkpoint: {checkpoint}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
