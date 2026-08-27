"""Train the Route 2 inference network: language -> comfort reward vector.

This is the H0/subgoal training entry point (distinct from
``train_route2_inference_network.py``, which is the strict paper-alignment
scaffold that deliberately stops before training). Here we actually fit the
network on the synthetic corpus:

    (tokens[, feature counts]) -> reward vector over the 53-dim schema

Paper-aligned full-reward corpora use both language and normalized trajectory
feature counts. ``--text-only-ablation`` keeps the identical network but zeros
the trajectory input. Paper cross-validation constructs stable teacher and
reward-configuration holdouts outside this generic training entry point.

Outputs ``outputs/route2/model.pt`` (+ ``vocab.json``) for downstream subgoal
evaluation and the PPO comfort-reward bridge.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.prepare_route2_dataset import build_dataset  # noqa: E402
from src.evaluation_splits import canonical_sha256, write_split_manifest  # noqa: E402
from src.feature_schema import load_features, read_json  # noqa: E402
from src.neural_inference import (  # noqa: E402
    TrajectoryFeedbackRewardPredictor,
    collate_batch,
    save_checkpoint,
)
from src.probe_evaluator import DEFAULT_PROBE_STATES_PATH, load_probe_states  # noqa: E402


DEFAULT_FEEDBACK_PATH = ROOT / "data" / "synthetic_feedback.json"
DEFAULT_OUTPUT = ROOT / "outputs" / "route2" / "model.pt"


def _make_batches(indices: list[int], batch_size: int, rng: torch.Generator) -> list[list[int]]:
    order = torch.randperm(len(indices), generator=rng).tolist()
    shuffled = [indices[i] for i in order]
    return [shuffled[i : i + batch_size] for i in range(0, len(shuffled), batch_size)]


def _prep_batch(examples: list[dict], vocab: dict, *, use_feature_counts: bool) -> dict:
    batch = collate_batch(examples, vocab)
    if not use_feature_counts:
        batch["feature_counts"] = torch.zeros_like(batch["feature_counts"])
    return batch


def _epoch_loss(
    model: TrajectoryFeedbackRewardPredictor,
    examples: list[dict],
    vocab: dict,
    loss_fn,
    *,
    use_feature_counts: bool,
) -> float:
    if not examples:
        return 0.0
    batch = _prep_batch(examples, vocab, use_feature_counts=use_feature_counts)
    with torch.no_grad():
        predictions = model(batch["tokens"], batch["offsets"], batch["feature_counts"])
        return float(loss_fn(predictions, batch["targets"]))


def _vector_metrics(
    model: TrajectoryFeedbackRewardPredictor,
    examples: list[dict],
    vocab: dict,
    *,
    use_feature_counts: bool,
) -> dict | None:
    if not examples:
        return None
    batch = _prep_batch(examples, vocab, use_feature_counts=use_feature_counts)
    with torch.no_grad():
        predictions = model(batch["tokens"], batch["offsets"], batch["feature_counts"])
    targets = batch["targets"]
    mse = float(torch.mean((predictions - targets) ** 2))
    cosine = torch.nn.functional.cosine_similarity(predictions, targets, dim=1, eps=1e-8)
    active = torch.abs(targets) > 1e-8
    sign_correct = (torch.sign(predictions) == torch.sign(targets)) & active
    return {
        "mse": mse,
        "mean_cosine_similarity": float(torch.mean(cosine)),
        "active_sign_accuracy": (
            float(sign_correct.sum() / active.sum()) if int(active.sum()) else 0.0
        ),
        "active_target_count": int(active.sum()),
        "examples": len(examples),
    }

def train(
    dataset: dict,
    *,
    val_fold: int = 0,
    epochs: int = 200,
    lr: float = 0.01,
    weight_decay: float = 1e-4,
    batch_size: int = 64,
    patience: int = 20,
    optimizer_name: str = "adam",
    use_feature_counts: bool | None = None,
    seed: int = 0,
    evaluate_test: bool = True,
    allow_local_feedback_ablation: bool = False,
    dimension_weights: list[float] | None = None,
    batch_reduction: str = "mean",
    early_stop_dimension_mask: list[bool] | None = None,
) -> dict:
    target_mode = dataset.get("target_mode")
    if target_mode != "full_teacher_reward" and not allow_local_feedback_ablation:
        raise ValueError(
            "Route 2 supervised training requires independent complete "
            "teacher_reward_weights (target_mode=full_teacher_reward). Natural "
            "language-only rows must remain unlabeled; model predictions must not "
            "be used as gold. Pass allow_local_feedback_ablation=True only for the "
            "explicit legacy local-grounding ablation."
        )
    if use_feature_counts is None:
        use_feature_counts = bool(dataset.get("use_feature_counts", False))
    torch.manual_seed(seed)
    rng = torch.Generator()
    rng.manual_seed(seed)

    split = dataset.get("split")
    if split:
        train_examples = [dataset["examples"][i] for i in split["train_indices"]]
        val_examples = [dataset["examples"][i] for i in split["dev_indices"]]
        test_examples = [dataset["examples"][i] for i in split["test_indices"]]
    else:
        # Backward compatibility for previously serialized datasets.
        fold = dataset["folds"][val_fold % len(dataset["folds"])]
        train_examples = [dataset["examples"][i] for i in fold["train_indices"]]
        val_examples = [dataset["examples"][i] for i in fold["test_indices"]]
        test_examples = []

    model = TrajectoryFeedbackRewardPredictor(
        vocab_size=dataset["vocab_size"], n_features=dataset["n_features"]
    )
    if optimizer_name == "adam":
        optimizer = torch.optim.Adam(
            model.parameters(), lr=lr, weight_decay=weight_decay
        )
    elif optimizer_name == "sgd":
        optimizer = torch.optim.SGD(
            model.parameters(), lr=lr, weight_decay=weight_decay
        )
    else:
        raise ValueError("optimizer_name must be 'adam' or 'sgd'")
    if batch_reduction not in {"mean", "sum_examples"}:
        raise ValueError("batch_reduction must be 'mean' or 'sum_examples'")
    if dimension_weights is None:
        weight_tensor = torch.ones(dataset["n_features"], dtype=torch.float32)
        resolved_dimension_weights = None
    else:
        if len(dimension_weights) != dataset["n_features"]:
            raise ValueError("dimension_weights must match the reward-vector width")
        weight_tensor = torch.tensor(dimension_weights, dtype=torch.float32)
        if bool(torch.any(weight_tensor < 0)) or not float(weight_tensor.sum()):
            raise ValueError("dimension_weights must be non-negative and nonzero")
        resolved_dimension_weights = [float(value) for value in dimension_weights]

    def per_example_loss(predictions, targets):
        return torch.mean((predictions - targets) ** 2 * weight_tensor, dim=1)

    def training_loss_fn(predictions, targets):
        losses = per_example_loss(predictions, targets)
        return losses.sum() if batch_reduction == "sum_examples" else losses.mean()

    if early_stop_dimension_mask is None:
        early_stop_mask_tensor = None
        resolved_early_stop_mask = None
    else:
        if len(early_stop_dimension_mask) != dataset["n_features"]:
            raise ValueError("early_stop_dimension_mask must match reward-vector width")
        early_stop_mask_tensor = torch.tensor(
            early_stop_dimension_mask, dtype=torch.bool
        )
        if not bool(early_stop_mask_tensor.any()):
            raise ValueError("early_stop_dimension_mask must select at least one dimension")
        resolved_early_stop_mask = [bool(value) for value in early_stop_dimension_mask]

    def evaluation_loss_fn(predictions, targets):
        # Early stopping and reports remain per-example means regardless of
        # how a training batch approximates the paper's one-example SGD steps.
        if early_stop_mask_tensor is None:
            return per_example_loss(predictions, targets).mean()
        return torch.mean(
            (predictions[:, early_stop_mask_tensor] - targets[:, early_stop_mask_tensor])
            ** 2
        )

    train_indices = list(range(len(train_examples)))
    best_val = float("inf")
    best_state = {k: v.clone() for k, v in model.state_dict().items()}
    best_epoch = -1
    epochs_no_improve = 0
    history: list[dict] = []

    for epoch in range(epochs):
        model.train()
        for batch_indices in _make_batches(train_indices, batch_size, rng):
            batch_examples = [train_examples[i] for i in batch_indices]
            batch = _prep_batch(batch_examples, dataset["vocab"], use_feature_counts=use_feature_counts)
            optimizer.zero_grad()
            predictions = model(batch["tokens"], batch["offsets"], batch["feature_counts"])
            loss = training_loss_fn(predictions, batch["targets"])
            loss.backward()
            optimizer.step()

        model.eval()
        train_loss = _epoch_loss(
            model, train_examples, dataset["vocab"], evaluation_loss_fn, use_feature_counts=use_feature_counts
        )
        val_loss = _epoch_loss(
            model, val_examples, dataset["vocab"], evaluation_loss_fn, use_feature_counts=use_feature_counts
        )
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})

        if val_loss < best_val - 1e-9:
            best_val = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            best_epoch = epoch
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                break

    model.load_state_dict(best_state)
    model.eval()
    untouched_test_loss = (
        _epoch_loss(
            model,
            test_examples,
            dataset["vocab"],
            evaluation_loss_fn,
            use_feature_counts=use_feature_counts,
        )
        if test_examples and evaluate_test
        else None
    )
    untouched_test_metrics = (
        _vector_metrics(
            model,
            test_examples,
            dataset["vocab"],
            use_feature_counts=use_feature_counts,
        )
        if test_examples and evaluate_test
        else None
    )
    return {
        "model": model,
        "best_val_loss": best_val,
        "best_epoch": best_epoch,
        "epochs_run": len(history),
        "history": history,
        "val_fold": val_fold,
        "use_feature_counts": use_feature_counts,
        "dimension_weights": resolved_dimension_weights,
        "batch_reduction": batch_reduction,
        "early_stop_dimension_mask": resolved_early_stop_mask,
        "untouched_test_loss": untouched_test_loss,
        "untouched_test_metrics": untouched_test_metrics,
        "split_sizes": {
            "train": len(train_examples),
            "dev": len(val_examples),
            "test": len(test_examples),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feedback", type=Path, default=DEFAULT_FEEDBACK_PATH)
    parser.add_argument("--probe-states", type=Path, default=DEFAULT_PROBE_STATES_PATH)
    parser.add_argument("--min-freq", type=int, default=1)
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--val-fold", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--optimizer", choices=("adam", "sgd"), default="adam")
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--split-manifest",
        type=Path,
        help="Read and validate a fixed split manifest.",
    )
    parser.add_argument("--near-duplicate-threshold", type=float, default=0.92)
    parser.add_argument(
        "--grouping-policy",
        choices=("joint", "scenario", "template"),
        default="joint",
        help="Holdout axis used when no fixed manifest is supplied.",
    )
    parser.add_argument(
        "--text-only-ablation",
        action="store_true",
        help="Zero trajectory counts while keeping the same network architecture.",
    )
    parser.add_argument(
        "--allow-local-feedback-ablation",
        action="store_true",
        help=(
            "Explicitly allow the legacy local feature-times-valence target. "
            "Never use this flag to train on raw human-language sessions."
        ),
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    feedback_examples = read_json(args.feedback)
    probes = load_probe_states(args.probe_states)
    features = load_features()
    dataset = build_dataset(
        feedback_examples,
        probes,
        features,
        min_freq=args.min_freq,
        n_folds=args.n_folds,
        seed=args.seed,
        split_manifest=args.split_manifest,
        near_duplicate_threshold=args.near_duplicate_threshold,
        grouping_policy=args.grouping_policy,
    )
    use_feature_counts = bool(dataset["use_feature_counts"]) and not args.text_only_ablation
    training_config = {
        "seed": args.seed,
        "min_freq": args.min_freq,
        "n_folds": args.n_folds,
        "val_fold": args.val_fold,
        "epochs": args.epochs,
        "optimizer": args.optimizer,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "batch_size": args.batch_size,
        "patience": args.patience,
        "use_feature_counts": use_feature_counts,
        "target_mode": dataset["target_mode"],
        "text_only_ablation": args.text_only_ablation,
        "near_duplicate_threshold": args.near_duplicate_threshold,
        "grouping_policy": args.grouping_policy,
        "allow_local_feedback_ablation": args.allow_local_feedback_ablation,
    }
    training_config["config_sha256"] = canonical_sha256(training_config)

    print("Route 2 training (language -> comfort reward):")
    print(f"  examples:   {len(dataset['examples'])}")
    print(f"  features:   {dataset['n_features']}")
    print(f"  vocab size: {dataset['vocab_size']}")
    print(f"  target_mode: {dataset['target_mode']}")
    print(f"  use_feature_counts: {use_feature_counts}")
    print(f"  corpus SHA256: {dataset['corpus_sha256']}")
    print(f"  split SHA256:  {dataset['split_sha256']}")

    result = train(
        dataset,
        val_fold=args.val_fold,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        batch_size=args.batch_size,
        patience=args.patience,
        optimizer_name=args.optimizer,
        use_feature_counts=use_feature_counts,
        seed=args.seed,
        allow_local_feedback_ablation=args.allow_local_feedback_ablation,
    )

    print(
        f"  trained {result['epochs_run']} epochs; "
        f"best val MSE={result['best_val_loss']:.6f} at epoch {result['best_epoch']}"
    )
    if result["untouched_test_loss"] is not None:
        print(f"  untouched test MSE={result['untouched_test_loss']:.6f}")

    save_checkpoint(
        args.output,
        result["model"],
        dataset["vocab"],
        dataset["features"],
        use_feature_counts=use_feature_counts,
        extra={
            "best_val_loss": result["best_val_loss"],
            "best_epoch": result["best_epoch"],
            "val_fold": result["val_fold"],
            "untouched_test_loss": result["untouched_test_loss"],
            "split_sizes": result["split_sizes"],
            "split_policy": dataset.get("split_policy"),
            "corpus_sha256": dataset["corpus_sha256"],
            "untouched_test_metrics": result["untouched_test_metrics"],
            "target_mode": dataset["target_mode"],
            "split_sha256": dataset["split_sha256"],
            "dataset_config_sha256": dataset["dataset_config_sha256"],
            "seed": args.seed,
            "training_config": training_config,
        },
    )
    manifest_out = args.output.parent / "split_manifest.json"
    write_split_manifest(manifest_out, dataset["split_manifest"])
    print(f"  saved checkpoint: {args.output}")
    print(f"  saved vocab:      {args.output.parent / 'vocab.json'}")
    print(f"  saved split:      {manifest_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
