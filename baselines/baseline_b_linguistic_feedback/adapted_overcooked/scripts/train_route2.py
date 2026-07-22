"""Train the Route 2 inference network: language -> comfort reward vector.

This is the H0/subgoal training entry point (distinct from
``train_route2_inference_network.py``, which is the strict paper-alignment
scaffold that deliberately stops before training). Here we actually fit the
network on the synthetic corpus:

    (tokens[, feature counts]) -> reward vector over the 53-dim schema

Design choices for the DURF setting:
- **Pure language -> reward by default**: the trajectory-feature input is
  zeroed unless ``--use-feature-counts`` is passed. At real runtime we only
  have the human's text, not the grounded reference vector, so conditioning on
  the grounding would leak the answer. Zeroing keeps the metric honest.
- **Grouped validation**: one CV fold's held-out scenarios are the validation
  set, so we early-stop on *unseen decision contexts*, not memorized ones.

Outputs ``outputs/route2/model.pt`` (+ ``vocab.json``) for downstream subgoal
evaluation and the PPO comfort-reward bridge.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.prepare_route2_dataset import build_dataset  # noqa: E402
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


def train(
    dataset: dict,
    *,
    val_fold: int = 0,
    epochs: int = 200,
    lr: float = 0.01,
    weight_decay: float = 1e-4,
    batch_size: int = 64,
    patience: int = 20,
    use_feature_counts: bool = False,
    seed: int = 0,
) -> dict:
    torch.manual_seed(seed)
    rng = torch.Generator()
    rng.manual_seed(seed)

    fold = dataset["folds"][val_fold % len(dataset["folds"])]
    train_examples = [dataset["examples"][i] for i in fold["train_indices"]]
    val_examples = [dataset["examples"][i] for i in fold["test_indices"]]

    model = TrajectoryFeedbackRewardPredictor(
        vocab_size=dataset["vocab_size"], n_features=dataset["n_features"]
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.MSELoss()

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
            loss = loss_fn(predictions, batch["targets"])
            loss.backward()
            optimizer.step()

        model.eval()
        train_loss = _epoch_loss(
            model, train_examples, dataset["vocab"], loss_fn, use_feature_counts=use_feature_counts
        )
        val_loss = _epoch_loss(
            model, val_examples, dataset["vocab"], loss_fn, use_feature_counts=use_feature_counts
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
    return {
        "model": model,
        "best_val_loss": best_val,
        "best_epoch": best_epoch,
        "epochs_run": len(history),
        "history": history,
        "val_fold": val_fold,
        "use_feature_counts": use_feature_counts,
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
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--use-feature-counts",
        action="store_true",
        help="Also condition on the grounded reference vector (leaks grounding; "
        "off by default so the network learns pure language -> reward).",
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
    )

    print("Route 2 training (language -> comfort reward):")
    print(f"  examples:   {len(dataset['examples'])}")
    print(f"  features:   {dataset['n_features']}")
    print(f"  vocab size: {dataset['vocab_size']}")
    print(f"  use_feature_counts: {args.use_feature_counts}")

    result = train(
        dataset,
        val_fold=args.val_fold,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        batch_size=args.batch_size,
        patience=args.patience,
        use_feature_counts=args.use_feature_counts,
        seed=args.seed,
    )

    print(
        f"  trained {result['epochs_run']} epochs; "
        f"best val MSE={result['best_val_loss']:.6f} at epoch {result['best_epoch']}"
    )

    save_checkpoint(
        args.output,
        result["model"],
        dataset["vocab"],
        dataset["features"],
        use_feature_counts=args.use_feature_counts,
        extra={
            "best_val_loss": result["best_val_loss"],
            "best_epoch": result["best_epoch"],
            "val_fold": result["val_fold"],
        },
    )
    print(f"  saved checkpoint: {args.output}")
    print(f"  saved vocab:      {args.output.parent / 'vocab.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
