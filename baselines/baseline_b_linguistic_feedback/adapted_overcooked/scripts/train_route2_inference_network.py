"""Route 2 inference-network scaffold -- STOPS before training by default.

This is the deliberate "train pretraining boundary" for the strict-alignment
task. It reproduces everything the paper does up to, but not including, the
optimization loop in ``aaai_inference_network_training.ipynb`` (the cell with
``loss.backward()`` / ``optimizer.step()``):

1. assemble the dataset (tokens / feature counts / target reward triples),
   vocabulary, and grouped CV folds;
2. instantiate ``TrajectoryFeedbackRewardPredictor``;
3. run a single forward pass to validate tensor shapes end to end;
4. print ``TRAINING BOUNDARY REACHED`` and exit -- no gradients, no updates.

The training loop is written out but guarded behind ``--execute-training``,
which is intentionally NOT used for this task (and would need a real
teacher-learner corpus to be meaningful).
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
from scripts.run_baseline_b_pipeline import DEFAULT_FEEDBACK_PATH  # noqa: E402
from src.feature_schema import load_features, read_json  # noqa: E402
from src.neural_inference import (  # noqa: E402
    TrajectoryFeedbackRewardPredictor,
    collate_batch,
)
from src.probe_evaluator import DEFAULT_PROBE_STATES_PATH, load_probe_states  # noqa: E402


TRAINING_BOUNDARY_MESSAGE = "TRAINING BOUNDARY REACHED -- stopping before any parameter update."


def forward_check(dataset: dict) -> dict:
    """Instantiate the model and run one forward pass to validate shapes."""

    model = TrajectoryFeedbackRewardPredictor(
        vocab_size=dataset["vocab_size"], n_features=dataset["n_features"]
    )
    fold = dataset["folds"][0]
    train_examples = [dataset["examples"][i] for i in fold["train_indices"]]
    batch = collate_batch(train_examples, dataset["vocab"])
    with torch.no_grad():
        predictions = model(batch["tokens"], batch["offsets"], batch["feature_counts"])
    return {
        "model": model,
        "batch": batch,
        "prediction_shape": tuple(predictions.shape),
        "expected_shape": (len(train_examples), dataset["n_features"]),
    }


def _train_loop(dataset: dict, *, epochs: int, lr: float, weight_decay: float) -> None:
    """The paper's optimization loop. Guarded; not used for the alignment task."""

    model = TrajectoryFeedbackRewardPredictor(
        vocab_size=dataset["vocab_size"], n_features=dataset["n_features"]
    )
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.MSELoss()
    for fold in dataset["folds"]:
        train_examples = [dataset["examples"][i] for i in fold["train_indices"]]
        for _ in range(epochs):
            for example in train_examples:
                batch = collate_batch([example], dataset["vocab"])
                optimizer.zero_grad()
                predictions = model(
                    batch["tokens"], batch["offsets"], batch["feature_counts"]
                )
                loss = loss_fn(predictions, batch["targets"])
                loss.backward()
                optimizer.step()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feedback", type=Path, default=DEFAULT_FEEDBACK_PATH)
    parser.add_argument("--probe-states", type=Path, default=DEFAULT_PROBE_STATES_PATH)
    parser.add_argument("--min-freq", type=int, default=1)
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=0.005)
    parser.add_argument("--weight-decay", type=float, default=0.0001)
    parser.add_argument(
        "--execute-training",
        action="store_true",
        help="Cross the training boundary and actually fit the network "
        "(NOT used for the strict-alignment task; needs a real corpus).",
    )
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

    print("Route 2 inference network -- data + model assembled:")
    print(f"  examples:   {len(dataset['examples'])}")
    print(f"  features:   {dataset['n_features']}")
    print(f"  vocab size: {dataset['vocab_size']}")
    print(f"  folds:      {len(dataset['folds'])}")

    check = forward_check(dataset)
    model = check["model"]
    n_params = sum(p.numel() for p in model.parameters())
    print("  model: EmbeddingBag(vocab,30) + Linear(30+n,128) + ReLU + Linear(128,n)")
    print(f"  trainable parameters: {n_params}")
    print(
        f"  forward pass output shape: {check['prediction_shape']} "
        f"(expected {check['expected_shape']})"
    )
    assert check["prediction_shape"] == check["expected_shape"], "shape mismatch"

    if not args.execute_training:
        print(f"\n{TRAINING_BOUNDARY_MESSAGE}")
        return 0

    print("\n--execute-training set: crossing the training boundary (not part of the task).")
    _train_loop(
        dataset, epochs=args.epochs, lr=args.lr, weight_decay=args.weight_decay
    )
    print("Training loop finished.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
