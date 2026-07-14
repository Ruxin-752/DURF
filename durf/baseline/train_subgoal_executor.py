"""Train a subgoal-conditioned low-level executor.

Input dataset format:

* observations: float32 [N, H, W, C]
* subgoals: int32 [N]
* labels: int32 [N], Overcooked action indices

The model learns ``executor(observation, subgoal) -> action``.  This is a
standalone warm-start diagnostic model, separate from the archived RLlib PPO
checkpoint format.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import tensorflow as tf

from durf.baseline.collect_rule_teacher_dataset import SUBGOALS
from overcooked_ai_py.mdp.actions import Action


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def build_model(
    observation_shape: tuple[int, int, int],
    num_subgoals: int,
    learning_rate: float,
) -> tf.keras.Model:
    obs_input = tf.keras.Input(shape=observation_shape, name="observation")
    subgoal_input = tf.keras.Input(shape=(num_subgoals,), name="subgoal")

    x = tf.keras.layers.Conv2D(32, 3, padding="same", activation="relu")(obs_input)
    x = tf.keras.layers.Conv2D(32, 3, padding="same", activation="relu")(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Concatenate()([x, subgoal_input])
    x = tf.keras.layers.Dense(128, activation="relu")(x)
    x = tf.keras.layers.Dense(64, activation="relu")(x)
    logits = tf.keras.layers.Dense(Action.NUM_ACTIONS, name="action_logits")(x)

    model = tf.keras.Model(inputs=[obs_input, subgoal_input], outputs=logits)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )
    return model


def main() -> int:
    args = parse_args()
    if not args.dataset.exists():
        raise FileNotFoundError(args.dataset)
    if not 0.0 <= args.validation_fraction < 1.0:
        raise ValueError("--validation-fraction must be in [0, 1)")

    tf.keras.utils.set_random_seed(args.seed)
    data = np.load(args.dataset)
    observations = data["observations"].astype(np.float32)
    labels = data["labels"].astype(np.int32)
    if "subgoals" not in data:
        raise KeyError(f"{args.dataset} does not contain 'subgoals'")
    subgoals = data["subgoals"].astype(np.int32)
    if observations.shape[0] != labels.shape[0] or labels.shape[0] != subgoals.shape[0]:
        raise ValueError("observations, labels, and subgoals must have the same length")

    num_subgoals = len(SUBGOALS)
    subgoal_onehot = tf.keras.utils.to_categorical(subgoals, num_classes=num_subgoals)
    rng = np.random.default_rng(args.seed)
    order = rng.permutation(labels.shape[0])
    observations = observations[order]
    labels = labels[order]
    subgoal_onehot = subgoal_onehot[order]

    val_size = int(round(labels.shape[0] * args.validation_fraction))
    if val_size:
        train_slice = slice(0, labels.shape[0] - val_size)
        val_slice = slice(labels.shape[0] - val_size, labels.shape[0])
        validation_data = (
            [observations[val_slice], subgoal_onehot[val_slice]],
            labels[val_slice],
        )
    else:
        train_slice = slice(0, labels.shape[0])
        validation_data = None

    model = build_model(
        tuple(observations.shape[1:]),
        num_subgoals,
        args.learning_rate,
    )
    history = model.fit(
        [observations[train_slice], subgoal_onehot[train_slice]],
        labels[train_slice],
        validation_data=validation_data,
        epochs=args.epochs,
        batch_size=args.batch_size,
        verbose=2,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    model.save(args.output_dir / "executor.keras")
    predictions = model.predict(
        [observations, subgoal_onehot],
        batch_size=args.batch_size,
        verbose=0,
    ).argmax(axis=1)
    accuracy = float((predictions == labels).mean())
    metadata = {
        "dataset": str(args.dataset),
        "output_model": str(args.output_dir / "executor.keras"),
        "examples": int(labels.shape[0]),
        "observation_shape": list(observations.shape),
        "subgoals": list(SUBGOALS),
        "subgoal_counts": {
            name: int((subgoals == index).sum())
            for index, name in enumerate(SUBGOALS)
        },
        "label_counts": {
            str(Action.INDEX_TO_ACTION[index]): int((labels == index).sum())
            for index in range(Action.NUM_ACTIONS)
        },
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "validation_fraction": args.validation_fraction,
        "train_history": {
            key: [float(value) for value in values]
            for key, values in history.history.items()
        },
        "full_dataset_accuracy": accuracy,
    }
    (args.output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(metadata, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
