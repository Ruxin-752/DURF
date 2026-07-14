"""Supervise a PPO checkpoint on a collected recovery dataset."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from durf.baseline.runtime import resolve_agent_dir
from durf.baseline.supervise_prefix_policy import (
    OUTPUT_ROOT,
    install_supervised_agent,
    load_params_for_agent,
    supervise_policy,
)
from human_aware_rl.rllib.rllib import load_trainer, save_trainer
from overcooked_ai_py.mdp.actions import Action


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--student-agent", required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--agent-name", required=True)
    parser.add_argument("--policy-id", default="ppo_0")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--learning-rate", type=float, default=5e-6)
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument(
        "--train-scope",
        choices=("all", "logits"),
        default="logits",
        help="Default is logits to reduce damage to the PPO representation.",
    )
    parser.add_argument("--max-examples", type=int)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.dataset.exists():
        raise FileNotFoundError(args.dataset)

    data = np.load(args.dataset)
    observations = data["observations"].astype(np.float32)
    labels = data["labels"].astype(np.int32)
    if args.max_examples:
        observations = observations[: args.max_examples]
        labels = labels[: args.max_examples]
    if observations.shape[0] != labels.shape[0]:
        raise ValueError(
            f"Observation/label length mismatch: {observations.shape[0]} vs {labels.shape[0]}"
        )

    student_dir = resolve_agent_dir(args.student_agent)
    params = load_params_for_agent(student_dir)
    trainer = load_trainer(str(student_dir))
    try:
        losses, trained_variables = supervise_policy(
            trainer,
            args.policy_id,
            observations,
            labels,
            args.epochs,
            args.learning_rate,
            args.repeat,
            args.train_scope,
        )
        run_dir = OUTPUT_ROOT / args.agent_name
        run_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_path = Path(save_trainer(trainer, params, str(run_dir)))
        installed = install_supervised_agent(
            run_dir,
            checkpoint_path,
            args.agent_name,
            args.overwrite,
        )
        result = {
            "student_agent": str(student_dir),
            "dataset": str(args.dataset),
            "agent_name": args.agent_name,
            "policy_id": args.policy_id,
            "examples": int(labels.shape[0]),
            "label_counts": np.bincount(labels, minlength=Action.NUM_ACTIONS).tolist(),
            "epochs": args.epochs,
            "learning_rate": args.learning_rate,
            "repeat": args.repeat,
            "train_scope": args.train_scope,
            "trained_variables": trained_variables,
            "initial_loss": losses[0],
            "final_loss": losses[-1],
            "checkpoint": str(checkpoint_path),
            "installed_agent": str(installed),
        }
        (run_dir / "recovery_training_summary.json").write_text(
            json.dumps(result, indent=2),
            encoding="utf-8",
        )
        print(json.dumps(result, indent=2))
    finally:
        trainer.stop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
