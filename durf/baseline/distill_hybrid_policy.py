"""Distill a successful hybrid policy into one PPO policy.

The teacher is a scripted prefix followed by an RLlib PPO suffix.  The student
keeps the RLlib checkpoint format, but receives supervised cross-entropy updates
on state-action pairs from successful teacher rollouts.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from durf.baseline.action_prior import (
    COUNTER_ONION_TO_POT_TOP,
    prefix_action_indices,
)
from durf.baseline.runtime import (
    make_direct_multi_env,
    load_rllib_agent,
    resolve_agent_dir,
    rllib_action_index,
)
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
    parser.add_argument("--teacher-agent", required=True)
    parser.add_argument("--layout", required=True)
    parser.add_argument("--agent-name", required=True)
    parser.add_argument("--prior", default=COUNTER_ONION_TO_POT_TOP)
    parser.add_argument("--policy-id", default="ppo_0")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--teacher-temperature", type=float, default=0.5)
    parser.add_argument("--max-examples", type=int, default=5000)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument(
        "--train-scope",
        choices=("all", "logits"),
        default="all",
        help="Which student PPO variables to update.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def featurize_for_policy(featurize_fn, state) -> np.ndarray:
    try:
        obs_pair = featurize_fn(state, debug=False)
    except TypeError:
        obs_pair = featurize_fn(state)
    return np.asarray(obs_pair[0], dtype=np.float32)


def collect_hybrid_examples(
    trainer,
    teacher_agent: str,
    layout: str,
    prior: str,
    policy_id: str,
    episodes: int,
    seed: int,
    teacher_temperature: float,
    max_examples: int,
) -> tuple[np.ndarray, np.ndarray, list[float]]:
    observations: list[np.ndarray] = []
    labels: list[int] = []
    rewards: list[float] = []
    dummy_env = trainer.env_creator(trainer.config["env_config"])
    featurize_fn = dummy_env._get_featurize_fn(policy_id)
    prefix_indices = prefix_action_indices(prior)
    stay = Action.ACTION_TO_INDEX[Action.STAY]

    teacher0 = load_rllib_agent(teacher_agent, agent_index=0)
    teacher1 = load_rllib_agent(teacher_agent, agent_index=1)

    for episode_index in range(episodes):
        env = make_direct_multi_env(layout, seed + episode_index)
        teacher0.reset()
        teacher1.reset()
        env.multi_reset()
        done = False
        reward_sum = 0.0
        episode_observations: list[np.ndarray] = []
        episode_labels: list[int] = []

        try:
            for action0 in prefix_indices:
                if done:
                    break
                state = env.base_env.state
                episode_observations.append(featurize_for_policy(featurize_fn, state))
                episode_labels.append(int(action0))
                _, reward, done, _ = env.multi_step(action0, stay)
                reward_sum += float(reward[0])

            while not done:
                state = env.base_env.state
                action0 = rllib_action_index(
                    teacher0,
                    state,
                    temperature=teacher_temperature,
                )
                action1 = rllib_action_index(
                    teacher1,
                    state,
                    temperature=teacher_temperature,
                )
                episode_observations.append(featurize_for_policy(featurize_fn, state))
                episode_labels.append(int(action0))
                _, reward, done, _ = env.multi_step(action0, action1)
                reward_sum += float(reward[0])
        finally:
            env.close()

        rewards.append(reward_sum)
        if reward_sum > 0:
            observations.extend(episode_observations)
            labels.extend(episode_labels)
            if len(labels) >= max_examples:
                break

    if not labels:
        raise RuntimeError("Teacher produced no successful episodes; cannot distill.")
    observations_array = np.stack(observations[:max_examples])
    labels_array = np.asarray(labels[:max_examples], dtype=np.int32)
    return observations_array, labels_array, rewards


def main() -> int:
    args = parse_args()
    student_dir = resolve_agent_dir(args.student_agent)
    params = load_params_for_agent(student_dir)
    trainer = load_trainer(str(student_dir))
    try:
        observations, labels, rewards = collect_hybrid_examples(
            trainer,
            args.teacher_agent,
            args.layout,
            args.prior,
            args.policy_id,
            args.episodes,
            args.seed,
            args.teacher_temperature,
            args.max_examples,
        )
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
        print(f"Teacher rewards: {rewards}")
        print(f"Collected examples: {len(labels)}")
        print(f"Label counts: {np.bincount(labels, minlength=Action.NUM_ACTIONS).tolist()}")
        print(f"Teacher temperature: {args.teacher_temperature}")
        print(f"Train scope: {args.train_scope}")
        print(f"Trained variables: {trained_variables}")
        print(f"Initial loss: {losses[0]:.6f}")
        print(f"Final loss: {losses[-1]:.6f}")
        print(f"Saved checkpoint: {checkpoint_path}")
        print(f"Installed agent: {installed}")
    finally:
        trainer.stop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
