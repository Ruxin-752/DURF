"""Evaluate a subgoal-conditioned executor in the Overcooked environment."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import tensorflow as tf

from durf.baseline.collect_rule_teacher_dataset import (
    SUBGOALS,
    SUBGOAL_TO_INDEX,
    action_name,
    featurize_for_policy,
    make_motion_planner,
    partner_action,
    partner_action_for_state,
    rule_teacher_decision,
    state_summary,
)
from durf.baseline.evaluate_baseline import count_event_value
from durf.baseline.runtime import make_direct_multi_env, resolve_agent_dir
from human_aware_rl.rllib.rllib import load_trainer
from overcooked_ai_py.mdp.actions import Action


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--executor", type=Path, required=True)
    parser.add_argument("--reference-agent", required=True)
    parser.add_argument("--layout", required=True)
    parser.add_argument("--policy-id", default="ppo_0")
    parser.add_argument("--teacher-player-index", type=int, choices=(0, 1), default=0)
    parser.add_argument(
        "--partner-mode",
        choices=("stay", "safe_corner", "safe_left", "safe_recycle", "dynamic_avoid"),
        default="safe_corner",
    )
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--seed", type=int, default=12000)
    parser.add_argument("--horizon", type=int, default=800)
    parser.add_argument("--success-threshold", type=float, default=20.0)
    parser.add_argument("--start-jitter-steps", type=int, default=0)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--trace-output",
        type=Path,
        help="Optional JSONL file with per-step state/subgoal/action records.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.executor.exists():
        raise FileNotFoundError(args.executor)
    if args.start_jitter_steps < 0:
        raise ValueError("--start-jitter-steps must be non-negative")
    reference_dir = resolve_agent_dir(args.reference_agent)
    trainer = load_trainer(str(reference_dir))
    motion_planner = make_motion_planner(args.layout, args.seed, args.horizon)
    rewards: list[float] = []
    lengths: list[int] = []
    event_counts_total: Counter[str] = Counter()
    subgoal_counts: Counter[str] = Counter()
    action_counts: Counter[str] = Counter()
    trace_records: list[dict] = []

    try:
        dummy_env = trainer.env_creator(trainer.config["env_config"])
        featurize_fn = dummy_env._get_featurize_fn(args.policy_id)
        # Load the Keras model after RLlib has created its TF1 graph/session.
        # Loading it before RLlib can invalidate the Keras graph in TF 2.10.
        tf.keras.backend.clear_session()
        model = tf.keras.models.load_model(args.executor)
        for episode_index in range(args.episodes):
            episode_seed = args.seed + episode_index
            episode_rng = np.random.default_rng(episode_seed)
            env = make_direct_multi_env(args.layout, seed=episode_seed, horizon=args.horizon)
            env.multi_reset()
            done = False
            reward_sum = 0.0
            steps = 0
            info = {}
            try:
                for jitter_step in range(args.start_jitter_steps):
                    teacher_move = int(
                        Action.ACTION_TO_INDEX[
                            Action.MOTION_ACTIONS[
                                episode_rng.integers(0, len(Action.MOTION_ACTIONS))
                            ]
                        ]
                    )
                    partner_move = partner_action_for_state(
                        args.partner_mode,
                        env.base_env.state,
                        motion_planner,
                        args.teacher_player_index,
                        jitter_step,
                        teacher_move,
                    )
                    joint_action = [partner_move, partner_move]
                    joint_action[args.teacher_player_index] = teacher_move
                    _, reward, done, info = env.multi_step(
                        int(joint_action[0]),
                        int(joint_action[1]),
                    )
                    reward_sum += float(reward[0])
                    if done:
                        break
                while not done:
                    state = env.base_env.state
                    subgoal_name, _ = rule_teacher_decision(
                        state,
                        motion_planner,
                        args.teacher_player_index,
                    )
                    observation = featurize_for_policy(
                        featurize_fn,
                        state,
                        args.teacher_player_index,
                    )[None, ...]
                    subgoal_id = SUBGOAL_TO_INDEX[subgoal_name]
                    subgoal_onehot = tf.keras.utils.to_categorical(
                        [subgoal_id],
                        num_classes=len(SUBGOALS),
                    )
                    logits = model.predict([observation, subgoal_onehot], verbose=0)
                    teacher_action = int(np.argmax(logits[0]))
                    partner_move = partner_action_for_state(
                        args.partner_mode,
                        state,
                        motion_planner,
                        args.teacher_player_index,
                        steps,
                        teacher_action,
                    )
                    joint_action = [partner_move, partner_move]
                    joint_action[args.teacher_player_index] = teacher_action

                    state_before = state_summary(state) if args.trace_output else None
                    _, reward, done, info = env.multi_step(
                        int(joint_action[0]),
                        int(joint_action[1]),
                    )
                    reward_sum += float(reward[0])
                    if args.trace_output:
                        trace_records.append(
                            {
                                "episode_index": episode_index,
                                "episode_seed": episode_seed,
                                "step": steps,
                                "reward_sum_after": reward_sum,
                                "subgoal": subgoal_name,
                                "subgoal_id": int(subgoal_id),
                                "teacher_player_index": args.teacher_player_index,
                                "executor_action": action_name(teacher_action),
                                "partner_action": action_name(partner_move),
                                "state_before": state_before,
                                "state_after": state_summary(env.base_env.state),
                            }
                        )
                    steps += 1
                    subgoal_counts[subgoal_name] += 1
                    action_counts[str(Action.INDEX_TO_ACTION[teacher_action])] += 1
                    if reward_sum >= args.success_threshold:
                        done = True
            finally:
                env.close()

            episode_info = info.get("episode", {}) if isinstance(info, dict) else {}
            episode_stats = episode_info.get("ep_game_stats", {})
            for event_name, value in episode_stats.items():
                event_counts_total[event_name] += count_event_value(value)
            rewards.append(reward_sum)
            lengths.append(steps)
            print(
                f"Episode {episode_index + 1:02d}/{args.episodes}: "
                f"seed={episode_seed}, reward={reward_sum:.1f}, steps={steps}"
            )
    finally:
        trainer.stop()

    result = {
        "executor": str(args.executor),
        "reference_agent": str(reference_dir),
        "layout": args.layout,
        "policy_id": args.policy_id,
        "teacher_player_index": args.teacher_player_index,
        "partner_mode": args.partner_mode,
        "episodes": args.episodes,
        "seed": args.seed,
        "horizon": args.horizon,
        "success_threshold": args.success_threshold,
        "start_jitter_steps": args.start_jitter_steps,
        "mean_reward": float(np.mean(rewards)),
        "success_rate": float(np.mean([reward >= args.success_threshold for reward in rewards])),
        "episode_rewards": rewards,
        "episode_lengths": lengths,
        "subgoal_counts": dict(subgoal_counts),
        "action_counts": dict(action_counts),
        "event_counts": dict(event_counts_total),
    }
    print(json.dumps(result, indent=2, ensure_ascii=False))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(f"Saved: {args.output}")
    if args.trace_output:
        args.trace_output.parent.mkdir(parents=True, exist_ok=True)
        with args.trace_output.open("w", encoding="utf-8") as handle:
            for record in trace_records:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        print(f"Saved trace: {args.trace_output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
