"""Collect DAgger-style recovery labels from student-induced states.

The student PPO is allowed to run from the real reset distribution.  Failed
episodes expose the states where the current policy drifts away from a useful
opening/serving strategy.  Those student-induced states are then labeled by a
more reliable teacher: a short scripted option at the beginning, followed by a
stable RLlib suffix policy.

Outputs:

* ``dataset.npz``: observations and action labels used by supervised training.
* ``dataset.jsonl``: lightweight, human-readable metadata for inspection.
* ``metadata.json``: collection settings and summary counts.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from collections import Counter
from pathlib import Path

import dill
import numpy as np

from durf.baseline.action_prior import (
    COUNTER_ONION_TO_POT_TOP,
    prefix_action_chars,
    prefix_action_indices,
)
from durf.baseline.evaluate_baseline import count_event_value
from durf.baseline.runtime import (
    load_rllib_agent,
    make_direct_multi_env,
    resolve_agent_dir,
    rllib_action_index,
)
from human_aware_rl.rllib.rllib import load_trainer
from overcooked_ai_py.mdp.actions import Action
from overcooked_ai_py.planning.planners import MotionPlanner


REPO_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_ROOT = REPO_ROOT / "outputs" / "recovery_datasets"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--student-agent", required=True)
    parser.add_argument("--teacher-agent", required=True)
    parser.add_argument("--layout", required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--prior", default=COUNTER_ONION_TO_POT_TOP)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--seed", type=int, default=5000)
    parser.add_argument("--horizon", type=int, default=400)
    parser.add_argument("--policy-id", default="ppo_0")
    parser.add_argument("--student-temperature", type=float, default=1.0)
    parser.add_argument("--teacher-temperature", type=float, default=0.5)
    parser.add_argument(
        "--success-threshold",
        type=float,
        default=1.0,
        help="Episodes with total reward below this value are labeled as recovery data.",
    )
    parser.add_argument(
        "--include-successes",
        action="store_true",
        help="Also label successful student episodes. Defaults to failed episodes only.",
    )
    parser.add_argument(
        "--max-examples",
        type=int,
        default=8000,
        help="Maximum labeled state-action pairs to save.",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=1,
        help="Keep one labeled state every N timesteps from selected episodes.",
    )
    parser.add_argument(
        "--disagreement-only",
        action="store_true",
        help="Keep only states where the teacher action differs from student action0.",
    )
    parser.add_argument(
        "--max-label-fraction",
        type=float,
        default=0.6,
        help=(
            "Soft cap for any one action label in the saved dataset. "
            "Use 1.0 to disable. This prevents recovery data from becoming mostly STAY."
        ),
    )
    parser.add_argument(
        "--labeler",
        choices=("task_rule", "teacher_policy", "prefix_then_teacher"),
        default="task_rule",
        help=(
            "How to label student-induced states. 'task_rule' uses a motion-planner "
            "subtask scaffold before falling back to the teacher policy."
        ),
    )
    return parser.parse_args()


def load_params_for_agent(agent_dir: Path) -> dict:
    config_path = agent_dir.parent / "config.pkl"
    if not config_path.exists():
        raise FileNotFoundError(f"Training config not found: {config_path}")
    with config_path.open("rb") as handle:
        return dill.load(handle)


def featurize_for_policy(featurize_fn, state) -> np.ndarray:
    try:
        obs_pair = featurize_fn(state, debug=False)
    except TypeError:
        obs_pair = featurize_fn(state)
    return np.asarray(obs_pair[0], dtype=np.float32)


def object_name(obj) -> str | None:
    if obj is None:
        return None
    return getattr(obj, "name", str(obj))


def state_summary(state) -> dict:
    """Small JSON-safe snapshot for dataset inspection."""
    players = []
    for player in state.players:
        players.append(
            {
                "position": list(player.position),
                "orientation": list(player.orientation),
                "held_object": object_name(player.held_object),
            }
        )
    objects = []
    for position, obj in sorted(state.objects.items()):
        objects.append(
            {
                "position": list(position),
                "name": object_name(obj),
                "is_ready": bool(getattr(obj, "is_ready", False)),
                "is_cooking": bool(getattr(obj, "is_cooking", False)),
            }
        )
    return {
        "timestep": int(state.timestep),
        "players": players,
        "objects": objects,
    }


def collect_student_episode(
    student0,
    student1,
    layout: str,
    seed: int,
    horizon: int,
    student_temperature: float,
) -> tuple[float, list[dict], Counter[str]]:
    env = make_direct_multi_env(layout, seed, horizon=horizon)
    student0.reset()
    student1.reset()
    env.multi_reset()
    done = False
    reward_sum = 0.0
    records: list[dict] = []
    info = {}

    try:
        while not done:
            state = copy.deepcopy(env.base_env.state)
            action0 = rllib_action_index(
                student0,
                env.base_env.state,
                temperature=student_temperature,
            )
            action1 = rllib_action_index(
                student1,
                env.base_env.state,
                temperature=student_temperature,
            )
            _, reward, done, info = env.multi_step(action0, action1)
            reward_value = float(reward[0])
            reward_sum += reward_value
            records.append(
                {
                    "state": state,
                    "step": len(records),
                    "student_action0": int(action0),
                    "student_action1": int(action1),
                    "reward": reward_value,
                    "done": bool(done),
                }
            )
    finally:
        env.close()

    event_counts: Counter[str] = Counter()
    episode_info = info.get("episode", {}) if isinstance(info, dict) else {}
    episode_stats = episode_info.get("ep_game_stats", {})
    for event_name, value in episode_stats.items():
        event_counts[event_name] += count_event_value(value)
    return reward_sum, records, event_counts


def teacher_label_for_state(
    teacher0,
    state,
    step: int,
    prefix_indices: list[int],
    teacher_temperature: float,
) -> int:
    if step < len(prefix_indices):
        return int(prefix_indices[step])
    return rllib_action_index(teacher0, state, temperature=teacher_temperature)


def first_action_to_feature(
    motion_planner: MotionPlanner,
    player,
    feature_positions: list[tuple[int, int]],
) -> int | None:
    best_plan = None
    best_cost = float("inf")
    for feature_pos in feature_positions:
        for goal in motion_planner.motion_goals_for_pos.get(feature_pos, []):
            start = player.pos_and_or
            if not motion_planner.is_valid_motion_start_goal_pair(start, goal):
                continue
            action_plan, _, cost = motion_planner.get_plan(start, goal)
            if action_plan and cost < best_cost:
                best_plan = action_plan
                best_cost = cost
    if not best_plan:
        return None
    return int(Action.ACTION_TO_INDEX[best_plan[0]])


def object_positions(state, name: str) -> list[tuple[int, int]]:
    return [
        position
        for position, obj in state.objects.items()
        if getattr(obj, "name", None) == name
    ]


def target_recipe(mdp) -> list[str]:
    """Return the first active order as a mutable ingredient list."""
    if not mdp.start_all_orders:
        return ["onion", "onion", "onion"]
    order = mdp.start_all_orders[0]
    if isinstance(order, dict):
        return list(order.get("ingredients", []))
    return list(getattr(order, "ingredients", order))


def pot_ingredients(state, pot_pos: tuple[int, int]) -> list[str]:
    if not state.has_object(pot_pos):
        return []
    obj = state.get_object(pot_pos)
    if getattr(obj, "name", None) != "soup":
        return []
    return list(getattr(obj, "ingredients", []))


def missing_ingredients_for_pot(
    recipe: list[str],
    current_ingredients: list[str],
) -> list[str]:
    missing = Counter(recipe)
    missing.subtract(Counter(current_ingredients))
    ordered_missing: list[str] = []
    for ingredient in recipe:
        if missing[ingredient] > 0:
            ordered_missing.append(ingredient)
            missing[ingredient] -= 1
    return ordered_missing


def ingredient_dispenser_locations(mdp, ingredient: str) -> list[tuple[int, int]]:
    if ingredient == "tomato":
        return mdp.get_tomato_dispenser_locations()
    if ingredient == "onion":
        return mdp.get_onion_dispenser_locations()
    return []


def pots_needing_ingredient(
    state,
    mdp,
    ingredient: str,
) -> list[tuple[int, int]]:
    recipe = target_recipe(mdp)
    targets = []
    for pot_pos in mdp.get_pot_locations():
        if state.has_object(pot_pos):
            obj = state.get_object(pot_pos)
            if getattr(obj, "name", None) != "soup":
                continue
            if getattr(obj, "is_ready", False) or getattr(obj, "is_cooking", False):
                continue
        missing = missing_ingredients_for_pot(recipe, pot_ingredients(state, pot_pos))
        if ingredient in missing:
            targets.append(pot_pos)
    return targets


def next_needed_ingredient(state, mdp) -> str | None:
    recipe = target_recipe(mdp)
    best_missing: list[str] = []
    best_filled = -1
    for pot_pos in mdp.get_pot_locations():
        if state.has_object(pot_pos):
            obj = state.get_object(pot_pos)
            if getattr(obj, "name", None) != "soup":
                continue
            if getattr(obj, "is_ready", False) or getattr(obj, "is_cooking", False):
                continue
        current = pot_ingredients(state, pot_pos)
        missing = missing_ingredients_for_pot(recipe, current)
        if missing and len(current) > best_filled:
            best_missing = missing
            best_filled = len(current)
    return best_missing[0] if best_missing else None


def task_rule_label(state, motion_planner: MotionPlanner) -> int | None:
    """Motion-planner scaffold for the target cooking chain.

    It labels broad recovery subtasks instead of blindly replaying a fixed
    prefix.  This is intentionally conservative: if the rule cannot identify a
    useful subtask, the caller should fall back to the RLlib teacher policy.
    """
    mdp = motion_planner.mdp
    player = state.players[0]
    held = player.held_object
    pot_states = mdp.get_pot_states(state)

    if held is None:
        ready_pots = mdp.get_ready_pots(pot_states)
        if ready_pots:
            return first_action_to_feature(
                motion_planner,
                player,
                mdp.get_dish_dispenser_locations(),
            )
        needed = next_needed_ingredient(state, mdp)
        if needed is None:
            return None
        counter_positions = object_positions(state, needed)
        if counter_positions:
            return first_action_to_feature(motion_planner, player, counter_positions)
        dispenser_action = first_action_to_feature(
            motion_planner,
            player,
            ingredient_dispenser_locations(mdp, needed),
        )
        if dispenser_action is not None:
            return dispenser_action
        return None

    held_name = getattr(held, "name", None)
    if held_name in ("onion", "tomato"):
        pot_targets = pots_needing_ingredient(state, mdp, held_name)
        if pot_targets:
            return first_action_to_feature(motion_planner, player, pot_targets)
        return None

    if held_name == "dish":
        ready_pots = mdp.get_ready_pots(pot_states)
        if ready_pots:
            return first_action_to_feature(motion_planner, player, ready_pots)
        return None

    if held_name == "soup":
        return first_action_to_feature(
            motion_planner,
            player,
            mdp.get_serving_locations(),
        )
    return None


def label_for_state(
    labeler: str,
    teacher0,
    state,
    step: int,
    prefix_indices: list[int],
    teacher_temperature: float,
    motion_planner: MotionPlanner,
) -> int:
    if labeler == "task_rule":
        task_action = task_rule_label(state, motion_planner)
        if task_action is not None:
            return task_action
    if labeler == "prefix_then_teacher":
        return teacher_label_for_state(
            teacher0,
            state,
            step,
            prefix_indices,
            teacher_temperature,
        )
    return rllib_action_index(teacher0, state, temperature=teacher_temperature)


def main() -> int:
    args = parse_args()
    if args.episodes <= 0:
        raise ValueError("--episodes must be positive")
    if args.stride <= 0:
        raise ValueError("--stride must be positive")

    output_dir = args.output_dir or (
        OUTPUT_ROOT
        / f"{args.layout}_{args.student_agent}_seed{args.seed}_n{args.episodes}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    student_dir = resolve_agent_dir(args.student_agent)
    trainer = load_trainer(str(student_dir))
    student0 = load_rllib_agent(args.student_agent, agent_index=0)
    student1 = load_rllib_agent(args.student_agent, agent_index=1)
    teacher0 = load_rllib_agent(args.teacher_agent, agent_index=0)

    dummy_env = trainer.env_creator(trainer.config["env_config"])
    featurize_fn = dummy_env._get_featurize_fn(args.policy_id)
    prefix_indices = prefix_action_indices(args.prior)
    planner_env = make_direct_multi_env(args.layout, args.seed, horizon=args.horizon)
    mdp = planner_env.base_env.mdp
    counter_goals = []
    if mdp.start_state is not None:
        counter_goals = [
            position
            for position, obj in mdp.start_state.objects.items()
            if mdp.get_terrain_type_at_pos(position) == "X"
        ]
    motion_planner = MotionPlanner.from_pickle_or_compute(
        mdp,
        counter_goals=counter_goals,
        info=False,
    )
    planner_env.close()

    observations: list[np.ndarray] = []
    labels: list[int] = []
    label_counts: Counter[int] = Counter()
    jsonl_records: list[dict] = []
    episode_summaries: list[dict] = []
    event_counts_total: Counter[str] = Counter()

    try:
        for episode_index in range(args.episodes):
            episode_seed = args.seed + episode_index
            reward_sum, records, event_counts = collect_student_episode(
                student0,
                student1,
                args.layout,
                episode_seed,
                args.horizon,
                args.student_temperature,
            )
            event_counts_total.update(event_counts)
            selected = args.include_successes or reward_sum < args.success_threshold
            episode_summaries.append(
                {
                    "episode_index": episode_index,
                    "seed": episode_seed,
                    "reward": reward_sum,
                    "selected_for_recovery": selected,
                    "steps": len(records),
                    "event_counts": dict(event_counts),
                }
            )
            print(
                f"Episode {episode_index + 1:02d}/{args.episodes}: "
                f"seed={episode_seed}, reward={reward_sum:.1f}, "
                f"selected={selected}, examples={len(labels)}"
            )
            if not selected:
                continue

            for record in records[:: args.stride]:
                if len(labels) >= args.max_examples:
                    break
                state = record["state"]
                teacher_action = label_for_state(
                    args.labeler,
                    teacher0,
                    state,
                    int(record["step"]),
                    prefix_indices,
                    args.teacher_temperature,
                    motion_planner,
                )
                if args.disagreement_only and teacher_action == int(record["student_action0"]):
                    continue
                if args.max_label_fraction < 1.0 and labels:
                    projected_count = label_counts[teacher_action] + 1
                    projected_total = len(labels) + 1
                    if projected_count / projected_total > args.max_label_fraction:
                        continue
                observations.append(featurize_for_policy(featurize_fn, state))
                labels.append(teacher_action)
                label_counts[teacher_action] += 1
                jsonl_records.append(
                    {
                        "episode_index": episode_index,
                        "seed": episode_seed,
                        "step": int(record["step"]),
                        "student_action0": Action.INDEX_TO_ACTION[
                            int(record["student_action0"])
                        ],
                        "student_action1": Action.INDEX_TO_ACTION[
                            int(record["student_action1"])
                        ],
                        "teacher_action0": Action.INDEX_TO_ACTION[int(teacher_action)],
                        "reward": float(record["reward"]),
                        "episode_reward": reward_sum,
                        "state": state_summary(state),
                    }
                )
            if len(labels) >= args.max_examples:
                break
    finally:
        trainer.stop()

    if not labels:
        raise RuntimeError("No recovery examples collected.")

    observations_array = np.stack(observations).astype(np.float32)
    labels_array = np.asarray(labels, dtype=np.int32)
    np.savez_compressed(
        output_dir / "dataset.npz",
        observations=observations_array,
        labels=labels_array,
    )
    with (output_dir / "dataset.jsonl").open("w", encoding="utf-8") as handle:
        for record in jsonl_records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    metadata = {
        "student_agent": str(student_dir),
        "teacher_agent": str(resolve_agent_dir(args.teacher_agent)),
        "layout": args.layout,
        "prior": args.prior,
        "prefix_actions": prefix_action_chars(args.prior),
        "episodes": args.episodes,
        "seed": args.seed,
        "horizon": args.horizon,
        "student_temperature": args.student_temperature,
        "teacher_temperature": args.teacher_temperature,
        "success_threshold": args.success_threshold,
        "include_successes": args.include_successes,
        "stride": args.stride,
        "disagreement_only": args.disagreement_only,
        "max_label_fraction": args.max_label_fraction,
        "labeler": args.labeler,
        "num_examples": int(labels_array.shape[0]),
        "label_counts": np.bincount(labels_array, minlength=Action.NUM_ACTIONS).tolist(),
        "event_counts": dict(event_counts_total),
        "episodes_summary": episode_summaries,
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"Saved dataset: {output_dir / 'dataset.npz'}")
    print(f"Saved records: {output_dir / 'dataset.jsonl'}")
    print(f"Saved metadata: {output_dir / 'metadata.json'}")
    print(f"Label counts: {metadata['label_counts']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
