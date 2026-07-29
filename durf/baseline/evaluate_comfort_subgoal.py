"""Round-level comparison: H0 pure rules vs H0 + learned-comfort subgoals.

Both agents play the AI (blue) seat next to the SAME fixed rule partner (the
H0 rule driving the green human), on the ring layout, for a fixed horizon. We
report task outcome (env soup reward) and *realized* comfort judged by the gold
teacher ``w*`` comfort partition -- the ground-truth rule, NOT the learned
weights the comfort agent optimizes -- so the comparison is not circular.

    python durf/baseline/evaluate_comfort_subgoal.py --horizon 400

Runs headless (overcooked + motion planner only; no pygame, ray or torch).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from durf.baseline.comfort_reward import (
    ADAPTED_ROOT,
    context_from_state,
)
from durf.baseline.comfort_subgoal_agent import ComfortSubgoalAgent, load_comfort_weights
from durf.baseline.coordination import StallBreaker
from durf.baseline.h0_planner import make_motion_planner, rule_teacher_decision
from durf.baseline.runtime import RING_TOMATO_ONION_H0_LAYOUT, make_direct_multi_env

from src.belief_model import score_action  # noqa: E402
from src.subgoal_featurizer import featurize_subgoal  # noqa: E402
from src.subgoal_reranker import COMFORT_FEATURES  # noqa: E402

GOLD_WEIGHTS_PATH = ADAPTED_ROOT / "data" / "gold_comfort_weights.json"
# Comfort features whose activation means the AI is inconveniencing the human.
DISCOMFORT_FEATURES = frozenset(
    {
        "blocks_human_path",
        "blocks_serving_route",
        "blocks_partner_on_ring",
        "duplicate_human_task",
        "steals_human_target",
        "crowds_human_target",
        "cuts_in_front_of_human",
        "collision_risk",
        "human_wait_cost",
        "frustrates_human",
    }
)


def _gold_comfort_judge() -> dict[str, float]:
    weights = load_comfort_weights(GOLD_WEIGHTS_PATH)
    return {f: v for f, v in weights.items() if f in COMFORT_FEATURES}


def rollout(kind: str, layout: str, seed: int, horizon: int, lambda_pref: float) -> dict:
    """Run one episode; return task + realized-comfort metrics for the AI seat."""

    motion_planner = make_motion_planner(layout, seed, horizon)
    mdp = motion_planner.mdp
    env = make_direct_multi_env(layout, seed, horizon=horizon)
    env.multi_reset()

    judge = _gold_comfort_judge()
    breaker = StallBreaker(mdp, seed=seed)
    agent = None
    if kind == "comfort_subgoal":
        agent = ComfortSubgoalAgent(
            motion_planner, lambda_pref=lambda_pref, ai_index=0
        )

    total_reward = 0.0
    comfort_sum = 0.0
    discomfort_steps = 0
    subgoal_counts: dict[str, int] = {}
    wait_streak = 0
    longest_wait_streak = 0
    wait_guard_steps = 0
    steps = 0

    for _ in range(horizon):
        state = env.base_env.state
        # Fixed rule partner in the human (green) seat.
        _, human_action = rule_teacher_decision(state, motion_planner, player_index=1)
        if kind == "h0_rule":
            ai_subgoal, ai_action = rule_teacher_decision(
                state, motion_planner, player_index=0
            )
        else:
            ai_action, ai_subgoal = agent.act(state)
            wait_guard_steps += int(bool(agent.last_decision.get("wait_guard_applied")))

        # Judge realized comfort of the AI's executed subgoal with gold w*.
        context = context_from_state(state, mdp, ai_index=0)
        phi = featurize_subgoal(context, ai_subgoal)
        comfort_phi = {f: v for f, v in phi.items() if f in COMFORT_FEATURES}
        comfort_sum += float(score_action(judge, comfort_phi))
        if any(phi.get(f, 0.0) > 0 for f in DISCOMFORT_FEATURES):
            discomfort_steps += 1
        subgoal_counts[ai_subgoal] = subgoal_counts.get(ai_subgoal, 0) + 1
        wait_streak = wait_streak + 1 if ai_subgoal == "WAIT" else 0
        longest_wait_streak = max(longest_wait_streak, wait_streak)

        ai_action, _ = breaker.resolve(state, 0, ai_action, other_index=1)
        human_action, _ = breaker.resolve(state, 1, human_action, other_index=0)
        (_, _), (reward, _), done, _ = env.multi_step(int(ai_action), int(human_action))
        total_reward += float(reward)
        steps += 1
        if done:
            break

    return {
        "kind": kind,
        "steps": steps,
        "soup_reward": total_reward,
        "comfort_sum": comfort_sum,
        "comfort_per_step": comfort_sum / steps if steps else 0.0,
        "discomfort_steps": discomfort_steps,
        "discomfort_rate": discomfort_steps / steps if steps else 0.0,
        "subgoal_counts": subgoal_counts,
        "longest_wait_streak": longest_wait_streak,
        "wait_guard_steps": wait_guard_steps,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--layout", default=RING_TOMATO_ONION_H0_LAYOUT)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--horizon", type=int, default=400)
    parser.add_argument("--comfort-lambda", type=float, default=1.0)
    parser.add_argument(
        "--output",
        type=Path,
        default=ADAPTED_ROOT / "outputs" / "route2" / "comfort_subgoal_eval.json",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    baseline = rollout(
        "h0_rule", args.layout, args.seed, args.horizon, args.comfort_lambda
    )
    comfort = rollout(
        "comfort_subgoal", args.layout, args.seed, args.horizon, args.comfort_lambda
    )

    summary = {
        "layout": args.layout,
        "seed": args.seed,
        "horizon": args.horizon,
        "lambda_pref": args.comfort_lambda,
        "h0_rule": baseline,
        "comfort_subgoal": comfort,
        "delta": {
            "soup_reward": comfort["soup_reward"] - baseline["soup_reward"],
            "comfort_per_step": comfort["comfort_per_step"]
            - baseline["comfort_per_step"],
            "discomfort_rate": comfort["discomfort_rate"]
            - baseline["discomfort_rate"],
        },
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    def fmt(row: dict) -> str:
        return (
            f"  soup_reward={row['soup_reward']:.1f}  "
            f"comfort/step={row['comfort_per_step']:+.3f}  "
            f"discomfort_rate={row['discomfort_rate']:.3f}"
        )

    print(f"Layout {args.layout} | horizon {args.horizon} | lambda {args.comfort_lambda}")
    print("H0 pure rules:")
    print(fmt(baseline))
    print(f"  subgoals: {baseline['subgoal_counts']}")
    print("H0 + learned comfort subgoals:")
    print(fmt(comfort))
    print(f"  subgoals: {comfort['subgoal_counts']}")
    print("Delta (comfort - baseline):")
    print(
        f"  soup_reward={summary['delta']['soup_reward']:+.1f}  "
        f"comfort/step={summary['delta']['comfort_per_step']:+.3f}  "
        f"discomfort_rate={summary['delta']['discomfort_rate']:+.3f}"
    )
    print(f"Wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
