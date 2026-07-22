"""Play a few rounds with a DeepSeek "human" beside each AI, and compare.

The DeepSeek partner reasons at the subgoal level (see ``deepseek_human.py``),
so it actually divides labor instead of deadlocking like a mirror rule agent.
We run the same partner against two AIs -- plain H0 rules and H0 + learned
comfort subgoals -- and report task outcome plus gold-judged comfort, with a
readable per-turn transcript of what the human "said" and chose.

Requires ``DEEPSEEK_API_KEY`` (see durf/group_a/deepseek_chat.py). Runs headless
(overcooked + motion planner only; no pygame/ray/torch).

    python durf/baseline/play_deepseek_rounds.py --horizon 150 --episodes 1
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from durf.baseline.comfort_reward import ADAPTED_ROOT, context_from_state
from durf.baseline.comfort_subgoal_agent import ComfortSubgoalAgent
from durf.baseline.coordination import StallBreaker
from durf.baseline.deepseek_human import DeepSeekHumanAgent
from durf.baseline.evaluate_comfort_subgoal import DISCOMFORT_FEATURES, _gold_comfort_judge
from durf.baseline.h0_planner import make_motion_planner, rule_teacher_decision
from durf.baseline.runtime import RING_TOMATO_ONION_H0_LAYOUT, make_direct_multi_env

from src.belief_model import score_action  # noqa: E402
from src.subgoal_featurizer import featurize_subgoal  # noqa: E402
from src.subgoal_reranker import COMFORT_FEATURES  # noqa: E402

DATASET_VERSION = 1
ACTION_NAMES = ("north", "south", "east", "west", "stay", "interact")


def _held(player) -> str | None:
    obj = getattr(player, "held_object", None)
    return getattr(obj, "name", None) if obj is not None else None


def _ai_policy(kind: str, motion_planner, lambda_pref: float):
    if kind == "comfort_subgoal":
        agent = ComfortSubgoalAgent(
            motion_planner, lambda_pref=lambda_pref, ai_index=0
        )
        return agent.act
    if kind == "h0_rule":

        def act(state):
            subgoal, action = rule_teacher_decision(
                state, motion_planner, player_index=0
            )
            return int(action), subgoal

        return act
    raise ValueError(f"unknown AI kind: {kind}")


def rollout(
    ai_kind: str,
    layout: str,
    seed: int,
    horizon: int,
    lambda_pref: float,
    replan_every: int,
    verbose: bool,
    episode: int = 0,
    record_fn=None,
) -> dict:
    motion_planner = make_motion_planner(layout, seed, horizon)
    mdp = motion_planner.mdp
    env = make_direct_multi_env(layout, seed, horizon=horizon)
    env.multi_reset()

    ai_act = _ai_policy(ai_kind, motion_planner, lambda_pref)
    human = DeepSeekHumanAgent(
        motion_planner, player_index=1, replan_every=replan_every
    )
    judge = _gold_comfort_judge()
    breaker = StallBreaker(mdp, seed=seed)

    total_reward = 0.0
    comfort_sum = 0.0
    discomfort_steps = 0
    steps = 0
    transcript: list[dict] = []
    last_human_subgoal = None

    for step in range(horizon):
        state = env.base_env.state
        human_action, human_subgoal = human.act(state)
        ai_action, ai_subgoal = ai_act(state)

        context = context_from_state(state, mdp, ai_index=0)
        phi = featurize_subgoal(context, ai_subgoal)
        comfort_phi = {f: v for f, v in phi.items() if f in COMFORT_FEATURES}
        comfort_sum += float(score_action(judge, comfort_phi))
        if any(phi.get(f, 0.0) > 0 for f in DISCOMFORT_FEATURES):
            discomfort_steps += 1

        fresh_utterance = human_subgoal != last_human_subgoal
        if fresh_utterance:
            entry = {
                "step": step,
                "human_subgoal": human_subgoal,
                "human_say": human.last_say,
                "human_reason": human.last_reason,
                "ai_subgoal": ai_subgoal,
            }
            transcript.append(entry)
            if verbose:
                say = f' "{human.last_say}"' if human.last_say else ""
                print(
                    f"  t={step:3d} | human:{human_subgoal}{say} "
                    f"[{human.last_reason}] | ai:{ai_subgoal}"
                )
            last_human_subgoal = human_subgoal

        ai_action, ai_nudged = breaker.resolve(state, 0, ai_action, other_index=1)
        human_action, human_nudged = breaker.resolve(
            state, 1, human_action, other_index=0
        )

        if record_fn is not None:
            gold_comfort = float(score_action(judge, comfort_phi))
            record_fn(
                {
                    "dataset_version": DATASET_VERSION,
                    "layout": layout,
                    "seed": seed,
                    "episode": episode,
                    "step": step,
                    "ai_kind": ai_kind,
                    "lambda_pref": lambda_pref,
                    "recipe": list(context.recipe),
                    "pot_ingredients": list(context.pot_ingredients),
                    "pot_status": context.pot_status,
                    "human_intent": context.human_intent,
                    "ai": {
                        "pos": list(state.players[0].position),
                        "held": _held(state.players[0]),
                        "subgoal": ai_subgoal,
                        "action": int(ai_action),
                        "action_name": ACTION_NAMES[int(ai_action)],
                        "nudged": bool(ai_nudged),
                    },
                    "human": {
                        "pos": list(state.players[1].position),
                        "held": _held(state.players[1]),
                        "subgoal": human_subgoal,
                        "action": int(human_action),
                        "action_name": ACTION_NAMES[int(human_action)],
                        "nudged": bool(human_nudged),
                        # Language + provenance. ``say`` is only fresh at replan
                        # steps; ``utterance_is_fresh`` flags those for language
                        # training (state, utterance, human_subgoal) triples.
                        "say": human.last_say,
                        "utterance_is_fresh": bool(fresh_utterance),
                        "source": human.last_reason,
                    },
                    "ai_comfort_features": {
                        f: v for f, v in comfort_phi.items() if v
                    },
                    "gold_comfort_score": gold_comfort,
                    "discomfort": any(
                        phi.get(f, 0.0) > 0 for f in DISCOMFORT_FEATURES
                    ),
                }
            )

        (_, _), (reward, _), done, _ = env.multi_step(int(ai_action), int(human_action))
        total_reward += float(reward)
        steps += 1
        if done:
            break

    return {
        "ai_kind": ai_kind,
        "steps": steps,
        "soup_reward": total_reward,
        "comfort_per_step": comfort_sum / steps if steps else 0.0,
        "discomfort_rate": discomfort_steps / steps if steps else 0.0,
        "transcript": transcript,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--layout", default=RING_TOMATO_ONION_H0_LAYOUT)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--horizon", type=int, default=150)
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--comfort-lambda", type=float, default=1.0)
    parser.add_argument("--replan-every", type=int, default=6)
    parser.add_argument(
        "--ai-modes",
        nargs="+",
        default=["h0_rule", "comfort_subgoal"],
        choices=["h0_rule", "comfort_subgoal"],
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ADAPTED_ROOT / "outputs" / "route2" / "deepseek_rounds.json",
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=ADAPTED_ROOT / "data" / "deepseek_play" / "trajectories.jsonl",
        help="Per-step JSONL trajectory log (APPENDED across runs) for training.",
    )
    parser.add_argument(
        "--no-dataset",
        dest="dataset",
        action="store_const",
        const=None,
        help="Disable per-step dataset logging.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not os.getenv("DEEPSEEK_API_KEY"):
        print(
            "DEEPSEEK_API_KEY is not set. Set it first, e.g.:\n"
            '  $env:DEEPSEEK_API_KEY="sk-..."\n'
            "The human partner would otherwise fall back to rules (defeating the "
            "purpose of a DeepSeek human)."
        )
        return 2

    dataset_handle = None
    dataset_count = 0
    record_fn = None
    if args.dataset is not None:
        args.dataset.parent.mkdir(parents=True, exist_ok=True)
        dataset_handle = args.dataset.open("a", encoding="utf-8")

        def record_fn(record: dict) -> None:
            nonlocal dataset_count
            dataset_handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            dataset_count += 1

    results: dict[str, list[dict]] = {kind: [] for kind in args.ai_modes}
    try:
        for episode in range(args.episodes):
            for kind in args.ai_modes:
                print(f"\n=== episode {episode} | AI={kind} ===")
                row = rollout(
                    kind,
                    args.layout,
                    args.seed + episode,
                    args.horizon,
                    args.comfort_lambda,
                    args.replan_every,
                    verbose=True,
                    episode=episode,
                    record_fn=record_fn,
                )
                results[kind].append(row)
                print(
                    f"  -> soup_reward={row['soup_reward']:.1f} "
                    f"comfort/step={row['comfort_per_step']:+.3f} "
                    f"discomfort_rate={row['discomfort_rate']:.3f}"
                )
                if dataset_handle is not None:
                    dataset_handle.flush()
    finally:
        if dataset_handle is not None:
            dataset_handle.close()

    def mean(kind: str, key: str) -> float:
        rows = results[kind]
        return sum(r[key] for r in rows) / len(rows) if rows else 0.0

    summary = {
        "layout": args.layout,
        "horizon": args.horizon,
        "episodes": args.episodes,
        "lambda_pref": args.comfort_lambda,
        "means": {
            kind: {
                "soup_reward": mean(kind, "soup_reward"),
                "comfort_per_step": mean(kind, "comfort_per_step"),
                "discomfort_rate": mean(kind, "discomfort_rate"),
            }
            for kind in args.ai_modes
        },
        "episodes_detail": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    print("\n=== means over episodes (DeepSeek human) ===")
    for kind in args.ai_modes:
        m = summary["means"][kind]
        print(
            f"  {kind:16s} soup={m['soup_reward']:6.1f}  "
            f"comfort/step={m['comfort_per_step']:+.3f}  "
            f"discomfort_rate={m['discomfort_rate']:.3f}"
        )
    print(f"Wrote {args.output}")
    if args.dataset is not None:
        print(f"Appended {dataset_count} step records to {args.dataset}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
