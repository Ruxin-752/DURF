"""Same-seed H0/frozen versus online-adaptive Route 2 utility simulation.

This diagnostic reuses the trajectory-required benchmark's real Overcooked
``phi(state, subgoal)`` vectors.  A synthetic teacher supplies generic local
approval/disapproval after the frozen H0 choice.  The adaptive condition turns
the full Route 2 prediction into the paper's precision-2 Gaussian update; the
frozen condition retains the neutral prior and therefore follows H0 ties.

Metrics are computed under an independently known synthetic teacher reward:
cumulative ``w* dot phi``, regret, task proxy, and safety violations.  This is
an automated systems check, not evidence about real human preferences.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.evaluate_route2_trajectory_required_benchmark import (  # noqa: E402
    _model_example,
    _predict,
    _read_json,
    _resolve_file,
    _validate_payload,
)
from scripts.generate_route2_trajectory_required_benchmark import (  # noqa: E402
    DEFAULT_OUTPUT_DIR,
    SCHEMA_VERSION,
)
from src.neural_inference import build_vocab, load_checkpoint  # noqa: E402
from src.subgoal_reranker import choose_subgoal  # noqa: E402
from src.text_analysis import nn_tokenize  # noqa: E402


DEFAULT_REPORT = DEFAULT_OUTPUT_DIR / "online_adaptation_simulation.json"
SAFETY_FEATURES = frozenset(
    {
        "wrong_ingredient",
        "breaks_recipe",
        "collision_risk",
        "blocks_human_path",
        "blocks_serving_route",
        "steals_human_target",
    }
)
TASK_FEATURES = frozenset(
    {
        "adds_needed_tomato",
        "adds_needed_onion",
        "matches_current_order",
        "dish_needed_for_ready_soup",
        "pick_ready_soup",
        "serve_ready_soup",
        "supports_serving",
        "completes_recipe",
    }
)


def _score(weights: dict[str, float], features: dict[str, float]) -> float:
    return float(sum(float(weights.get(key, 0.0)) * float(value) for key, value in features.items()))


def _select(weights: dict[str, float], row: dict) -> dict:
    return choose_subgoal(
        weights,
        row["context"],
        row["feasible_subgoals"],
        tie_fallback=row["h0_fallback"],
    )


def _summarize(traces: list[dict]) -> dict:
    count = len(traces)
    return {
        "episodes": count,
        "cumulative_teacher_utility": float(sum(row["teacher_utility"] for row in traces)),
        "mean_teacher_utility": float(np.mean([row["teacher_utility"] for row in traces])),
        "cumulative_regret": float(sum(row["regret"] for row in traces)),
        "mean_regret": float(np.mean([row["regret"] for row in traces])),
        "preference_satisfaction_rate": float(
            np.mean([row["regret"] <= 1e-9 for row in traces])
        ),
        "task_proxy_total": float(sum(row["task_proxy"] for row in traces)),
        "task_proxy_mean": float(np.mean([row["task_proxy"] for row in traces])),
        "safety_violation_count": int(sum(row["safety_violation"] for row in traces)),
        "safety_violation_rate": float(np.mean([row["safety_violation"] for row in traces])),
        "chosen_subgoals": dict(Counter(row["chosen_subgoal"] for row in traces)),
    }


def run_simulation(
    manifest_path: str | Path,
    *,
    model_path: str | Path,
    seed: int,
    rounds: int = 20,
    observation_precision: float = 2.0,
    output: str | Path = DEFAULT_REPORT,
) -> dict:
    if rounds <= 0 or observation_precision <= 0:
        raise ValueError("rounds and observation_precision must be positive")
    manifest_path = Path(manifest_path).resolve()
    manifest = _read_json(manifest_path)
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("unexpected benchmark manifest schema")
    train_dev_path = _resolve_file(manifest_path, manifest["files"]["train_dev"])
    payload = _read_json(train_dev_path)
    features = list(payload["features"])
    rows = _validate_payload(payload, features, allowed_splits={"train", "dev"})
    train_rows = [row for row in rows if row["split"] == "train"]
    vocab = build_vocab([nn_tokenize(row["text"]) for row in train_rows])

    model_path = Path(model_path).resolve()
    model, checkpoint_vocab, checkpoint_features, use_feature_counts = load_checkpoint(model_path)
    if list(checkpoint_features) != features or checkpoint_vocab != vocab:
        raise ValueError("simulation checkpoint does not match benchmark vocabulary/features")
    if not use_feature_counts:
        raise ValueError("online adaptation simulation requires a full-input checkpoint")

    # One approval example per context is enough: same-seed repetition measures
    # accumulation of the paper Gaussian observations without changing scenarios.
    by_context: dict[str, dict] = {}
    for row in train_rows:
        if row["polarity"] == "positive":
            by_context.setdefault(row["context_id"], row)
    ordered_contexts = [by_context[key] for key in sorted(by_context)]
    # One fixed hidden teacher reward governs the whole simulation.  Each local
    # approval is a context-specific observation whose expectation averages to
    # this teacher, so the online posterior never changes teacher mid-run.
    teacher_weights = {
        feature: float(
            np.mean(
                [row["teacher_reward_weights"][feature] for row in ordered_contexts]
            )
        )
        for feature in features
    }
    rng = np.random.default_rng(seed)
    schedule = [ordered_contexts[index] for index in rng.integers(0, len(ordered_contexts), rounds)]

    frozen_weights = {feature: 0.0 for feature in features}
    adaptive_weights = dict(frozen_weights)
    adaptive_precision = {feature: 1.0 / 25.0 for feature in features}
    condition_traces = {"frozen_h0": [], "online_adaptive": []}
    update_traces = []
    for round_index, row in enumerate(schedule):
        oracle_scores = {
            item["subgoal"]: _score(teacher_weights, item["features"])
            for item in _select(teacher_weights, row)["ranking"]
        }
        oracle_best = max(oracle_scores.values())
        for condition, current in (
            ("frozen_h0", frozen_weights),
            ("online_adaptive", adaptive_weights),
        ):
            choice = _select(current, row)
            chosen = choice["chosen_subgoal"]
            chosen_row = next(item for item in choice["ranking"] if item["subgoal"] == chosen)
            phi = chosen_row["features"]
            utility = oracle_scores[chosen]
            condition_traces[condition].append(
                {
                    "round": round_index,
                    "context_id": row["context_id"],
                    "chosen_subgoal": chosen,
                    "teacher_utility": utility,
                    "regret": float(oracle_best - utility),
                    "task_proxy": float(sum(float(phi.get(key, 0.0)) for key in TASK_FEATURES)),
                    "safety_violation": any(float(phi.get(key, 0.0)) > 0 for key in SAFETY_FEATURES),
                }
            )

        example = _model_example(row, features, input_mode="full")
        predicted = _predict(model, [example], vocab, use_feature_counts=True)[0]
        prior_mean = np.asarray([adaptive_weights[feature] for feature in features], dtype=np.float64)
        prior_precision = np.asarray([adaptive_precision[feature] for feature in features], dtype=np.float64)
        posterior_precision = prior_precision + observation_precision
        posterior = (prior_precision * prior_mean + observation_precision * predicted) / posterior_precision
        for index, feature in enumerate(features):
            adaptive_weights[feature] = float(posterior[index])
            adaptive_precision[feature] = float(posterior_precision[index])
        update_traces.append(
            {
                "round": round_index,
                "context_id": row["context_id"],
                "text": row["text"],
                "update_rule": "paper_independent_gaussian",
                "observation_precision": observation_precision,
            }
        )

    summaries = {name: _summarize(trace) for name, trace in condition_traces.items()}
    frozen = summaries["frozen_h0"]
    adaptive = summaries["online_adaptive"]
    report = {
        "schema_version": 1,
        "claim_scope": "synthetic online adaptation systems diagnostic; not human preference evidence",
        "seed": int(seed),
        "rounds": rounds,
        "score_formula": "w_dot_phi",
        "update_rule": "paper_independent_gaussian",
        "observation_precision": observation_precision,
        "benchmark_manifest": str(manifest_path),
        "benchmark_manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        "model_path": str(model_path),
        "model_sha256": hashlib.sha256(model_path.read_bytes()).hexdigest(),
        "conditions": summaries,
        "adaptive_minus_frozen": {
            "cumulative_teacher_utility": (
                adaptive["cumulative_teacher_utility"] - frozen["cumulative_teacher_utility"]
            ),
            "cumulative_regret_reduction": (
                frozen["cumulative_regret"] - adaptive["cumulative_regret"]
            ),
            "preference_satisfaction_rate": (
                adaptive["preference_satisfaction_rate"] - frozen["preference_satisfaction_rate"]
            ),
            "task_proxy_total": adaptive["task_proxy_total"] - frozen["task_proxy_total"],
            "safety_violation_count": (
                adaptive["safety_violation_count"] - frozen["safety_violation_count"]
            ),
        },
        "validity_gate": {
            "adaptive_utility_not_worse": (
                adaptive["cumulative_teacher_utility"] >= frozen["cumulative_teacher_utility"]
            ),
            "adaptive_regret_lower": adaptive["cumulative_regret"] < frozen["cumulative_regret"],
            "adaptive_safety_not_worse": (
                adaptive["safety_violation_count"] <= frozen["safety_violation_count"]
            ),
        },
        "updates": update_traces,
        "traces": condition_traces,
    }
    report["validity_gate"]["status"] = (
        "passed"
        if all(
            value
            for key, value in report["validity_gate"].items()
            if key != "status"
        )
        else "failed"
    )
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    report["report_path"] = str(output)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest", type=Path, default=DEFAULT_OUTPUT_DIR / "benchmark_manifest.json"
    )
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=137)
    parser.add_argument("--rounds", type=int, default=20)
    parser.add_argument("--observation-precision", type=float, default=2.0)
    parser.add_argument("--output", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args()
    report = run_simulation(
        args.manifest,
        model_path=args.model,
        seed=args.seed,
        rounds=args.rounds,
        observation_precision=args.observation_precision,
        output=args.output,
    )
    for condition, metrics in report["conditions"].items():
        print(
            f"{condition}: utility={metrics['cumulative_teacher_utility']:.3f}, "
            f"regret={metrics['cumulative_regret']:.3f}, "
            f"preference={metrics['preference_satisfaction_rate']:.2%}, "
            f"task={metrics['task_proxy_total']:.1f}, safety={metrics['safety_violation_count']}"
        )
    print(f"Validity gate: {report['validity_gate']['status']}")
    print(f"Report: {report['report_path']}")
    return 0 if report["validity_gate"]["status"] == "passed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
