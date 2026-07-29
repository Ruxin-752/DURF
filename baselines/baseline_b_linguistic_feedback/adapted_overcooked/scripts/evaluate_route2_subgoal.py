"""Evaluate Route 2 by the metric that matters: held-out subgoal accuracy.

MSE on the reward vector is not the goal. The goal is: given *unseen* feedback
text about a *held-out* decision scenario, does the predicted reward re-rank
H0's feasible subgoals to the human-comfortable one?

    text -> Route 2 -> w_hat -> subgoal_reranker.choose_subgoal -> subgoal

We compare four reward sources on the held-out fold:
- ``gold``:   the hand-authored w* (upper bound / rule sanity check);
- ``zero``:   zero weights (ties everywhere -> failure baseline);
- ``route1``: one global Bayesian belief learned from the training feedback;
- ``route2``: the trained network, per feedback utterance (text-only).

We also export ``outputs/route2/learned_comfort_weights.json``: among
utterance-mean / scenario-balanced-mean / utterance-median aggregations of
text-only Route 2 predictions, keep the one that best recovers the
hand-authored subgoal probes (never used in training). That frozen vector
drives Path A subgoal re-ranking (and optional PPO shaping).
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.run_baseline_b_pipeline import learn_from_feedback  # noqa: E402
from src.evaluation_splits import (  # noqa: E402
    canonical_sha256,
    deduplicate_corpus,
    load_split_manifest,
    make_split_manifest,
)
from src.feature_schema import empty_weights, load_features, read_json, write_json  # noqa: E402
from src.neural_inference import (  # noqa: E402
    load_checkpoint,
    predict_reward_vector,
)
from src.probe_evaluator import DEFAULT_PROBE_STATES_PATH, load_probe_states  # noqa: E402
from src.subgoal_reranker import choose_subgoal  # noqa: E402
from src.subgoal_teacher import load_gold_weights  # noqa: E402


DEFAULT_FEEDBACK_PATH = ROOT / "data" / "synthetic_feedback.json"
DEFAULT_MODEL_PATH = ROOT / "outputs" / "route2" / "model.pt"
DEFAULT_WEIGHTS_OUT = ROOT / "outputs" / "route2" / "learned_comfort_weights.json"


def _rerank_correct(weights: dict[str, float], example: dict, *, lambda_pref: float) -> bool:
    choice = choose_subgoal(
        weights, example["context"], example["feasible_subgoals"], lambda_pref=lambda_pref
    )
    acceptable = example.get("acceptable_subgoals") or [example.get("expected_subgoal")]
    return (not choice["is_tie"]) and choice["chosen_subgoal"] in acceptable


def _scenario_accuracy(weights: dict[str, float], scenarios: list[dict], *, lambda_pref: float) -> dict:
    correct = sum(_rerank_correct(weights, sc, lambda_pref=lambda_pref) for sc in scenarios)
    total = len(scenarios)
    return {"correct": correct, "total": total, "accuracy": correct / total if total else 0.0}


def _unique_scenarios(examples: list[dict]) -> list[dict]:
    scenarios: dict[str, dict] = {}
    for example in examples:
        gid = example["group_id"]
        if gid not in scenarios:
            scenarios[gid] = {
                "group_id": gid,
                "context": example["context"],
                "feasible_subgoals": example["feasible_subgoals"],
                "acceptable_subgoals": example.get("acceptable_subgoals"),
                "expected_subgoal": example.get("expected_subgoal"),
            }
    return list(scenarios.values())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feedback", type=Path, default=DEFAULT_FEEDBACK_PATH)
    parser.add_argument("--probe-states", type=Path, default=DEFAULT_PROBE_STATES_PATH)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dev-fraction", type=float, default=0.2)
    parser.add_argument("--test-fraction", type=float, default=0.2)
    parser.add_argument("--lambda-pref", type=float, default=1.0)
    parser.add_argument("--weights-out", type=Path, default=DEFAULT_WEIGHTS_OUT)
    parser.add_argument(
        "--split-manifest",
        type=Path,
        help="Fixed split manifest (defaults to split_manifest.json beside model).",
    )
    parser.add_argument("--near-duplicate-threshold", type=float, default=0.92)
    parser.add_argument(
        "--report-out",
        type=Path,
        help="Evaluation metadata/metrics JSON (default: beside --weights-out).",
    )
    args = parser.parse_args()

    feedback_examples, corpus_audit = deduplicate_corpus(
        read_json(args.feedback),
        near_duplicate_threshold=args.near_duplicate_threshold,
    )
    features = load_features()
    probes = load_probe_states(args.probe_states)

    groups = [f.get("group_id") or f.get("probe_id") for f in feedback_examples]
    manifest_path = args.split_manifest
    default_manifest = args.model.parent / "split_manifest.json"
    if manifest_path is None and default_manifest.exists():
        manifest_path = default_manifest
    if manifest_path is not None:
        split = load_split_manifest(manifest_path, feedback_examples)
    else:
        # Backward compatibility for old runs that predate persisted manifests.
        split = make_split_manifest(
            feedback_examples,
            seed=args.seed,
            dev_fraction=args.dev_fraction,
            test_fraction=args.test_fraction,
            near_duplicate_threshold=args.near_duplicate_threshold,
        )
        print("Warning: no split manifest found; using a deterministic legacy split.")
    test_groups = set(split["test_groups"])

    train_feedback = [f for i, f in enumerate(feedback_examples) if groups[i] not in test_groups]
    test_feedback = [f for i, f in enumerate(feedback_examples) if groups[i] in test_groups]
    test_scenarios = _unique_scenarios(test_feedback)

    print(
        f"Untouched test split: {len(test_groups)} scenarios, "
        f"{len(test_feedback)} utterances"
    )

    # --- gold / zero baselines (scenario level) ---
    gold = load_gold_weights()
    zero = empty_weights(features)
    gold_acc = _scenario_accuracy(gold, test_scenarios, lambda_pref=args.lambda_pref)
    zero_acc = _scenario_accuracy(zero, test_scenarios, lambda_pref=args.lambda_pref)

    # --- Route 1: one global Bayesian belief from training feedback ---
    model1, _ = learn_from_feedback(
        feedback_examples=train_feedback, probe_states=probes, features=features
    )
    route1_acc = _scenario_accuracy(model1.as_dict(), test_scenarios, lambda_pref=args.lambda_pref)

    # --- Route 2: trained network, per utterance (text-only) ---
    model2, vocab, model_features, use_fc = load_checkpoint(args.model)
    if use_fc:
        raise ValueError(
            "Route 2 checkpoint uses feature counts; text-only evaluation requires False"
        )
    checkpoint = torch.load(args.model, map_location="cpu", weights_only=False)
    checkpoint_extra = checkpoint.get("extra") or {}
    for field in ("corpus_sha256", "split_sha256"):
        expected = split[field]
        actual = checkpoint_extra.get(field)
        if actual is not None and actual != expected:
            raise ValueError(
                f"Checkpoint {field} mismatch: expected {expected}, got {actual}"
            )
    per_example_correct = 0
    scenario_votes: dict[str, Counter] = {}
    for example in test_feedback:
        w_hat = predict_reward_vector(model2, vocab, model_features, example["text"])
        choice = choose_subgoal(
            w_hat, example["context"], example["feasible_subgoals"], lambda_pref=args.lambda_pref
        )
        acceptable = example.get("acceptable_subgoals") or [example.get("expected_subgoal")]
        correct = (not choice["is_tie"]) and choice["chosen_subgoal"] in acceptable
        per_example_correct += int(correct)
        scenario_votes.setdefault(example["group_id"], Counter())[choice["chosen_subgoal"]] += 1

    route2_example_acc = per_example_correct / len(test_feedback) if test_feedback else 0.0
    scenario_correct = 0
    for scenario in test_scenarios:
        votes = scenario_votes.get(scenario["group_id"])
        if not votes:
            continue
        chosen = votes.most_common(1)[0][0]
        acceptable = scenario.get("acceptable_subgoals") or [scenario.get("expected_subgoal")]
        scenario_correct += int(chosen in acceptable)
    route2_scenario_acc = scenario_correct / len(test_scenarios) if test_scenarios else 0.0

    print("\nHeld-out subgoal accuracy (scenario level):")
    print(f"  gold w*      : {gold_acc['correct']}/{gold_acc['total']} = {gold_acc['accuracy']*100:.1f}%")
    print(f"  zero weights : {zero_acc['correct']}/{zero_acc['total']} = {zero_acc['accuracy']*100:.1f}%")
    print(f"  route1 global: {route1_acc['correct']}/{route1_acc['total']} = {route1_acc['accuracy']*100:.1f}%")
    print(
        f"  route2 net   : {scenario_correct}/{len(test_scenarios)} = {route2_scenario_acc*100:.1f}% "
        f"(majority vote); per-utterance {route2_example_acc*100:.1f}%"
    )

    # --- export frozen comfort weights (tier-2: not a naive utterance mean) ---
    # Cache per-utterance predictions once, then try several aggregations and
    # keep the one that best recovers the hand-authored subgoal probes (never
    # used in training). Scenario-balanced mean avoids over-weighting verbose
    # scenarios; median damps outlier paraphrases.
    export_feedback = [
        example
        for example in feedback_examples
        if example.get("group_id") not in test_groups
    ]
    per_text_w = [
        predict_reward_vector(model2, vocab, model_features, example["text"])
        for example in export_feedback
    ]
    by_group: dict[str, list[dict[str, float]]] = {}
    for example, w_hat in zip(export_feedback, per_text_w):
        by_group.setdefault(example["group_id"], []).append(w_hat)

    def _mean_weights(vectors: list[dict[str, float]]) -> dict[str, float]:
        if not vectors:
            return {f: 0.0 for f in model_features}
        out = {f: 0.0 for f in model_features}
        for vec in vectors:
            for f, v in vec.items():
                out[f] += v
        n = float(len(vectors))
        return {f: v / n for f, v in out.items()}

    def _median_weights(vectors: list[dict[str, float]]) -> dict[str, float]:
        if not vectors:
            return {f: 0.0 for f in model_features}
        out: dict[str, float] = {}
        for f in model_features:
            vals = sorted(vec[f] for vec in vectors)
            mid = len(vals) // 2
            out[f] = (
                float(vals[mid])
                if len(vals) % 2
                else 0.5 * (vals[mid - 1] + vals[mid])
            )
        return out

    utterance_mean = _mean_weights(per_text_w)
    scenario_means = [_mean_weights(vecs) for vecs in by_group.values()]
    scenario_balanced = _mean_weights(scenario_means)
    utterance_median = _median_weights(per_text_w)

    full_scenarios = _unique_scenarios(feedback_examples)
    probe_scenarios = [
        {
            "group_id": p["probe_id"],
            "context": p["context"],
            "feasible_subgoals": p["feasible_subgoals"],
            "acceptable_subgoals": p["acceptable_subgoals"],
            "expected_subgoal": p["expected_subgoal"],
        }
        for p in read_json(ROOT / "data" / "subgoal_probe_states.json")
    ]

    candidates = {
        "utterance_mean": utterance_mean,
        "scenario_balanced_mean": scenario_balanced,
        "utterance_median": utterance_median,
    }
    print("\nExport candidates (reported only; selection is fixed in advance):")
    for name, weights in candidates.items():
        probe = _scenario_accuracy(weights, probe_scenarios, lambda_pref=args.lambda_pref)
        synth = _scenario_accuracy(weights, full_scenarios, lambda_pref=args.lambda_pref)
        print(
            f"  {name:24s} probes {probe['correct']}/{probe['total']}="
            f"{probe['accuracy']*100:.1f}% | synth {synth['correct']}/{synth['total']}="
            f"{synth['accuracy']*100:.1f}%"
        )
    # Fixed before looking at probes/test: scenario balancing prevents verbose
    # scenarios from dominating and removes probe-based model selection.
    best_name = "scenario_balanced_mean"
    learned_weights = scenario_balanced
    write_json(args.weights_out, learned_weights)
    export_acc = _scenario_accuracy(learned_weights, full_scenarios, lambda_pref=args.lambda_pref)
    probe_acc = _scenario_accuracy(learned_weights, probe_scenarios, lambda_pref=args.lambda_pref)

    print(f"\nExported comfort weights ({best_name}):")
    print(f"  all synthetic scenarios: {export_acc['correct']}/{export_acc['total']} = {export_acc['accuracy']*100:.1f}%")
    print(f"  hand-authored subgoal probes (held-out, only-tested): "
          f"{probe_acc['correct']}/{probe_acc['total']} = {probe_acc['accuracy']*100:.1f}%")
    print(f"  weights JSON: {args.weights_out}")
    evaluation_config = {
        "seed": args.seed,
        "lambda_pref": args.lambda_pref,
        "use_feature_counts": False,
        "near_duplicate_threshold": args.near_duplicate_threshold,
    }
    evaluation_config["config_sha256"] = canonical_sha256(evaluation_config)
    report = {
        "corpus_sha256": split["corpus_sha256"],
        "split_sha256": split["split_sha256"],
        "split_manifest": str(manifest_path) if manifest_path else None,
        "checkpoint": str(args.model),
        "checkpoint_training_config": checkpoint_extra.get("training_config"),
        "evaluation_config": evaluation_config,
        "corpus_audit": corpus_audit,
        "test_split": {
            "groups": sorted(test_groups),
            "utterances": len(test_feedback),
            "scenarios": len(test_scenarios),
        },
        "metrics": {
            "gold": gold_acc,
            "zero": zero_acc,
            "route1": route1_acc,
            "route2_scenario_accuracy": route2_scenario_acc,
            "route2_example_accuracy": route2_example_acc,
            "export_all_scenarios": export_acc,
            "export_probes": probe_acc,
        },
        "export_aggregation": best_name,
    }
    report_out = args.report_out or args.weights_out.with_name(
        args.weights_out.stem + ".evaluation.json"
    )
    write_json(report_out, report)
    print(f"  evaluation report: {report_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
