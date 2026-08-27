"""Evaluate paper-aligned Route 2 on untouched reward/style configurations."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.prepare_route2_dataset import build_dataset  # noqa: E402
from src.feature_schema import load_features, read_json, write_json  # noqa: E402
from src.neural_inference import (  # noqa: E402
    collate_batch,
    load_checkpoint,
    predict_reward_vector,
)
from src.observations import reference_vector  # noqa: E402
from src.subgoal_featurizer import featurize_subgoal  # noqa: E402
from src.subgoal_planner import enumerate_feasible_subgoals  # noqa: E402
from src.probe_evaluator import DEFAULT_PROBE_STATES_PATH, load_probe_states  # noqa: E402
from src.subgoal_reranker import choose_subgoal  # noqa: E402


DEFAULT_REPORT = ROOT / "outputs" / "route2" / "paper_aligned" / "reward_report.json"


def _reward_metrics(predictions: torch.Tensor, targets: torch.Tensor) -> dict:
    active = torch.abs(targets) > 1e-8
    sign_correct = (torch.sign(predictions) == torch.sign(targets)) & active
    cosine = torch.nn.functional.cosine_similarity(predictions, targets, dim=1, eps=1e-8)
    return {
        "mse": float(torch.mean((predictions - targets) ** 2)),
        "mean_cosine_similarity": float(torch.mean(cosine)),
        "active_sign_accuracy": (
            float(sign_correct.sum() / active.sum()) if int(active.sum()) else 0.0
        ),
        "active_target_count": int(active.sum()),
        "examples": int(targets.shape[0]),
    }


def _live_language_proactivity_probe(
    model, vocab: dict[str, int], features: list[str], use_feature_counts: bool
) -> dict:
    """Test unseen human-style wording on one fixed cooking state."""

    context = {
        "recipe": ["tomato", "tomato", "onion"],
        "pot_ingredients": ["tomato", "tomato", "onion"],
        "pot_status": "cooking",
        "agent_holding": None,
    }
    candidates = enumerate_feasible_subgoals(context)
    cases = (
        (
            "positive_specific",
            "Please prepare an onion for the next round while this soup cooks.",
            "WAIT",
            "prefetch",
        ),
        (
            "positive_paraphrase",
            "Use the cooking time to get an ingredient ready for our next soup.",
            "WAIT",
            "prefetch",
        ),
        (
            "negative_specific",
            "Do not grab an onion this early; keep your hands free while the soup cooks.",
            "GET_ONION",
            "wait",
        ),
        (
            "negative_paraphrase",
            "Wait until this soup is done; do not fill your hands yet.",
            "GET_ONION",
            "wait",
        ),
    )
    rows = []
    for case_id, text, grounding_subgoal, expected in cases:
        counts = reference_vector(
            featurize_subgoal(context, grounding_subgoal), features
        ).tolist()
        weights = predict_reward_vector(
            model,
            vocab,
            features,
            text,
            use_feature_counts=use_feature_counts,
            feature_counts=counts,
        )
        decision = choose_subgoal(
            weights,
            context,
            candidates,
            tie_fallback="WAIT",
        )
        chosen = decision["chosen_subgoal"]
        correct = chosen != "WAIT" if expected == "prefetch" else chosen == "WAIT"
        rows.append(
            {
                "case_id": case_id,
                "text": text,
                "grounding_subgoal": grounding_subgoal,
                "expected": expected,
                "chosen_subgoal": chosen,
                "correct": correct,
                "reward_margin": decision["reward_margin"],
                "scores": {
                    row["subgoal"]: row["total_score"]
                    for row in decision["ranking"]
                },
            }
        )
    correct = sum(row["correct"] for row in rows)
    return {
        "correct": correct,
        "total": len(rows),
        "accuracy": correct / len(rows),
        "cases": rows,
    }


def evaluate(*, feedback_path: Path, model_path: Path, probe_states_path: Path) -> dict:
    feedback = read_json(feedback_path)
    dataset = build_dataset(
        feedback,
        load_probe_states(probe_states_path),
        load_features(),
        split_manifest=model_path.parent / "split_manifest.json",
    )
    model, vocab, features, use_feature_counts = load_checkpoint(model_path)
    if dataset["target_mode"] != "full_teacher_reward":
        raise ValueError("paper-aligned evaluation requires full teacher reward targets")
    checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
    extra = checkpoint.get("extra") or {}
    if extra.get("split_sha256") not in {None, dataset["split_sha256"]}:
        raise ValueError("checkpoint and evaluation split hashes differ")

    indices = list(dataset["split"]["test_indices"])
    examples = [dataset["examples"][index] for index in indices]
    batch = collate_batch(examples, vocab)
    if not use_feature_counts:
        batch["feature_counts"] = torch.zeros_like(batch["feature_counts"])
    with torch.no_grad():
        predictions = model(
            batch["tokens"], batch["offsets"], batch["feature_counts"]
        )
    targets = batch["targets"]

    per_config_predictions: dict[str, list[torch.Tensor]] = defaultdict(list)
    per_config_targets: dict[str, list[torch.Tensor]] = defaultdict(list)
    subgoal_correct = 0
    subgoal_total = 0
    per_config_subgoals: dict[str, dict[str, int]] = defaultdict(
        lambda: {"correct": 0, "total": 0}
    )
    seen_contexts: set[tuple[str, str]] = set()
    for row_index, (example, prediction) in enumerate(zip(examples, predictions)):
        config_id = str(example["reward_config_id"])
        per_config_predictions[config_id].append(prediction)
        per_config_targets[config_id].append(targets[row_index])
        context_key = (config_id, str(example["group_id"]))
        if context_key in seen_contexts:
            continue
        seen_contexts.add(context_key)
        weights = {
            feature: float(value)
            for feature, value in zip(features, prediction.tolist())
        }
        choice = choose_subgoal(
            weights,
            example["context"],
            example["feasible_subgoals"],
        )
        acceptable = example.get("acceptable_subgoals") or [
            example.get("expected_subgoal")
        ]
        correct = (not choice["is_tie"]) and choice["chosen_subgoal"] in acceptable
        subgoal_correct += int(correct)
        subgoal_total += 1
        per_config_subgoals[config_id]["correct"] += int(correct)
        per_config_subgoals[config_id]["total"] += 1

    by_config = {}
    for config_id in sorted(per_config_predictions):
        predicted = torch.stack(per_config_predictions[config_id])
        target = torch.stack(per_config_targets[config_id])
        subgoals = per_config_subgoals[config_id]
        by_config[config_id] = {
            "reward_metrics": _reward_metrics(predicted, target),
            "subgoal_accuracy": (
                subgoals["correct"] / subgoals["total"] if subgoals["total"] else 0.0
            ),
            **subgoals,
        }

    return {
        "feedback": str(feedback_path),
        "checkpoint": str(model_path),
        "target_mode": dataset["target_mode"],
        "use_feature_counts": use_feature_counts,
        "corpus_sha256": dataset["corpus_sha256"],
        "split_sha256": dataset["split_sha256"],
        "test_reward_configs": dataset["split"]["splits"]["test"]["reward_configs"],
        "test_teacher_styles": dataset["split"]["splits"]["test"]["teacher_ids"],
        "reward_metrics": _reward_metrics(predictions, targets),
        "subgoal_metrics": {
            "correct": subgoal_correct,
            "total": subgoal_total,
            "accuracy": subgoal_correct / subgoal_total if subgoal_total else 0.0,
        },
        "by_reward_config": by_config,
        "gold_config": by_config.get("reward_00_gold"),
        "live_language_proactivity": _live_language_proactivity_probe(
            model, vocab, features, use_feature_counts
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feedback", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--probe-states", type=Path, default=DEFAULT_PROBE_STATES_PATH)
    parser.add_argument("--report-out", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args()
    report = evaluate(
        feedback_path=args.feedback,
        model_path=args.model,
        probe_states_path=args.probe_states,
    )
    write_json(args.report_out, report)
    print(json.dumps(report["reward_metrics"], indent=2))
    print(json.dumps(report["subgoal_metrics"], indent=2))
    print(json.dumps(report["live_language_proactivity"], indent=2))
    print(f"Report: {args.report_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
