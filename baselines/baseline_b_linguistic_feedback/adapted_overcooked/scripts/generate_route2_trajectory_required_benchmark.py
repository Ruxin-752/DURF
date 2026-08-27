"""Generate a frozen Route 2 diagnostic in which both inputs are necessary.

The benchmark is deliberately crossed along two axes:

* one local utterance is repeated across many different executed trajectories;
* one executed trajectory is paired with both approval and disapproval.

Consequently, text-only can recover polarity but not the referenced reward
context, while trajectory-only can recover the context but not polarity.  The
complete input must combine both to predict the complete 53-dimensional target.
This is a synthetic identifiability diagnostic, not a human-language estimate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.feature_schema import load_features  # noqa: E402
from src.subgoal_featurizer import featurize_subgoal  # noqa: E402


SCHEMA_VERSION = "route2-trajectory-required-v1"
DEFAULT_OUTPUT_DIR = ROOT / "outputs" / "route2" / "trajectory_required_v1"

TRAINING_CONFIG = {
    "seeds": [17, 31, 47],
    "epochs": 240,
    "learning_rate": 0.01,
    "weight_decay": 0.0001,
    "batch_size": 64,
    "patience": 35,
    "optimizer": "adam",
    "dimension_weighting": "varying_only",
}


TEMPLATES = {
    "train": {
        "positive": [
            "I approve of that move.",
            "That move matches what I prefer.",
            "I liked that action.",
            "Yes, that behavior is helpful.",
            "Please keep doing that.",
            "That was a good choice for me.",
            "This is the kind of move I want.",
            "The action you took was useful.",
        ],
        "negative": [
            "I disapprove of that move.",
            "That move does not match what I prefer.",
            "I disliked that action.",
            "No, that behavior is unhelpful.",
            "Please stop doing that.",
            "That was a bad choice for me.",
            "This is not the kind of move I want.",
            "The action you took was useless.",
        ],
    },
    "dev": {
        "positive": [
            "I liked the move you made.",
            "That was useful and matches what I want.",
            "Keep doing that helpful action.",
        ],
        "negative": [
            "I disliked the move you made.",
            "That was useless and does not match what I want.",
            "Stop doing that unhelpful action.",
        ],
    },
    "test": {
        "positive": [
            "I approve because that action was helpful.",
            "The move was good and useful for me.",
            "Yes, keep that behavior.",
        ],
        "negative": [
            "I disapprove because that action was unhelpful.",
            "The move was bad and useless for me.",
            "No, stop that behavior.",
        ],
    },
}


def benchmark_contexts() -> list[dict]:
    """Return actual Overcooked subgoal contexts used by the diagnostic."""

    cooking = {
        "recipe": ["tomato", "tomato", "onion"],
        "pot_ingredients": ["tomato", "tomato", "onion"],
        "pot_status": "cooking",
        "agent_holding": None,
        "human_holding": None,
        "human_intent": None,
    }
    return [
        {
            "context_id": "prefetch_tomato",
            "context": dict(cooking),
            "feasible_subgoals": ["GET_DISH", "GET_TOMATO", "GET_ONION", "WAIT"],
            "referenced_subgoal": "GET_TOMATO",
            "h0_fallback": "WAIT",
        },
        {
            "context_id": "prefetch_onion",
            "context": dict(cooking),
            "feasible_subgoals": ["GET_DISH", "GET_TOMATO", "GET_ONION", "WAIT"],
            "referenced_subgoal": "GET_ONION",
            "h0_fallback": "WAIT",
        },
        {
            "context_id": "prefetch_dish",
            "context": dict(cooking),
            "feasible_subgoals": ["GET_DISH", "GET_TOMATO", "GET_ONION", "WAIT"],
            "referenced_subgoal": "GET_DISH",
            "h0_fallback": "WAIT",
        },
        {
            "context_id": "ready_get_dish",
            "context": {
                **cooking,
                "pot_status": "ready",
            },
            "feasible_subgoals": ["GET_DISH", "WAIT"],
            "referenced_subgoal": "GET_DISH",
            "h0_fallback": "WAIT",
        },
        {
            "context_id": "ready_pickup_soup",
            "context": {
                **cooking,
                "pot_status": "ready",
                "agent_holding": "dish",
            },
            "feasible_subgoals": ["PICKUP_SOUP", "WAIT"],
            "referenced_subgoal": "PICKUP_SOUP",
            "h0_fallback": "WAIT",
        },
        {
            "context_id": "serve_held_soup",
            "context": {
                **cooking,
                "pot_status": "ready",
                "agent_holding": "soup",
            },
            "feasible_subgoals": ["SERVE_SOUP", "WAIT"],
            "referenced_subgoal": "SERVE_SOUP",
            "h0_fallback": "WAIT",
        },
        {
            "context_id": "complement_human_tomato",
            "context": {
                **cooking,
                "human_intent": "get tomato",
            },
            "feasible_subgoals": ["GET_DISH", "GET_TOMATO", "GET_ONION", "WAIT"],
            "referenced_subgoal": "GET_ONION",
            "h0_fallback": "GET_TOMATO",
        },
        {
            "context_id": "complement_human_onion",
            "context": {
                **cooking,
                "human_intent": "get onion",
            },
            "feasible_subgoals": ["GET_DISH", "GET_TOMATO", "GET_ONION", "WAIT"],
            "referenced_subgoal": "GET_TOMATO",
            "h0_fallback": "GET_ONION",
        },
        {
            "context_id": "yield_human_tomato",
            "context": {
                **cooking,
                "human_intent": "get tomato",
            },
            "feasible_subgoals": ["GET_TOMATO", "GET_ONION", "WAIT"],
            "referenced_subgoal": "WAIT",
            "h0_fallback": "GET_TOMATO",
        },
        {
            "context_id": "yield_human_serving",
            "context": {
                **cooking,
                "pot_status": "ready",
                "human_intent": "serve soup",
            },
            "feasible_subgoals": ["GET_DISH", "WAIT"],
            "referenced_subgoal": "WAIT",
            "h0_fallback": "GET_DISH",
        },
    ]


def _normalized_text(text: str) -> str:
    return " ".join(str(text).casefold().split())


def _vector_fingerprint(vector: dict[str, float]) -> str:
    encoded = json.dumps(vector, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, payload: object) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def context_reward_basis(spec: dict, features: list[str]) -> dict[str, float]:
    """Create a complete reward that uniquely prefers the referenced subgoal."""

    reference = featurize_subgoal(spec["context"], spec["referenced_subgoal"])
    alternatives = [
        featurize_subgoal(spec["context"], subgoal)
        for subgoal in spec["feasible_subgoals"]
        if subgoal != spec["referenced_subgoal"]
    ]
    # A referenced action need not own a positive feature that no alternative
    # has.  In particular, a correctly featurized WAIT can be a strict subset
    # of productive action vectors: preferring it is then represented by a
    # negative weight on features introduced only by those alternatives.
    #
    # Use both directions of each binary support difference while omitting
    # features shared by every candidate.  This preserves the diagnostic's
    # trajectory dependence instead of filling its targets with context-only
    # state facts that cancel from every candidate score.
    positive_discriminators = {
        feature
        for feature, value in reference.items()
        if value > 0.0
        and any(other.get(feature, 0.0) == 0.0 for other in alternatives)
    }
    negative_discriminators = {
        feature
        for other in alternatives
        for feature, value in other.items()
        if value > 0.0 and reference.get(feature, 0.0) == 0.0
    }
    if not positive_discriminators and not negative_discriminators:
        raise ValueError(
            f"{spec['context_id']}: referenced subgoal has no discriminating feature"
        )
    weights = {feature: 0.0 for feature in features}
    for feature in positive_discriminators:
        weights[feature] = 2.0
    for feature in negative_discriminators:
        weights[feature] = -2.0
    reference_score = sum(weights.get(k, 0.0) * v for k, v in reference.items())
    alternative_scores = [
        sum(weights.get(k, 0.0) * v for k, v in vector.items())
        for vector in alternatives
    ]
    if alternative_scores and reference_score <= max(alternative_scores):
        raise ValueError(f"{spec['context_id']}: reward basis does not identify reference")
    return weights


def build_rows(features: list[str]) -> list[dict]:
    rows: list[dict] = []
    for split, polarities in TEMPLATES.items():
        for polarity, texts in polarities.items():
            sign = 1.0 if polarity == "positive" else -1.0
            for template_index, text in enumerate(texts):
                text_id = f"{split}_{polarity}_{template_index:02d}"
                for spec in benchmark_contexts():
                    basis = context_reward_basis(spec, features)
                    target = {
                        feature: sign * float(basis[feature]) for feature in features
                    }
                    trajectory = featurize_subgoal(
                        spec["context"], spec["referenced_subgoal"]
                    )
                    context_id = spec["context_id"]
                    rows.append(
                        {
                            "feedback_id": f"cf_{text_id}_{context_id}",
                            "split": split,
                            "text": text,
                            "polarity": polarity,
                            "attributed_sentiment_score": sign,
                            "expected_feedback_type": "evaluative",
                            "source": "synthetic_counterfactual_diagnostic",
                            "target_mode": "full_teacher_reward",
                            "target_provenance": "deterministic_counterfactual_reward_context",
                            "teacher_id": f"diagnostic_teacher_{context_id}_{polarity}",
                            "reward_config_id": f"diagnostic_reward_{context_id}_{polarity}",
                            "group_id": context_id,
                            "text_pair_id": text_id,
                            "trajectory_pair_id": f"{split}_{context_id}",
                            "context_id": context_id,
                            "context": spec["context"],
                            "feasible_subgoals": spec["feasible_subgoals"],
                            "referenced_subgoal": spec["referenced_subgoal"],
                            "h0_fallback": spec["h0_fallback"],
                            "route2_trajectory_features": {
                                str(key): float(value) for key, value in trajectory.items()
                            },
                            "teacher_reward_weights": target,
                        }
                    )
    return rows


def audit_rows(rows: list[dict]) -> dict:
    text_groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    trajectory_groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    inputs: dict[tuple[str, str], set[str]] = defaultdict(set)
    split_texts: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        text = _normalized_text(row["text"])
        trajectory = _vector_fingerprint(row["route2_trajectory_features"])
        target = _vector_fingerprint(row["teacher_reward_weights"])
        text_groups[(row["split"], text)].append(row)
        trajectory_groups[(row["split"], trajectory)].append(row)
        inputs[(text, trajectory)].add(target)
        split_texts[row["split"]].add(text)

    text_crossings = [
        group
        for group in text_groups.values()
        if len({_vector_fingerprint(row["route2_trajectory_features"]) for row in group}) > 1
        and len({_vector_fingerprint(row["teacher_reward_weights"]) for row in group}) > 1
    ]
    trajectory_crossings = [
        group
        for group in trajectory_groups.values()
        if {row["polarity"] for row in group} == {"positive", "negative"}
        and len({_vector_fingerprint(row["teacher_reward_weights"]) for row in group}) > 1
    ]
    conflicts = sum(len(targets) > 1 for targets in inputs.values())
    overlaps = {
        f"{left}_{right}": len(split_texts[left] & split_texts[right])
        for left, right in (("train", "dev"), ("train", "test"), ("dev", "test"))
    }
    if conflicts:
        raise ValueError("same exact (text, trajectory) maps to conflicting targets")
    if len(text_crossings) != len(text_groups):
        raise ValueError("every text must cross multiple trajectories and targets")
    if len(trajectory_crossings) != len(trajectory_groups):
        raise ValueError("every trajectory must cross both target polarities")
    if any(overlaps.values()):
        raise ValueError("template text leaked across train/dev/test")
    return {
        "exact_input_target_conflicts": conflicts,
        "same_text_multi_trajectory_target_groups": len(text_crossings),
        "same_trajectory_opposite_polarity_groups": len(trajectory_crossings),
        "normalized_text_split_overlap": overlaps,
        "crossing_rule": (
            "same text crosses reward contexts; same trajectory crosses approval and disapproval"
        ),
    }


def generate_benchmark(
    output_dir: str | Path,
    *,
    overwrite: bool = False,
    training_config: dict | None = None,
) -> dict:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "train_dev": output_dir / "train_dev.json",
        "test": output_dir / "test.json",
        "manifest": output_dir / "benchmark_manifest.json",
    }
    existing = [path for path in paths.values() if path.exists()]
    if existing and not overwrite:
        raise FileExistsError("benchmark outputs already exist: " + ", ".join(map(str, existing)))

    features = sorted(load_features())
    rows = build_rows(features)
    audit = audit_rows(rows)
    train_dev_rows = [row for row in rows if row["split"] != "test"]
    test_rows = [row for row in rows if row["split"] == "test"]
    _write_json(
        paths["train_dev"],
        {"schema_version": SCHEMA_VERSION, "features": features, "examples": train_dev_rows},
    )
    _write_json(
        paths["test"],
        {"schema_version": SCHEMA_VERSION, "features": features, "examples": test_rows},
    )
    varying = sorted(
        {
            feature
            for row in rows
            for feature, value in row["teacher_reward_weights"].items()
            if abs(float(value)) > 1e-12
        }
    )
    split_counts = {
        split: sum(row["split"] == split for row in rows)
        for split in ("train", "dev", "test")
    }
    config = dict(TRAINING_CONFIG if training_config is None else training_config)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "claim_scope": (
            "synthetic trajectory-input identifiability diagnostic; not human-language accuracy"
        ),
        "feature_count": len(features),
        "feature_schema_sha256": hashlib.sha256(
            json.dumps(features, separators=(",", ":")).encode("utf-8")
        ).hexdigest(),
        "varying_features": varying,
        "varying_feature_count": len(varying),
        "context_count": len(benchmark_contexts()),
        "reward_config_count": len(benchmark_contexts()) * 2,
        "split_counts": split_counts,
        "audit": audit,
        "training_config": config,
        "files": {
            "train_dev": {
                "path": paths["train_dev"].name,
                "sha256": _file_sha256(paths["train_dev"]),
            },
            "test": {
                "path": paths["test"].name,
                "sha256": _file_sha256(paths["test"]),
            },
        },
        "validity_gate": {
            "minimum_full_relative_gain_over_text_only": 0.30,
            "minimum_full_relative_gain_over_trajectory_only": 0.30,
            "minimum_full_nearest_config_accuracy": 0.85,
            "minimum_full_active_sign_accuracy": 0.90,
            "minimum_config_accuracy_gain_over_each_ablation": 0.25,
        },
    }
    _write_json(paths["manifest"], manifest)
    manifest["manifest_path"] = str(paths["manifest"])
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    manifest = generate_benchmark(args.output_dir, overwrite=args.overwrite)
    print(f"Train/dev/test: {manifest['split_counts']}")
    print(f"Contexts: {manifest['context_count']}")
    print(f"Varying features: {manifest['varying_feature_count']}")
    print(f"Manifest: {manifest['manifest_path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
