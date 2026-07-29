"""Assemble the Route 2 (neural inference network) dataset -- up to training.

Reproduces the input-preparation stage of
``aaai_inference_network_training.ipynb`` (tokenize utterances, build the
vocabulary, assemble ``(tokens, trajectory feature counts, target reward)``
triples, and build grouped CV folds) -- but stops there. No model is trained.

Overcooked analogs of the paper's quantities:

- **tokens**: ``nn_tokenize(utterance)`` (paper ``nn_preprocess_chat_phrase``);
- **feature_counts**: the grounded reference vector over the 53-dim schema,
  normalized to sum 1 (paper: 15-dim normalized trajectory feature counts);
- **target_reward**: ``feature_counts * valence`` -- the reward the utterance
  implies (paper: the ground-truth conjunction reward vector). Without a real
  teacher-learner corpus this is a self-supervised analog, documented in
  DIFFERENCES_FROM_PAPER.md;
- **folds**: grouped by ``probe_id`` (held-out analog of teachers / reward
  configs).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.run_baseline_b_pipeline import DEFAULT_FEEDBACK_PATH  # noqa: E402
from src.feedback_observations import build_feedback_observations  # noqa: E402
from src.feature_schema import (  # noqa: E402
    collect_action_feature_library,
    load_features,
    read_json,
    write_json,
)
from src.feedback_form_classifier import classify_feedback  # noqa: E402
from src.evaluation_splits import (  # noqa: E402
    canonical_sha256,
    deduplicate_corpus,
    load_split_manifest,
    make_split_manifest,
    validate_split_manifest,
    write_split_manifest,
)
from src.neural_inference import (  # noqa: E402
    build_vocab,
    make_folds,
)
from src.observations import reference_vector  # noqa: E402
from src.probe_evaluator import DEFAULT_PROBE_STATES_PATH, load_probe_states  # noqa: E402
from src.text_analysis import nn_tokenize  # noqa: E402


def build_examples(
    feedback_examples: list[dict],
    probes: list[dict],
    features: list[str],
) -> list[dict]:
    """Assemble Route 2 training triples (no tensors, no training)."""

    ordered_features = sorted(features)
    action_library = collect_action_feature_library(probes)
    examples = []

    for feedback in feedback_examples:
        text = feedback.get("text") or ""
        feedback_type = feedback.get("expected_feedback_type") or classify_feedback(text)
        sub_observations = build_feedback_observations(
            feedback,
            feedback_type=feedback_type,
            action_feature_library=action_library,
        )
        merged: dict[str, float] = {}
        valences = []
        for sub in sub_observations:
            for feature, value in sub["target_features"].items():
                merged[feature] = merged.get(feature, 0.0) + float(value)
            valences.append(sub["valence"])
        valence = sum(valences) / len(valences) if valences else 0.0

        counts = reference_vector(merged, ordered_features, normalize=True)
        target_reward = counts * valence

        example = {
            "feedback_id": feedback.get("feedback_id"),
            # Prefer the synthetic scenario grouping; fall back to probe_id for
            # the legacy hand-authored corpus. Grouped CV holds out whole
            # scenarios / probes so the network is tested on unseen decisions.
            "group_id": feedback.get("group_id") or feedback.get("probe_id"),
            "feedback_type": feedback_type,
            "tokens": nn_tokenize(text),
            "feature_counts": counts.tolist(),
            "target_reward": target_reward.tolist(),
            "valence": valence,
        }
        # Carry subgoal-decision metadata through when the corpus provides it,
        # so downstream subgoal-accuracy evaluation can re-rank on held-out
        # scenarios without recomputing the teacher.
        for field in ("context", "feasible_subgoals", "expected_subgoal", "acceptable_subgoals"):
            if feedback.get(field) is not None:
                example[field] = feedback[field]
        examples.append(example)
    return examples


def build_dataset(
    feedback_examples: list[dict],
    probes: list[dict],
    features: list[str],
    *,
    min_freq: int = 1,
    n_folds: int = 5,
    seed: int = 0,
    dev_fraction: float = 0.2,
    test_fraction: float = 0.2,
    split_manifest: dict | str | Path | None = None,
    near_duplicate_threshold: float = 0.92,
) -> dict:
    ordered_features = sorted(features)
    feedback_examples, corpus_audit = deduplicate_corpus(
        feedback_examples,
        near_duplicate_threshold=near_duplicate_threshold,
    )
    examples = build_examples(feedback_examples, probes, features)
    group_ids = [example["group_id"] for example in examples]
    if split_manifest is None:
        split = make_split_manifest(
            feedback_examples,
            seed=seed,
            dev_fraction=dev_fraction,
            test_fraction=test_fraction,
            near_duplicate_threshold=near_duplicate_threshold,
        )
    elif isinstance(split_manifest, (str, Path)):
        split = load_split_manifest(split_manifest, feedback_examples)
    else:
        split = validate_split_manifest(split_manifest, feedback_examples)
    # The vocabulary is fitted on train only. Looking at held-out text before
    # evaluation is a small but real source of leakage.
    train_tokens = [examples[index]["tokens"] for index in split["train_indices"]]
    vocab = build_vocab(train_tokens, min_freq=min_freq)
    folds = make_folds(
        group_ids, n_folds=n_folds, seed=seed
    )
    return {
        "features": ordered_features,
        "n_features": len(ordered_features),
        "vocab": vocab,
        "vocab_size": len(vocab),
        "examples": examples,
        "folds": folds,
        "split": split,
        "split_manifest": split,
        "corpus_sha256": split["corpus_sha256"],
        "split_sha256": split["split_sha256"],
        "corpus_audit": corpus_audit,
        "split_policy": {
            "seed": split["seed"],
            "dev_fraction": split["dev_fraction"],
            "test_fraction": split["test_fraction"],
            "vocab_fit": "train_only",
            "grouping": split["grouping"]["policy"],
            "manifest_version": split["version"],
            "near_duplicate_threshold": near_duplicate_threshold,
        },
        "dataset_config_sha256": canonical_sha256(
            {
                "min_freq": min_freq,
                "n_folds": n_folds,
                "seed": seed,
                "dev_fraction": dev_fraction,
                "test_fraction": test_fraction,
                "near_duplicate_threshold": near_duplicate_threshold,
                "features": ordered_features,
            }
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feedback", type=Path, default=DEFAULT_FEEDBACK_PATH)
    parser.add_argument("--probe-states", type=Path, default=DEFAULT_PROBE_STATES_PATH)
    parser.add_argument("--min-freq", type=int, default=1)
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dev-fraction", type=float, default=0.2)
    parser.add_argument("--test-fraction", type=float, default=0.2)
    parser.add_argument(
        "--split-manifest",
        type=Path,
        help="Read and validate this fixed manifest instead of rebuilding a split.",
    )
    parser.add_argument(
        "--split-manifest-out",
        type=Path,
        help="Manifest output (default: split_manifest.json beside --output).",
    )
    parser.add_argument("--near-duplicate-threshold", type=float, default=0.92)
    parser.add_argument(
        "--output", type=Path, default=ROOT / "outputs" / "route2" / "dataset.json"
    )
    args = parser.parse_args()

    feedback_examples = read_json(args.feedback)
    probes = load_probe_states(args.probe_states)
    features = load_features()

    dataset = build_dataset(
        feedback_examples,
        probes,
        features,
        min_freq=args.min_freq,
        n_folds=args.n_folds,
        seed=args.seed,
        dev_fraction=args.dev_fraction,
        test_fraction=args.test_fraction,
        split_manifest=args.split_manifest,
        near_duplicate_threshold=args.near_duplicate_threshold,
    )
    write_json(args.output, dataset)
    manifest_out = args.split_manifest_out or args.output.with_name("split_manifest.json")
    write_split_manifest(manifest_out, dataset["split_manifest"])

    print("Route 2 dataset assembled (no training performed):")
    print(f"  examples:   {len(dataset['examples'])}")
    print(f"  features:   {dataset['n_features']}")
    print(f"  vocab size: {dataset['vocab_size']}")
    print(f"  corpus SHA256: {dataset['corpus_sha256']}")
    print(f"  split SHA256:  {dataset['split_sha256']}")
    print(
        "  corpus audit: "
        f"removed={dataset['corpus_audit']['exact_duplicates_removed']} "
        f"supervision_conflicts={dataset['corpus_audit']['supervision_conflict_count']} "
        f"near_pairs={dataset['corpus_audit']['near_duplicates']['count']}"
    )
    print(f"  folds:      {len(dataset['folds'])}")
    print(
        "  fixed split: "
        f"train={len(dataset['split']['train_indices'])} "
        f"dev={len(dataset['split']['dev_indices'])} "
        f"test={len(dataset['split']['test_indices'])}"
    )
    for fold in dataset["folds"]:
        print(
            f"    fold {fold['fold']}: "
            f"train={len(fold['train_indices'])} test={len(fold['test_indices'])} "
            f"held-out groups={fold['test_groups']}"
        )
    print(f"  dataset JSON: {args.output}")
    print(f"  split manifest: {manifest_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
