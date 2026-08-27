"""Assemble the Route 2 neural-inference dataset.

This reproduces the paper's tokenization, vocabulary, normalized trajectory
feature counts, reward-vector regression triples, and grouped folds. The main
paper-aligned synthetic corpus targets the complete 53-dimensional hidden
teacher reward. Its stable teacher and reward axes are split only by the paper
cross-validation manifest. Legacy local-feedback corpora remain supported as
an explicit ablation with ``feature_counts * valence`` targets.
"""

from __future__ import annotations

import argparse
import hashlib
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
    SPLIT_MANIFEST_VERSION,
    canonical_sha256,
    deduplicate_corpus,
    load_split_manifest,
    make_declared_split_manifest,
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


INDEPENDENT_HUMAN_TARGET_SOURCES = frozenset(
    {"experiment_assigned_reward_config", "independent_reward_annotation"}
)
PROHIBITED_TARGET_TERMS = (
    "model_prediction",
    "prediction",
    "pseudo",
    "recovered",
    "posterior",
    "route2_inference",
)


def audit_full_reward_supervision(
    feedback_examples: list[dict],
    features: list[str],
) -> dict:
    """Fail closed when an identical Route 2 input has different full rewards.

    A per-example regressor cannot learn two different complete reward vectors
    from the same language and trajectory.  Treating those rows as ordinary
    examples silently turns the target into an arbitrary conditional mean.
    """

    ordered_features = sorted(features)
    seen: dict[tuple[bytes, tuple[tuple[int, float], ...]], tuple[tuple[float, ...], int]] = {}
    target_cache: dict[str, tuple[float, ...]] = {}
    consistent_duplicates = 0
    conflicts: list[dict] = []
    for index, example in enumerate(feedback_examples):
        text = " ".join(str(example.get("text") or "").casefold().split())
        trajectory = example.get("route2_trajectory_features") or {}
        input_signature = (
            hashlib.sha256(text.encode("utf-8")).digest(),
            tuple(
                (feature_index, rounded)
                for feature_index, feature in enumerate(ordered_features)
                if (rounded := round(float(trajectory.get(feature, 0.0)), 12))
            ),
        )
        weights = example.get("teacher_reward_weights") or {}
        reward_id = str(example.get("reward_config_id") or f"weights_object:{id(weights)}")
        computed_target = tuple(
            round(float(weights.get(feature, 0.0)), 12) for feature in ordered_features
        )
        target = target_cache.setdefault(reward_id, computed_target)
        if target != computed_target:
            raise ValueError(
                f"reward_config_id {reward_id!r} has inconsistent complete weights"
            )
        previous = seen.get(input_signature)
        if previous is None:
            seen[input_signature] = (target, index)
            continue
        previous_target, previous_index = previous
        if previous_target == target:
            consistent_duplicates += 1
            continue
        conflicts.append(
            {
                "first_index": previous_index,
                "second_index": index,
                "first_feedback_id": feedback_examples[previous_index].get("feedback_id"),
                "second_feedback_id": example.get("feedback_id"),
            }
        )
        if len(conflicts) >= 10:
            break

    if conflicts:
        sample = ", ".join(
            f"{row['first_feedback_id']!r}<->{row['second_feedback_id']!r}"
            for row in conflicts[:3]
        )
        raise ValueError(
            "conflicting complete Route 2 supervision: the same normalized "
            "(text, trajectory_features) maps to different teacher reward vectors; "
            f"examples: {sample}. Regenerate reward-conditioned language instead "
            "of training on an unidentifiable target."
        )

    return {
        "input_count": len(feedback_examples),
        "output_count": len(feedback_examples),
        "unique_input_signatures": len(seen),
        "consistent_duplicate_inputs": consistent_duplicates,
        "exact_duplicates_removed": 0,
        "supervision_conflict_count": 0,
        "near_duplicates": {
            "count": None,
            "pair_indices": [],
            "audit": "not computed for declared multi-reward corpus",
        },
    }


def _make_route2_seed_split_manifest(
    feedback_examples: list[dict],
    model_examples: list[dict],
    *,
    n_folds: int,
    seed: int,
) -> dict:
    """Create a lightweight dual-axis split for non-CV callers.

    The formal trainer creates all folds separately.  This seed fold keeps the
    generic dataset/training API usable without applying unrelated scenario or
    near-duplicate splitting to the fully crossed teacher/reward corpus.
    """

    from src.paper_cross_validation import make_paper_cross_validation_manifest

    teacher_count = len({str(example.get("teacher_id")) for example in model_examples})
    reward_count = len(
        {str(example.get("reward_config_id")) for example in model_examples}
    )
    fold_count = min(max(3, n_folds), teacher_count, reward_count)
    if fold_count < 3:
        raise ValueError(
            "full-reward Route 2 data needs at least three teachers and reward "
            "configurations for a leakage-safe seed split"
        )
    paper = make_paper_cross_validation_manifest(
        model_examples, n_folds=fold_count, seed=seed
    )
    fold = paper["folds"][0]
    sections = {}
    for name in ("train", "dev", "test"):
        indices = list(fold[f"{name}_indices"])
        sections[name] = {
            "indices": indices,
            "groups": sorted(
                {str(model_examples[index].get("group_id")) for index in indices}
            ),
            "template_families": [],
            "reward_configs": list(fold[f"{name}_rewards"]),
            "teacher_ids": list(fold[f"{name}_teachers"]),
            "components": [],
        }
    manifest = {
        "version": SPLIT_MANIFEST_VERSION,
        "corpus_sha256": canonical_sha256(feedback_examples),
        "grouping": {
            "policy": "paper_seed_teacher_and_reward_holdout",
            "holdout_axis": "teacher_id_and_reward_config_id",
            "disjoint_fields": ["teacher_ids", "reward_configs"],
            "near_duplicate_grouping_threshold": None,
        },
        "seed": seed,
        "dev_fraction": None,
        "test_fraction": None,
        "splits": sections,
        "excluded_indices": list(fold["excluded_indices"]),
        "near_duplicate_exclusions": [],
        **{
            f"{name}_indices": sections[name]["indices"]
            for name in ("train", "dev", "test")
        },
        **{
            f"{name}_groups": sections[name]["groups"]
            for name in ("train", "dev", "test")
        },
        "conflict_audit": {
            "exact_normalized_text_overlap_count": None,
            "exact_normalized_text_overlaps": [],
            "near_duplicate_audit": (
                "not applicable; formal isolation is by teacher and reward axes"
            ),
        },
        "paper_cross_validation_manifest_sha256": paper["manifest_sha256"],
    }
    manifest["split_sha256"] = canonical_sha256(manifest)
    return validate_split_manifest(manifest, feedback_examples)


def build_examples(
    feedback_examples: list[dict],
    probes: list[dict],
    features: list[str],
) -> list[dict]:
    """Assemble local or full-teacher Route 2 training triples."""

    ordered_features = sorted(features)
    action_library = collect_action_feature_library(probes)
    examples = []
    full_target_cache: dict[str, list[float]] = {}

    for feedback in feedback_examples:
        text = feedback.get("text") or ""
        feedback_type = feedback.get("expected_feedback_type") or classify_feedback(text)
        teacher_weights = feedback.get("teacher_reward_weights")
        full_teacher_target = isinstance(teacher_weights, dict)

        if full_teacher_target:
            trajectory_features = feedback.get("route2_trajectory_features") or {}
            counts = reference_vector(
                {
                    str(feature): float(value)
                    for feature, value in trajectory_features.items()
                },
                ordered_features,
                normalize=True,
            )
            reward_id = str(
                feedback.get("reward_config_id")
                or f"weights_object:{id(teacher_weights)}"
            )
            target_reward = full_target_cache.get(reward_id)
            if target_reward is None:
                target_reward = [
                    float(teacher_weights.get(feature, 0.0))
                    for feature in ordered_features
                ]
                full_target_cache[reward_id] = target_reward
            valence = float(feedback.get("attributed_sentiment_score", 0.0))
            target_mode = "full_teacher_reward"
        else:
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
            target_reward = (counts * valence).tolist()
            target_mode = "local_feedback"

        example = {
            "feedback_id": feedback.get("feedback_id"),
            "text": text,
            "group_id": feedback.get("group_id") or feedback.get("probe_id"),
            "feedback_type": feedback_type,
            "tokens": nn_tokenize(text),
            "feature_counts": counts.tolist(),
            "target_reward": target_reward,
            "target_mode": target_mode,
            "valence": valence,
        }
        for field in (
            "context",
            "feasible_subgoals",
            "expected_subgoal",
            "acceptable_subgoals",
            "referenced_subgoal",
            "reward_config_id",
            "synthetic_teacher_style_id",
            "feedback_form_style_id",
            "author_language_style",
            "teacher_id",
            "cv_teacher_id",
            "split",
            "source",
            "session_id",
            "target_provenance",
            "supervision_status",
            "eligible_for_route2_supervised_training",
        ):
            if feedback.get(field) is not None:
                example[field] = feedback[field]
        examples.append(example)
    return examples


def expand_feedback_payload(payload: list[dict] | dict) -> list[dict]:
    """Expand compact multi-reward corpora into ordinary feedback rows."""

    if isinstance(payload, list):
        return list(payload)
    if not isinstance(payload, dict) or not isinstance(payload.get("examples"), list):
        raise ValueError("feedback corpus must be a list or a packaged examples object")
    configurations = payload.get("reward_configurations") or []
    by_id = {
        str(config["reward_config_id"]): config
        for config in configurations
        if isinstance(config, dict) and config.get("reward_config_id")
    }
    contexts = payload.get("contexts") or []
    contexts_by_id = {
        str(context["group_id"]): context
        for context in contexts
        if isinstance(context, dict) and context.get("group_id") is not None
    }
    expanded = []
    for source in payload["examples"]:
        row = dict(source)
        if not isinstance(row.get("teacher_reward_weights"), dict):
            config_id = str(row.get("reward_config_id") or "")
            config = by_id.get(config_id)
            if config is None or not isinstance(config.get("weights"), dict):
                raise ValueError(f"missing reward configuration {config_id!r}")
            row["teacher_reward_weights"] = config["weights"]
        if row.get("context") is None:
            group_id = str(row.get("group_id") or "")
            context = contexts_by_id.get(group_id)
            if context is None:
                raise ValueError(f"missing context {group_id!r}")
            row["context"] = context["context"]
            row["feasible_subgoals"] = context["feasible_subgoals"]
        expanded.append(row)
    return expanded

def build_dataset(
    feedback_examples: list[dict] | dict,
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
    grouping_policy: str = "joint",
    paper_cv_mode: bool = False,
    source_corpus_sha256: str | None = None,
) -> dict:
    ordered_features = sorted(features)
    feedback_examples = expand_feedback_payload(feedback_examples)
    explicitly_unlabeled = [
        example.get("feedback_id") or f"item[{index}]"
        for index, example in enumerate(feedback_examples)
        if example.get("eligible_for_route2_supervised_training") is False
        or str(example.get("supervision_status") or "").startswith("unlabeled_")
    ]
    if explicitly_unlabeled:
        raise ValueError(
            "Route 2 supervised dataset contains unlabeled human language; keep it "
            "in the unlabeled pool instead of inventing reward targets: "
            + ", ".join(str(value) for value in explicitly_unlabeled[:10])
        )
    for index, example in enumerate(feedback_examples):
        provenance = str(example.get("target_provenance") or "").strip()
        lowered = provenance.lower()
        if provenance and any(term in lowered for term in PROHIBITED_TARGET_TERMS):
            raise ValueError(
                f"item[{index}]: model predictions/recovered posteriors cannot be reward gold"
            )
        if str(example.get("source") or "").startswith("human") and isinstance(
            example.get("teacher_reward_weights"), dict
        ):
            if provenance not in INDEPENDENT_HUMAN_TARGET_SOURCES:
                raise ValueError(
                    f"item[{index}]: human Route 2 targets require independent provenance"
                )
            if not example.get("reward_config_id") or not example.get("teacher_id"):
                raise ValueError(
                    f"item[{index}]: human Route 2 targets require teacher_id and reward_config_id"
                )
    full_target_flags = [
        isinstance(example.get("teacher_reward_weights"), dict)
        for example in feedback_examples
    ]
    if any(full_target_flags) and not all(full_target_flags):
        raise ValueError("Route 2 corpus mixes local and full-teacher targets")
    target_mode = (
        "full_teacher_reward" if full_target_flags and all(full_target_flags) else "local_feedback"
    )

    if target_mode == "full_teacher_reward":
        feedback_examples = list(feedback_examples)
        corpus_audit = audit_full_reward_supervision(feedback_examples, ordered_features)
        near_duplicate_pairs: list[tuple[int, int, float]] = []
    else:
        feedback_examples, corpus_audit = deduplicate_corpus(
            feedback_examples,
            near_duplicate_threshold=near_duplicate_threshold,
        )
        near_duplicate_pairs = [
            (int(left), int(right), float(score))
            for left, right, score in corpus_audit["near_duplicates"]["pair_indices"]
        ]

    examples = build_examples(feedback_examples, probes, features)
    reward_configs = {
        str(example.get("reward_config_id"))
        for example in examples
        if example.get("reward_config_id") is not None
    }
    teacher_ids = {
        str(example.get("teacher_id"))
        for example in examples
        if example.get("teacher_id") is not None
    }
    feedback_form_styles = {
        str(example.get("feedback_form_style_id"))
        for example in examples
        if example.get("feedback_form_style_id") is not None
    }
    if paper_cv_mode:
        if target_mode != "full_teacher_reward":
            raise ValueError("paper_cv_mode requires full teacher rewards")
        if not source_corpus_sha256:
            raise ValueError(
                "paper_cv_mode requires source_corpus_sha256 to avoid expensive "
                "canonical serialization of the expanded corpus"
            )
        return {
            "features": ordered_features,
            "n_features": len(ordered_features),
            "vocab": {},
            "vocab_size": 0,
            "examples": examples,
            "folds": [],
            "split": None,
            "split_manifest": None,
            "target_mode": target_mode,
            "use_feature_counts": True,
            "n_reward_configs": len(reward_configs),
            "n_teachers": len(teacher_ids),
            "n_teacher_styles": len(teacher_ids),
            "n_feedback_form_styles": len(feedback_form_styles),
            "corpus_sha256": source_corpus_sha256,
            "split_sha256": None,
            "corpus_audit": corpus_audit,
            "paper_cv_mode": True,
            "split_policy": {
                "policy": "deferred_to_paper_cross_validation_manifest",
                "vocab_fit": "per_fold_train_only",
                "holdout_axis": "teacher_id_and_reward_config_id",
            },
            "dataset_config_sha256": canonical_sha256(
                {
                    "paper_cv_mode": True,
                    "source_corpus_sha256": source_corpus_sha256,
                    "target_mode": target_mode,
                    "features": ordered_features,
                }
            ),
        }
    group_ids = [
        example.get("reward_config_id") or example["group_id"]
        for example in examples
    ]
    if split_manifest is None:
        declared = target_mode == "full_teacher_reward" and all(
            example.get("split") in {"train", "dev", "test"}
            for example in feedback_examples
        )
        split = (
            make_declared_split_manifest(feedback_examples)
            if declared
            else _make_route2_seed_split_manifest(
                feedback_examples,
                examples,
                n_folds=n_folds,
                seed=seed,
            )
            if target_mode == "full_teacher_reward"
            else make_split_manifest(
                feedback_examples,
                seed=seed,
                dev_fraction=dev_fraction,
                test_fraction=test_fraction,
                near_duplicate_threshold=near_duplicate_threshold,
                grouping_policy=grouping_policy,
                near_duplicate_pairs=near_duplicate_pairs,
            )
        )
    elif isinstance(split_manifest, (str, Path)):
        split = load_split_manifest(split_manifest, feedback_examples)
    else:
        split = validate_split_manifest(split_manifest, feedback_examples)

    train_tokens = [examples[index]["tokens"] for index in split["train_indices"]]
    vocab = build_vocab(train_tokens, min_freq=min_freq)
    folds = make_folds(group_ids, n_folds=n_folds, seed=seed)
    return {
        "features": ordered_features,
        "n_features": len(ordered_features),
        "vocab": vocab,
        "vocab_size": len(vocab),
        "examples": examples,
        "folds": folds,
        "split": split,
        "split_manifest": split,
        "target_mode": target_mode,
        "use_feature_counts": target_mode == "full_teacher_reward",
        "n_reward_configs": len(reward_configs),
        "n_teachers": len(teacher_ids),
        "n_teacher_styles": len(teacher_ids),
        "n_feedback_form_styles": len(feedback_form_styles),
        "corpus_sha256": split["corpus_sha256"],
        "split_sha256": split["split_sha256"],
        "corpus_audit": corpus_audit,
        "split_policy": {
            "seed": split["seed"],
            "dev_fraction": split["dev_fraction"],
            "test_fraction": split["test_fraction"],
            "vocab_fit": "train_only",
            "grouping": split["grouping"]["policy"],
            "holdout_axis": split["grouping"].get("holdout_axis", "joint"),
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
                "grouping_policy": grouping_policy,
                "target_mode": target_mode,
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
        "--paper-cv-mode",
        action="store_true",
        help="Skip generic split/fold/vocab work; formal CV owns those artifacts.",
    )
    parser.add_argument(
        "--grouping-policy",
        choices=("joint", "scenario", "template"),
        default="joint",
        help="Hold out strict joint components, scenarios, or template families.",
    )
    parser.add_argument(
        "--output", type=Path, default=ROOT / "outputs" / "route2" / "dataset.json"
    )
    args = parser.parse_args()

    raw = args.feedback.read_bytes()
    feedback_examples = json.loads(raw.decode("utf-8"))
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
        grouping_policy=args.grouping_policy,
        paper_cv_mode=args.paper_cv_mode,
        source_corpus_sha256=(
            hashlib.sha256(raw).hexdigest() if args.paper_cv_mode else None
        ),
    )
    write_json(args.output, dataset)
    if args.paper_cv_mode:
        print("Route 2 paper-CV examples assembled (split/vocab deferred):")
        print(f"  examples:   {len(dataset['examples'])}")
        print(f"  features:   {dataset['n_features']}")
        print(f"  teachers:   {dataset['n_teachers']}")
        print(f"  rewards:    {dataset['n_reward_configs']}")
        print(f"  corpus SHA256: {dataset['corpus_sha256']}")
        print(f"  dataset JSON: {args.output}")
        return 0
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
