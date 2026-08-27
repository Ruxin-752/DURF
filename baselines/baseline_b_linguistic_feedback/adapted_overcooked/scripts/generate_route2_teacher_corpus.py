"""Generate paper-aligned synthetic Route 2 supervision without human data.

Each row contains language, an actually referenced trajectory feature vector,
a complete hidden teacher reward vector, and explicit reward/style identifiers.
Each base feedback row has one stable synthetic author and is crossed with all
reward configurations.  Only the paper cross-validation manifest owns splits.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.enumerate_subgoal_contexts import enumerate_contexts  # noqa: E402
from scripts.generate_synthetic_feedback import build_corpus  # noqa: E402
from scripts.prepare_route2_dataset import audit_full_reward_supervision  # noqa: E402
from src.feature_schema import (  # noqa: E402
    load_features,
    read_json,
    validate_feedback_examples,
    write_json,
)
from src.probe_evaluator import DEFAULT_PROBE_STATES_PATH, load_probe_states  # noqa: E402
from src.reward_configurations import (  # noqa: E402
    PROACTIVITY_LABELS,
    sample_reward_configurations,
    validate_reward_configurations,
)
from src.subgoal_featurizer import featurize_subgoal  # noqa: E402
from src.subgoal_teacher import (  # noqa: E402
    GROUNDED_REFERENCE_TYPES,
    SPEECH_ACTS,
    label_context,
    load_gold_weights,
)


DEFAULT_OUTPUT = ROOT / "data" / "route2_teacher_feedback.paper_v5.synthetic.json"
GENERATOR_VERSION = "route2-multiteacher-v5-identifiable-local-feedback"
SPLITS = ("train", "dev", "test")
COOKING_PREPARATION_SUBGOALS = frozenset({"GET_TOMATO", "GET_ONION", "GET_DISH"})
PROFILE_ADJECTIVES = {
    "path_coordination": {0.25: "maneuverable", 1.0: "coordinated", 1.75: "unobstructed"},
    "division_of_labor": {0.25: "independent", 1.0: "cooperative", 1.75: "complementary"},
    "target_respect": {0.25: "opportunistic", 1.0: "considerate", 1.75: "deferential"},
    "efficiency": {0.25: "unhurried", 1.0: "steady", 1.75: "urgent"},
    "proactivity": dict(PROACTIVITY_LABELS),
}
REWARD_CONDITIONING_PHRASES = {
    "path_coordination": {
        0.25: "leaving some room to maneuver",
        1.0: "keeping our lanes coordinated",
        1.75: "keeping every lane completely clear",
    },
    "division_of_labor": {
        0.25: "letting each cook work independently",
        1.0: "sharing the work cooperatively",
        1.75: "strictly separating our responsibilities",
    },
    "target_respect": {
        0.25: "using any available target opportunistically",
        1.0: "respecting claimed targets",
        1.75: "always deferring to a claimed target",
    },
    "efficiency": {
        0.25: "keeping an unhurried pace",
        1.0: "maintaining a steady pace",
        1.75: "serving with urgency",
    },
    "proactivity": {
        -1.0: "preparing only when an item is needed",
        0.0: "keeping preparation just in time",
        1.0: "preparing the next step early",
    },
}
AUTHOR_STYLE_PREFIXES = (
    "Quick note:",
    "For coordination:",
    "My practical suggestion:",
    "Just so we're aligned:",
    "Direct feedback:",
    "From my side:",
    "One kitchen thought:",
    "To be clear:",
    "A small request:",
    "Here's what I noticed:",
    "For next time:",
    "Team note:",
)


def verbalize_reward_profile(configuration: dict) -> str:
    """Encode each preference/strength pair as one compositional word."""

    multipliers = configuration["multipliers"]
    adjectives = [
        PROFILE_ADJECTIVES[group][float(multipliers[group])]
        for group in PROFILE_ADJECTIVES
    ]
    return "My overall kitchen style is " + ", ".join(adjectives[:-1]) + f", and {adjectives[-1]}."


def condition_local_feedback(text: str, configuration: dict) -> str:
    """Make local feedback identifiable under a hidden reward configuration.

    The rationale communicates preferences in ordinary language.  It never
    exposes a reward ID, feature name, vector coordinate, or numeric weight.
    This is the synthetic analogue of the paper's reward-conditioned token
    switching: changing the hidden reward changes the observable utterance.
    """

    multipliers = configuration["multipliers"]
    clauses = [
        REWARD_CONDITIONING_PHRASES[group][float(multipliers[group])]
        for group in REWARD_CONDITIONING_PHRASES
    ]
    action = str(text).strip().rstrip(".!?")
    rationale = ", ".join(clauses[:-1]) + f", and {clauses[-1]}"
    return f"{action}. I say that because I prefer {rationale}."


def apply_author_language_style(text: str, author_id: str) -> str:
    """Apply a stable author-specific surface style unrelated to reward/form."""

    try:
        index = int(author_id.rsplit("_", 1)[1])
    except (IndexError, ValueError) as error:
        raise ValueError(f"invalid synthetic author ID: {author_id!r}") from error
    return f"{AUTHOR_STYLE_PREFIXES[index]} {str(text).strip()}"


def _partition_identifiers(
    identifiers: list[str],
    *,
    train_count: int,
    dev_count: int,
    seed: int,
) -> dict[str, str]:
    shuffled = list(identifiers)
    random.Random(seed).shuffle(shuffled)
    assignment: dict[str, str] = {}
    for index, identifier in enumerate(shuffled):
        split = "train" if index < train_count else "dev" if index < train_count + dev_count else "test"
        assignment[identifier] = split
    return assignment


def _is_empty_hand_cooking_context(context: dict) -> bool:
    facts = context.get("context") or {}
    return (
        facts.get("pot_status") == "cooking"
        and facts.get("agent_holding") is None
        and bool(
            set(context.get("feasible_subgoals") or ())
            & COOKING_PREPARATION_SUBGOALS
        )
    )


def load_generation_contexts(
    contexts_path: Path | None, weights: dict[str, float]
) -> list[dict]:
    """Re-enumerate by default; only an explicit path may select cached contexts."""

    return read_json(contexts_path) if contexts_path is not None else enumerate_contexts(weights)


def _context_split_assignment(contexts: list[dict], *, seed: int) -> dict[str, str]:
    by_id = {
        str(context.get("group_id") or context.get("scenario_id")): context
        for context in contexts
    }
    identifiers = sorted(by_id)
    total = len(identifiers)
    if not total:
        raise ValueError("at least one context is required")
    if total < 3:
        return {
            identifier: SPLITS[index]
            for index, identifier in enumerate(identifiers)
        }
    train_count = max(1, round(total * 0.6))
    dev_count = max(1, round(total * 0.2))
    if train_count + dev_count >= total:
        train_count = total - dev_count - 1
    assignment = _partition_identifiers(
        identifiers,
        train_count=train_count,
        dev_count=dev_count,
        seed=seed,
    )

    # Preserve split sizes while guaranteeing that a sufficiently large
    # cooking-preparation family is represented in train, dev, and test.
    preparation_ids = sorted(
        identifier
        for identifier, context in by_id.items()
        if _is_empty_hand_cooking_context(context)
    )
    if len(preparation_ids) >= len(SPLITS):
        preparation_set = set(preparation_ids)
        for split in SPLITS:
            if any(assignment[identifier] == split for identifier in preparation_ids):
                continue
            donor = next(
                (
                    identifier
                    for identifier in preparation_ids
                    if sum(
                        assignment[candidate] == assignment[identifier]
                        for candidate in preparation_ids
                    )
                    > 1
                ),
                None,
            )
            receiver = next(
                (
                    identifier
                    for identifier in identifiers
                    if assignment[identifier] == split
                    and identifier not in preparation_set
                ),
                None,
            )
            if donor is not None and receiver is not None:
                assignment[donor], assignment[receiver] = (
                    assignment[receiver],
                    assignment[donor],
                )
    return assignment


def _reward_split_assignment(
    configurations: list[dict],
    *,
    seed: int,
    gold_split: str,
) -> dict[str, str]:
    identifiers = [str(config["reward_config_id"]) for config in configurations]
    gold = "reward_00_gold"
    others = [identifier for identifier in identifiers if identifier != gold]
    random.Random(seed).shuffle(others)
    counts = {"train": 24, "dev": 6, "test": 6}
    counts[gold_split] -= 1
    if counts[gold_split] < 0 or sum(counts.values()) != len(others):
        # General fallback for non-default configuration counts.
        total = len(identifiers)
        train = max(1, round(total * 2 / 3))
        dev = max(1, round(total / 6))
        counts = {"train": train, "dev": dev, "test": total - train - dev}
        counts[gold_split] -= 1
    assignment = {gold: gold_split}
    cursor = 0
    for split in SPLITS:
        count = counts[split]
        for identifier in others[cursor : cursor + count]:
            assignment[identifier] = split
        cursor += count
    if cursor != len(others) or set(assignment) != set(identifiers):
        raise ValueError("reward configuration split allocation failed")

    # Every sufficiently large split should contain cautious, neutral, and
    # proactive hidden rewards. Swap whole configuration IDs so split sizes and
    # reward-disjointness remain unchanged; the canonical gold stays held out.
    style_by_id = {
        str(config["reward_config_id"]): str(config["preparation_style"])
        for config in configurations
    }
    styles = set(PROACTIVITY_LABELS.values())
    if (
        all(
            sum(style == target for style in style_by_id.values()) >= len(SPLITS)
            for target in styles
        )
        and all(
            sum(value == split for value in assignment.values()) >= len(styles)
            for split in SPLITS
        )
    ):
        for split in SPLITS:
            for style in sorted(styles):
                if any(
                    assignment[identifier] == split and style_by_id[identifier] == style
                    for identifier in identifiers
                ):
                    continue
                donor = next(
                    (
                        identifier
                        for identifier in sorted(identifiers)
                        if identifier != gold
                        and style_by_id[identifier] == style
                        and sum(
                            assignment[candidate] == assignment[identifier]
                            and style_by_id[candidate] == style
                            for candidate in identifiers
                        )
                        > 1
                    ),
                    None,
                )
                receiver = next(
                    (
                        identifier
                        for identifier in sorted(identifiers)
                        if identifier != gold
                        and assignment[identifier] == split
                        and sum(
                            assignment[candidate] == split
                            and style_by_id[candidate] == style_by_id[identifier]
                            for candidate in identifiers
                        )
                        > 1
                    ),
                    None,
                )
                if donor is not None and receiver is not None:
                    assignment[donor], assignment[receiver] = (
                        assignment[receiver],
                        assignment[donor],
                    )
    return assignment


def _style_identifiers() -> list[str]:
    return [
        f"style_{reference_type}_{speech_act}"
        for reference_type in GROUNDED_REFERENCE_TYPES
        for speech_act in SPEECH_ACTS
    ]


def _author_identifiers() -> list[str]:
    return [
        f"synthetic_author_{index:02d}"
        for index in range(len(AUTHOR_STYLE_PREFIXES))
    ]


def _is_cooking_preparation(row: dict) -> bool:
    context = row.get("context") or {}
    return (
        context.get("pot_status") == "cooking"
        and row.get("referenced_subgoal") in COOKING_PREPARATION_SUBGOALS
    )


def build_teacher_corpus(
    *,
    contexts: list[dict],
    configurations: list[dict],
    seed: int = 0,
    gold_split: str = "test",
    include_profile_language: bool = False,
) -> tuple[list[dict], dict]:
    if gold_split not in SPLITS:
        raise ValueError(f"gold_split must be one of {SPLITS}")
    validate_reward_configurations(configurations)
    features = load_features()
    feedback_form_styles = _style_identifiers()
    authors = _author_identifiers()
    author_order = list(authors)
    random.Random(seed + 1).shuffle(author_order)
    context_order = {
        str(context.get("group_id") or context.get("scenario_id")): index
        for index, context in enumerate(contexts)
    }
    reference_order = {
        reference_type: index
        for index, reference_type in enumerate(GROUNDED_REFERENCE_TYPES)
    }
    speech_order = {speech_act: index for index, speech_act in enumerate(SPEECH_ACTS)}

    def author_for(row: dict, slot_ordinal: int) -> str:
        # This deterministic base-feedback assignment excludes reward,
        # sentiment, referenced subgoal, and label outcome.  The context offset
        # rotates form assignments so an author is not synonymous with one
        # reference/speech label.
        group_id = str(row.get("group_id") or "")
        form_index = (
            reference_order[str(row["reference_type"])] * len(SPEECH_ACTS)
            + speech_order[str(row["expected_feedback_type"])]
        )
        author_index = (
            context_order[group_id] * 5 + form_index + slot_ordinal * 7
        ) % len(author_order)
        return author_order[author_index]

    canonical_config = next(
        (
            config
            for config in configurations
            if str(config["reward_config_id"]) == "reward_00_gold"
        ),
        configurations[0],
    )
    canonical_weights = {
        feature: float(canonical_config["weights"][feature]) for feature in features
    }
    base_rows = build_corpus(
        contexts=contexts,
        weights=canonical_weights,
        use_llm=False,
        seed=seed,
    )

    def slot_key(row: dict) -> tuple[str, str, str, int]:
        sentiment = float(row.get("attributed_sentiment_score", 0.0))
        polarity = 1 if sentiment > 0 else -1 if sentiment < 0 else 0
        return (
            str(row.get("group_id") or ""),
            str(row.get("reference_type") or ""),
            str(row.get("expected_feedback_type") or ""),
            polarity,
        )

    slot_cursors: dict[tuple[str, str, str, int], int] = defaultdict(int)
    base_records = []
    for base_row in base_rows:
        key = slot_key(base_row)
        ordinal = slot_cursors[key]
        slot_cursors[key] += 1
        base_records.append((base_row, key, ordinal))

    output: list[dict] = []
    for config in configurations:
        config_id = str(config["reward_config_id"])
        weights = {feature: float(config["weights"][feature]) for feature in features}
        rows = build_corpus(
            contexts=contexts,
            weights=weights,
            use_llm=False,
            seed=seed,
        )
        candidates: dict[tuple[str, str, str, int], list[dict]] = defaultdict(list)
        for row in rows:
            candidates[slot_key(row)].append(row)
        for base_row, key, ordinal in base_records:
            matching = candidates.get(key) or [base_row]
            row = matching[ordinal % len(matching)]
            feedback_form_style_id = (
                f"style_{row['reference_type']}_{row['expected_feedback_type']}"
            )
            teacher_id = author_for(base_row, ordinal)
            context = row["context"]
            feasible = row["feasible_subgoals"]
            label = label_context(weights, context, feasible)
            trajectory = featurize_subgoal(context, row["referenced_subgoal"])
            unconditioned_local_text = row["text"]
            reward_conditioned_text = condition_local_feedback(
                unconditioned_local_text, config
            )
            local_text = apply_author_language_style(
                reward_conditioned_text, teacher_id
            )
            profile_text = verbalize_reward_profile(config)
            enriched = dict(row)
            # Split ownership belongs exclusively to paper_cross_validation;
            # template-source split tags would recreate disconnected axes.
            enriched.pop("split", None)
            enriched.update(
                {
                    "feedback_id": f"{config_id}::{base_row['feedback_id']}",
                    "base_feedback_id": base_row["feedback_id"],
                    "text": (
                        f"{profile_text} {local_text}"
                        if include_profile_language
                        else local_text
                    ),
                    "local_text": local_text,
                    "unconditioned_local_text": unconditioned_local_text,
                    "reward_conditioned_text": reward_conditioned_text,
                    "reward_profile_text": profile_text,
                    "reward_config_id": config_id,
                    "teacher_id": teacher_id,
                    "author_language_style": AUTHOR_STYLE_PREFIXES[
                        int(teacher_id.rsplit("_", 1)[1])
                    ],
                    "feedback_form_style_id": feedback_form_style_id,
                    "teacher_reward_weights": weights,
                    "route2_trajectory_features": {
                        feature: float(value) for feature, value in trajectory.items()
                    },
                    "target_mode": "full_teacher_reward",
                    "reward_profile": config["profile"],
                    "reward_multipliers": config["multipliers"],
                    "preparation_style": config["preparation_style"],
                    "expected_subgoal": label["expected_subgoal"],
                    "acceptable_subgoals": label["acceptable_subgoals"],
                    "paraphrase_family": (
                        f"{base_row['role']}|{base_row['feedback_id']}|{feedback_form_style_id}"
                    ),
                    "generator_version": GENERATOR_VERSION,
                    "reward_conditioning": "semantic_local_rationale_v1",
                }
            )
            output.append(enriched)

    corpus_audit = audit_full_reward_supervision(output, features)
    augmentation_counts = Counter(row["base_feedback_id"] for row in output)
    expected_augmentations = len(configurations)
    if not augmentation_counts or set(augmentation_counts.values()) != {
        expected_augmentations
    }:
        raise ValueError("every base feedback row must be augmented over every reward config")
    trajectory_signatures: dict[str, set[tuple]] = defaultdict(set)
    for row in output:
        trajectory_signatures[row["base_feedback_id"]].add(
            tuple(sorted(row["route2_trajectory_features"].items()))
        )
    trajectory_variation = {
        base_feedback_id: len(signatures)
        for base_feedback_id, signatures in trajectory_signatures.items()
    }
    author_coverage = {
        author: {
            "examples": sum(row["teacher_id"] == author for row in output),
            "reward_configs": sorted(
                {row["reward_config_id"] for row in output if row["teacher_id"] == author}
            ),
            "reference_types": sorted(
                {row["reference_type"] for row in output if row["teacher_id"] == author}
            ),
            "speech_acts": sorted(
                {
                    row["expected_feedback_type"]
                    for row in output
                    if row["teacher_id"] == author
                }
            ),
        }
        for author in authors
    }
    insufficient_authors = [
        author
        for author, coverage in author_coverage.items()
        if coverage["examples"] == 0
        or set(coverage["reward_configs"])
        != {str(config["reward_config_id"]) for config in configurations}
        or len(coverage["reference_types"]) < 2
        or len(coverage["speech_acts"]) < 2
    ]
    if insufficient_authors:
        raise ValueError(
            "synthetic author coverage is insufficient; each author must span "
            "every reward configuration plus multiple reference types and speech acts: "
            + ", ".join(insufficient_authors)
        )

    teacher_reward_edges = {
        (row["teacher_id"], row["reward_config_id"]) for row in output
    }
    expected_teacher_reward_edges = {
        (author, str(config["reward_config_id"]))
        for author in authors
        for config in configurations
    }
    if teacher_reward_edges != expected_teacher_reward_edges:
        raise ValueError("teacher/reward augmentation is not a complete bipartite graph")
    context_reward_edges = {
        (str(row["group_id"]), row["reward_config_id"]) for row in output
    }
    # Some enumerated planner states legitimately produce no feedback intent
    # under any reward.  The paper analogue is the set of games that actually
    # yielded a base utterance, so audit augmentation over represented groups.
    context_ids = {str(row["group_id"]) for row in output}
    expected_context_reward_edges = {
        (context_id, str(config["reward_config_id"]))
        for context_id in context_ids
        for config in configurations
    }
    if context_reward_edges != expected_context_reward_edges:
        raise ValueError("context/reward augmentation is incomplete")

    cooking_preparation = [row for row in output if _is_cooking_preparation(row)]
    report = {
        "generator_version": GENERATOR_VERSION,
        "seed": seed,
        "gold_split": gold_split,
        "gold_split_status": "deprecated_not_applied_cv_manifest_owns_splits",
        "include_profile_language": include_profile_language,
        "teacher_identity": "stable_independent_synthetic_author",
        "augmentation_protocol": (
            "one_stable_author_per_base_feedback_crossed_with_every_reward_config"
        ),
        "declared_split_policy": None,
        "corpus_audit": corpus_audit,
        "author_coverage": author_coverage,
        "teacher_reward_bipartite": {
            "teachers": len(authors),
            "reward_configs": len(configurations),
            "observed_edges": len(teacher_reward_edges),
            "expected_edges": len(expected_teacher_reward_edges),
            "complete": True,
        },
        "context_reward_augmentation": {
            "contexts": len(context_ids),
            "reward_configs": len(configurations),
            "observed_edges": len(context_reward_edges),
            "expected_edges": len(expected_context_reward_edges),
            "complete": True,
        },
        "base_feedback_augmentation": {
            "base_rows": len(base_records),
            "reward_configs_per_base_row_min": min(augmentation_counts.values()),
            "reward_configs_per_base_row_max": max(augmentation_counts.values()),
            "complete": True,
            "trajectory_varies_for_base_rows": sum(
                count > 1 for count in trajectory_variation.values()
            ),
            "trajectory_varies_fraction": (
                sum(count > 1 for count in trajectory_variation.values())
                / len(trajectory_variation)
            ),
        },
        "counts": {
            "examples": len(output),
            "reward_configs": len(configurations),
            "synthetic_teachers": len({row["teacher_id"] for row in output}),
            "feedback_form_styles": len(feedback_form_styles),
            "preparation_styles": dict(
                sorted(Counter(row["preparation_style"] for row in output).items())
            ),
            "cooking_preparation_feedback": {
                "examples": len(cooking_preparation),
                "positive": sum(
                    row["attributed_sentiment_score"] > 0
                    for row in cooking_preparation
                ),
                "negative": sum(
                    row["attributed_sentiment_score"] < 0
                    for row in cooking_preparation
                ),
            },
            "empty_hand_cooking_contexts": {
                "total": sum(_is_empty_hand_cooking_context(row) for row in contexts),
            },
            "contexts": len(context_ids),
        },
    }
    return output, report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--contexts",
        type=Path,
        default=None,
        help=(
            "Optional explicit contexts JSON. By default contexts are re-enumerated "
            "from the current planner/featurizer so stale caches cannot hide new states."
        ),
    )
    parser.add_argument("--probe-states", type=Path, default=DEFAULT_PROBE_STATES_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--n-configs", type=int, default=36)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--gold-split",
        choices=SPLITS,
        default="test",
        help="Deprecated compatibility option; CV now owns every split.",
    )
    language_group = parser.add_mutually_exclusive_group()
    language_group.add_argument(
        "--include-profile-language",
        action="store_true",
        help="Explicit profile-prefix ablation; the default is semantic local feedback only.",
    )
    language_group.add_argument(
        "--local-feedback-only",
        action="store_true",
        help="Deprecated compatibility flag; semantic local feedback is already the default.",
    )
    parser.add_argument("--max-contexts", type=int)
    args = parser.parse_args()

    gold = load_gold_weights()
    contexts = load_generation_contexts(args.contexts, gold)
    if args.max_contexts is not None:
        contexts = contexts[: max(0, args.max_contexts)]
    configurations = sample_reward_configurations(n=args.n_configs)
    corpus, report = build_teacher_corpus(
        contexts=contexts,
        configurations=configurations,
        seed=args.seed,
        gold_split=args.gold_split,
        include_profile_language=args.include_profile_language,
    )
    validate_feedback_examples(corpus, probe_states=load_probe_states(args.probe_states))
    context_records = {}
    for row in corpus:
        group_id = str(row["group_id"])
        context_records.setdefault(
            group_id,
            {
                "group_id": group_id,
                "context": row["context"],
                "feasible_subgoals": row["feasible_subgoals"],
            },
        )
    compact_fields = {
        "feedback_id",
        "base_feedback_id",
        "text",
        "local_text",
        "unconditioned_local_text",
        "reward_conditioned_text",
        "reward_profile_text",
        "group_id",
        "expected_feedback_type",
        "reference_type",
        "referenced_subgoal",
        "expected_subgoal",
        "acceptable_subgoals",
        "route2_trajectory_features",
        "attributed_sentiment_score",
        "reward_config_id",
        "preparation_style",
        "teacher_id",
        "author_language_style",
        "feedback_form_style_id",
        "split",
        "paraphrase_family",
        "source",
        "generator_version",
        "target_mode",
        "reward_conditioning",
    }
    compact_examples = [
        {key: value for key, value in row.items() if key in compact_fields}
        for row in corpus
    ]
    package = {
        "schema_version": "route2-teacher-corpus-v4",
        "reward_configurations": configurations,
        "contexts": list(context_records.values()),
        "examples": compact_examples,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(package, handle, ensure_ascii=False, separators=(",", ":"))
        handle.write("\n")
    write_json(args.output.with_name(args.output.stem + ".reward_configs.json"), configurations)
    write_json(args.output.with_name(args.output.stem + ".report.json"), report)
    print(json.dumps(report["counts"], ensure_ascii=False, indent=2))
    print(f"Corpus: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
