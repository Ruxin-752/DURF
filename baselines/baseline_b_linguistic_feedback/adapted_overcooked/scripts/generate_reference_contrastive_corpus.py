"""Generate a deterministic hard five-class reference-scope corpus.

The existing LLM corpus deliberately uses explicit scope markers.  This corpus
adds natural, weaker-cue minimal contrasts without calling an LLM.  Template
families are assigned to exactly one split before expansion, so paraphrases,
groups, and normalized text cannot leak across train/dev/test.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.evaluation_splits import normalize_text  # noqa: E402
from src.feature_schema import write_json  # noqa: E402


VERSION = "reference-hard-contrastive-v1"
DEFAULT_OUTPUT = ROOT / "data" / "reference_classifier_contrastive.v1.json"
DEFAULT_REPORT = ROOT / "outputs" / "synth" / "reference_classifier_contrastive.v1.report.json"
REFERENCE_TYPES = (
    "trajectory",
    "feature",
    "action_spatial",
    "action_behavioral",
    "other",
)
SPLIT_BY_FRAME = {
    **{index: "train" for index in range(9)},
    **{index: "dev" for index in range(9, 12)},
    **{index: "test" for index in range(12, 15)},
}

CONTEXTS = (
    {
        "item": "onion",
        "source": "onion dispenser",
        "station": "left pot",
        "tail": "for the first onion order",
    },
    {
        "item": "tomato",
        "source": "tomato dispenser",
        "station": "right pot",
        "tail": "for the tomato order on screen",
    },
    {
        "item": "onion",
        "source": "lower counter",
        "station": "near pot",
        "tail": "while we work the lower lane",
    },
    {
        "item": "tomato",
        "source": "upper counter",
        "station": "far pot",
        "tail": "while the upper pot is active",
    },
    {
        "item": "onion",
        "source": "corner counter",
        "station": "center pot",
        "tail": "during the center-pot order",
    },
)

TEMPLATES = {
    "trajectory": (
        "That trip from the {source} to the {station} took a costly detour.",
        "The route you chose for the {item} made us lose several steps.",
        "Going out for the {item} and coming back that way slowed the order.",
        "Your run to collect the {item} ended in the wrong lane.",
        "The way you carried the {item} across the kitchen worked well.",
        "Fetching the {item}, circling the counter, and returning took too long.",
        "You went for the {item}, crossed my lane, then doubled back.",
        "That pickup-to-pot trip was smooth and direct.",
        "The path from the {source} back to our pot was inefficient.",
        "What you did between leaving the pot and returning with the {item} helped.",
        "The round trip for that {item} put us behind.",
        "Your approach, pickup, and return fit the order nicely.",
        "Taking the long side to fetch the {item} made the delivery late.",
        "You left the station, got the {item}, and came back without blocking me.",
        "The route around the counter before placing the {item} was unnecessary.",
    ),
    "feature": (
        "Blocking the center lane leaves me no way through.",
        "Holding an extra {item} ties up a hand we need for soup.",
        "Supplying the missing {item} keeps the recipe on track.",
        "Crowding my target makes our work overlap.",
        "Leaving the serving lane clear makes delivery easier.",
        "Duplicating my {item} task wastes both players' time.",
        "A ready dish beside the pot supports fast plating.",
        "An unnecessary ingredient in hand reduces our options.",
        "Good lane spacing helps us move without collisions.",
        "Taking the ingredient I claimed creates duplicate work.",
        "Keeping access to the {station} open helps the team.",
        "The extra {item} on the counter is clutter we do not need.",
        "Matching the current recipe is what makes that pickup useful.",
        "Standing in my shortest path causes avoidable delay.",
        "Having the dish ready before the soup finishes improves flow.",
    ),
    "action_spatial": (
        "Put that {item} into this pot.",
        "Take the dish beside you.",
        "Move one tile left and let me pass.",
        "Serve the soup at this counter.",
        "Leave that extra {item} on the counter beside the pot.",
        "Use the open lane next to me.",
        "Pick up the soup in front of you.",
        "Set the dish beside this pot.",
        "Step away from the serving entrance.",
        "Use the other side and leave this route to me.",
        "Take the {item} sitting by your feet.",
        "Place what you are holding in this empty spot.",
        "Use that dish for the ready soup.",
        "Wait on the tile behind you.",
        "Come through the gap beside the counter.",
    ),
    "action_behavioral": (
        "You tend to grab ingredients before the pot needs them.",
        "Lately you have been standing still while work is available.",
        "Most rounds you take the target I was already heading toward.",
        "You repeatedly enter the serving lane while I am delivering.",
        "I often find you duplicating my ingredient task.",
        "You have made a habit of waiting with empty hands.",
        "Whenever soup starts cooking, you prepare the next ingredient.",
        "You usually leave space when I approach the pot.",
        "On several orders you fetched a new dish instead of the nearby one.",
        "You keep returning to the same dispenser without checking the recipe.",
        "Across these orders you have avoided blocking my route.",
        "More than once you picked up an ingredient we did not need.",
        "You regularly help with the task I am not doing.",
        "Every round you rush into my lane before I can pass.",
        "The same early-pickup habit has shown up throughout the game.",
    ),
    "other": (
        "I am heading to the {source}.",
        "Is the soup ready yet?",
        "Which ingredient is next in the recipe?",
        "My hands are empty.",
        "I will take care of serving.",
        "The order display shows tomato soup.",
        "Where is the nearest dish?",
        "I am waiting by the counter.",
        "The pot has room for one more ingredient.",
        "Can you hear my microphone?",
        "I plan to use the upper lane.",
        "How much time is left?",
        "The next order needs {item}.",
        "I am carrying the dish.",
        "What does the blue timer mean?",
    ),
}

EXPLICIT_CUES = {
    "trajectory": ("overall", "whole move", "that sequence", "start to finish"),
    "action_spatial": ("right now", "right there", "at the pot", "from that spot"),
    "action_behavioral": ("keep", "always", "again", "every time", "usually", "pattern"),
}


def _token_jaccard(left: str, right: str) -> float:
    left_tokens = set(normalize_text(left).split())
    right_tokens = set(normalize_text(right).split())
    union = left_tokens | right_tokens
    return len(left_tokens & right_tokens) / len(union) if union else 1.0


def _add_context_tail(text: str, tail: str) -> str:
    if text.endswith("?"):
        return f"{text[:-1]} {tail}?"
    return f"{text.rstrip('.')} {tail}."


def build_contrastive_corpus() -> tuple[list[dict], dict]:
    template_lengths = {label: len(TEMPLATES[label]) for label in REFERENCE_TYPES}
    if set(template_lengths.values()) != {len(SPLIT_BY_FRAME)}:
        raise ValueError(f"all labels need {len(SPLIT_BY_FRAME)} frames: {template_lengths}")

    rows: list[dict] = []
    for frame_index, split in SPLIT_BY_FRAME.items():
        for context_index, context in enumerate(CONTEXTS):
            contrast_set_id = f"hard_scope_frame_{frame_index:02d}_ctx_{context_index:02d}"
            group_id = f"{VERSION}:{split}:{contrast_set_id}"
            for label in REFERENCE_TYPES:
                text = _add_context_tail(
                    TEMPLATES[label][frame_index].format(**context),
                    context["tail"],
                )
                feedback_id = f"{contrast_set_id}_{label}"
                rows.append(
                    {
                        "feedback_id": feedback_id,
                        "group_id": group_id,
                        "paraphrase_family": f"{VERSION}:frame_{frame_index:02d}:{label}",
                        "contrast_set_id": contrast_set_id,
                        "reference_type": label,
                        "phrase_annotations": [
                            {
                                "start": 0,
                                "end": len(text),
                                "text": text,
                                "reference_type": label,
                            }
                        ],
                        "text": text,
                        "role": "hard_reference_contrast",
                        "source": "deterministic_contrastive",
                        "label_source": "operational_reference_rubric",
                        "generator_version": VERSION,
                        "split": split,
                        "context": dict(context),
                    }
                )

    normalized_locations: dict[str, list[dict]] = defaultdict(list)
    group_splits: dict[str, set[str]] = defaultdict(set)
    contrast_labels: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        normalized_locations[normalize_text(row["text"])].append(row)
        group_splits[row["group_id"]].add(row["split"])
        contrast_labels[row["contrast_set_id"]].add(row["reference_type"])
    duplicate_texts = {
        text: values for text, values in normalized_locations.items() if len(values) > 1
    }
    if duplicate_texts:
        raise ValueError(f"normalized duplicate hard phrases: {list(duplicate_texts)[:3]}")
    leaking_groups = [group for group, splits in group_splits.items() if len(splits) > 1]
    if leaking_groups:
        raise ValueError(f"hard-corpus group leakage: {leaking_groups[:3]}")
    incomplete_sets = [
        group for group, labels in contrast_labels.items() if labels != set(REFERENCE_TYPES)
    ]
    if incomplete_sets:
        raise ValueError(f"incomplete contrast sets: {incomplete_sets[:3]}")

    cross_label_near_duplicates = []
    for index, left in enumerate(rows):
        for right in rows[index + 1 :]:
            if left["reference_type"] == right["reference_type"]:
                continue
            similarity = _token_jaccard(left["text"], right["text"])
            if similarity >= 0.90:
                cross_label_near_duplicates.append(
                    {
                        "left": left["feedback_id"],
                        "right": right["feedback_id"],
                        "similarity": similarity,
                    }
                )
    if cross_label_near_duplicates:
        raise ValueError(
            "cross-label near duplicate hard phrases: "
            f"{cross_label_near_duplicates[:3]}"
        )

    cue_free_counts = {
        label: sum(
            row["reference_type"] == label
            and not any(cue in row["text"].lower() for cue in cues)
            for row in rows
        )
        for label, cues in EXPLICIT_CUES.items()
    }
    report = {
        "version": VERSION,
        "selection_uses_model_predictions": False,
        "generation_uses_llm": False,
        "rows": len(rows),
        "contrast_sets": len(contrast_labels),
        "contexts": len(CONTEXTS),
        "frames": len(SPLIT_BY_FRAME),
        "split_counts": dict(sorted(Counter(row["split"] for row in rows).items())),
        "label_counts": dict(
            sorted(Counter(row["reference_type"] for row in rows).items())
        ),
        "split_label_counts": {
            split: {
                label: sum(
                    row["split"] == split and row["reference_type"] == label
                    for row in rows
                )
                for label in REFERENCE_TYPES
            }
            for split in ("train", "dev", "test")
        },
        "explicit_cue_free_counts": cue_free_counts,
        "normalized_duplicate_count": 0,
        "cross_label_near_duplicate_count": 0,
        "group_overlap_count": 0,
    }
    return rows, report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args()

    rows, report = build_contrastive_corpus()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.output, rows)
    report["output"] = str(args.output)
    report["output_sha256"] = hashlib.sha256(args.output.read_bytes()).hexdigest()
    write_json(args.report, report)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
