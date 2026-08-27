"""Build a diverse feedback-form candidate pool for independent human review.

These examples are assistant-curated prompts, not human observations and not
training data.  The generated JSONL deliberately carries a third
``review_status`` field, so it cannot be consumed by the strict two-field
human-annotation importer by accident.  A person must accept, edit, or reject
each candidate before any accepted language is copied into a separately
managed annotation/evaluation set.
"""

from __future__ import annotations

import argparse
from collections import Counter
from difflib import SequenceMatcher
import hashlib
import json
from pathlib import Path
import sys
from typing import Iterable, Sequence


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.prepare_human_feedback_form_annotations import (  # noqa: E402
    CANONICAL_LABELS,
    template_family_signature,
)
from src.evaluation_splits import normalize_text  # noqa: E402
from src.feature_schema import write_json  # noqa: E402


DEFAULT_OUTPUT = (
    ROOT / "data" / "assistant_curated_feedback_form_candidates.pending_review.jsonl"
)
DEFAULT_BLIND_OUTPUT = (
    ROOT / "data" / "feedback_form_candidates.blind_labeling.template.jsonl"
)
DEFAULT_REPORT = (
    ROOT / "outputs" / "assistant_curated_feedback_form_candidates.report.json"
)
REVIEW_STATUS = "assistant_curated_pending_human_review"
BLIND_SHUFFLE_VERSION = "feedback-form-blind-review-order-v1"
MINIMUM_FAMILIES_PER_LABEL = 15
NEAR_DUPLICATE_THRESHOLD = 0.78
SEQUENCE_SIMILARITY_THRESHOLD = 0.90

# One entry represents one intentionally distinct surface construction.  Item
# substitution is not used to inflate the family count.
CURATED_CANDIDATES: tuple[tuple[str, str, str], ...] = (
    # Evaluative: an assessment of an action/trajectory, without a command.
    ("eval_timing", "Nice timing on that handoff.", "Evaluative"),
    ("eval_handled", "You handled that exchange well.", "Evaluative"),
    ("eval_liked", "I liked your last decision.", "Evaluative"),
    ("eval_right_call", "That was exactly the right call.", "Evaluative"),
    ("eval_hurt_run", "The previous step hurt our run.", "Evaluative"),
    ("eval_happy_outcome", "I'm happy with how that turned out.", "Evaluative"),
    ("eval_contribution", "Your contribution there was excellent.", "Evaluative"),
    ("eval_dissatisfied", "I wasn't satisfied with that choice.", "Evaluative"),
    ("eval_worked", "What you just did worked perfectly.", "Evaluative"),
    ("eval_better", "That could have gone much better.", "Evaluative"),
    ("eval_solid", "Solid play on that order.", "Evaluative"),
    ("eval_approve", "I approve of that decision.", "Evaluative"),
    ("eval_mess", "The last few seconds were a mess.", "Evaluative"),
    ("eval_spot_on", "Your execution there was spot on.", "Evaluative"),
    ("eval_acceptable", "That was acceptable, though not ideal.", "Evaluative"),
    ("eval_nailed", "You nailed that part.", "Evaluative"),
    ("eval_team_help", "That action helped our team a lot.", "Evaluative"),
    ("eval_impressed", "I'm impressed by that choice.", "Evaluative"),
    ("eval_bad_moment", "We handled that moment badly.", "Evaluative"),
    ("eval_clean_teamwork", "That was a clean piece of teamwork.", "Evaluative"),
    ("eval_rate_poorly", "I would rate that attempt poorly.", "Evaluative"),
    ("eval_worthwhile", "Your last contribution was worthwhile.", "Evaluative"),
    ("eval_smoother", "That sequence was smoother than before.", "Evaluative"),
    ("eval_regret", "I regret that last choice.", "Evaluative"),
    ("eval_disappointing", "The outcome of that move was disappointing.", "Evaluative"),
    ("eval_recovered", "You recovered from that situation nicely.", "Evaluative"),
    ("eval_did_not_work", "That attempt did not work for me.", "Evaluative"),
    ("eval_best_result", "I could not have asked for a better result.", "Evaluative"),
    ("eval_clumsy", "The way you completed that step was clumsy.", "Evaluative"),
    ("eval_coordination", "Our coordination there felt excellent.", "Evaluative"),
    # Imperative: an explicit request or instruction for a future action.
    ("imp_clear_tile", "Could you clear the tile beside the pot?", "Imperative"),
    ("imp_leave_onion", "Leave the onion on the open counter.", "Imperative"),
    ("imp_hold_serving", "Hold off on serving until the recipe is complete.", "Imperative"),
    ("imp_dish_duty", "Take over dish duty for this order.", "Imperative"),
    ("imp_give_room", "Give me room to reach the cooker.", "Imperative"),
    ("imp_use_counter_tomato", "Use the tomato that is already on the counter.", "Imperative"),
    ("imp_head_to_service", "Head toward the serving station with the finished soup.", "Imperative"),
    ("imp_set_plate", "Set an empty plate beside the cooker.", "Imperative"),
    ("imp_stop_collecting", "Stop collecting ingredients while the pot is full.", "Imperative"),
    ("imp_stay_left", "Stay on the left side of the kitchen.", "Imperative"),
    ("imp_let_me_handle", "Let me handle the next onion.", "Imperative"),
    ("imp_keep_corridor", "Keep the central corridor open.", "Imperative"),
    ("imp_start_next", "Start preparing the following order.", "Imperative"),
    ("imp_plate_soup", "Transfer the cooked soup onto a plate.", "Imperative"),
    ("imp_wait_dispenser", "Wait by the dispenser until I pass.", "Imperative"),
    ("imp_pick_closest", "Pick up the dish closest to the pot.", "Imperative"),
    ("imp_dont_take", "Don't take the ingredient I'm walking toward.", "Imperative"),
    ("imp_clockwise", "Move clockwise around the island.", "Imperative"),
    ("imp_focus_plating", "Focus on plating while I collect produce.", "Imperative"),
    ("imp_drop_holding", "Drop what you're holding on the empty counter.", "Imperative"),
    ("imp_save_tomato", "Save that spare tomato for the next ticket.", "Imperative"),
    ("imp_switch_delivery", "Switch to delivery while I finish cooking.", "Imperative"),
    ("imp_make_space", "Make space at the serving window.", "Imperative"),
    ("imp_pass_plate", "Pass me the plate through the middle counter.", "Imperative"),
    ("imp_avoid_crossing", "Avoid crossing in front of me near the pot.", "Imperative"),
    ("imp_circle_bottom", "Circle around the bottom instead of meeting me in the center.", "Imperative"),
    ("imp_return_station", "Return to the ingredient station after serving.", "Imperative"),
    ("imp_take_ready", "Take the ready soup before starting another task.", "Imperative"),
    ("imp_keep_plate", "Keep carrying that plate until the pot finishes.", "Imperative"),
    ("imp_cover_right_pot", "Cover the right-hand pot for this order.", "Imperative"),
    # Descriptive: a task-relevant state, feature, relation, or consequence;
    # no direct command and no overall judgement of the completed trajectory.
    ("desc_needs_one", "The pot still needs one more ingredient.", "Descriptive"),
    ("desc_only_route", "The center lane is our only route to the serving window.", "Descriptive"),
    ("desc_plate_waiting", "A clean plate is already waiting beside the cooker.", "Descriptive"),
    ("desc_right_ready", "The soup on the right is ready to plate.", "Descriptive"),
    ("desc_wrong_order", "The onion you picked up does not match this order.", "Descriptive"),
    ("desc_same_dispenser", "Both of us are walking toward the same dispenser.", "Descriptive"),
    ("desc_counter_space", "An unused tomato is taking up counter space.", "Descriptive"),
    ("desc_ticket_order", "The current order expires before the next one.", "Descriptive"),
    ("desc_blocks_left", "Standing beside the pot blocks access from the left.", "Descriptive"),
    ("desc_handoff_free", "The empty counter near me is free for a handoff.", "Descriptive"),
    ("desc_enough", "There are enough ingredients in the pot already.", "Descriptive"),
    ("desc_shortest_route", "The shortest route to delivery goes around the bottom.", "Descriptive"),
    ("desc_carrying_final", "I am carrying the final onion for this recipe.", "Descriptive"),
    ("desc_plate_closer", "The plate by the window is closer than the dispenser.", "Descriptive"),
    ("desc_duplicate_task", "Fetching a second dish would duplicate my task.", "Descriptive"),
    ("desc_unreachable", "The cooker cannot be reached while that tile is occupied.", "Descriptive"),
    ("desc_next_ticket", "Our next ticket asks for three tomatoes.", "Descriptive"),
    ("desc_not_served", "The finished soup has not been served yet.", "Descriptive"),
    ("desc_following_order", "The loose onion belongs to the following order.", "Descriptive"),
    ("desc_both_waiting", "We are both waiting while the pot is empty.", "Descriptive"),
    ("desc_one_counter", "Only one free counter remains between us.", "Descriptive"),
    ("desc_upper_blocked", "The upper passage is blocked by your position.", "Descriptive"),
    ("desc_routes_intersect", "Our two routes intersect beside the cooker.", "Descriptive"),
    ("desc_dish_at_service", "A completed dish is sitting at the service counter.", "Descriptive"),
    ("desc_second_empty", "The second pot has no ingredients in it.", "Descriptive"),
    ("desc_plate_reserved", "The plate in my hands is reserved for the ready soup.", "Descriptive"),
    ("desc_shared_tomato", "A tomato is already available on our shared counter.", "Descriptive"),
    ("desc_lane_clear", "The serving lane is clear from the lower side.", "Descriptive"),
    ("desc_left_finishes", "The left pot will finish before the right one.", "Descriptive"),
    ("desc_same_counter", "My route ends at the same counter you are approaching.", "Descriptive"),
)

DEFAULT_EXISTING_SOURCES: tuple[tuple[str, Path, str], ...] = (
    (
        "human_raw",
        ROOT / "data" / "human_feedback_form_annotations.jsonl",
        "jsonl",
    ),
    ("human_train", ROOT / "data" / "human_feedback_form_train.json", "json"),
    ("human_dev", ROOT / "data" / "human_feedback_form_dev.json", "json"),
    ("human_test", ROOT / "data" / "human_feedback_form_test.json", "json"),
    (
        "human_train_augmentation",
        ROOT / "data" / "human_feedback_form_train_augmentation.json",
        "json",
    ),
    (
        "feedback_form_hard_train",
        ROOT / "data" / "feedback_form_hard_train.v1.json",
        "json",
    ),
    (
        "reference_classifier_v8",
        ROOT / "data" / "reference_classifier_feedback.v8.json",
        "json",
    ),
)


def _text_from_row(row: dict) -> str:
    for key in ("language", "text", "normalized_text", "normalized"):
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            return value
    return ""


def _load_rows(path: Path, file_format: str) -> list[dict]:
    if not path.exists():
        return []
    if file_format == "json":
        value = json.loads(path.read_text(encoding="utf-8-sig"))
        if not isinstance(value, list):
            raise ValueError(f"{path}: expected a JSON array")
        return [row for row in value if isinstance(row, dict)]
    if file_format == "jsonl":
        rows: list[dict] = []
        for line_number, raw in enumerate(
            path.read_text(encoding="utf-8-sig").splitlines(), start=1
        ):
            if not raw.strip():
                continue
            value = json.loads(raw)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: expected an object")
            rows.append(value)
        return rows
    raise ValueError(f"unsupported file format: {file_format!r}")


def load_existing_texts(
    sources: Sequence[tuple[str, Path, str]] = DEFAULT_EXISTING_SOURCES,
) -> tuple[dict[str, list[str]], dict[str, int]]:
    texts_by_source: dict[str, list[str]] = {}
    source_row_counts: dict[str, int] = {}
    for source_name, path, file_format in sources:
        rows = _load_rows(path, file_format)
        source_row_counts[source_name] = len(rows)
        texts_by_source[source_name] = [
            text for row in rows if (text := _text_from_row(row))
        ]
    return texts_by_source, source_row_counts


def _tokens(normalized: str) -> frozenset[str]:
    return frozenset(normalized.split())


def _near_duplicate(left: str, right: str) -> bool:
    left_tokens = _tokens(left)
    right_tokens = _tokens(right)
    if not left_tokens or not right_tokens:
        return False
    intersection = len(left_tokens & right_tokens)
    union = len(left_tokens | right_tokens)
    jaccard = intersection / union
    shorter = min(len(left_tokens), len(right_tokens))
    longer = max(len(left_tokens), len(right_tokens))
    containment = intersection / shorter
    length_ratio = shorter / longer
    if jaccard >= NEAR_DUPLICATE_THRESHOLD:
        return True
    if shorter >= 4 and containment >= 0.90 and length_ratio >= 0.60:
        return True
    if length_ratio >= 0.60:
        return (
            SequenceMatcher(a=left, b=right, autojunk=False).ratio()
            >= SEQUENCE_SIMILARITY_THRESHOLD
        )
    return False


def curate_candidate_pool(
    texts_by_source: dict[str, Iterable[str]],
    candidates: Sequence[tuple[str, str, str]] = CURATED_CANDIDATES,
    *,
    minimum_families_per_label: int = MINIMUM_FAMILIES_PER_LABEL,
) -> tuple[list[dict], dict]:
    """Filter candidates against existing corpora without assigning labels by model."""

    existing_records: list[tuple[str, str]] = []
    for texts in texts_by_source.values():
        for text in texts:
            normalized = normalize_text(text)
            if normalized:
                existing_records.append(
                    (normalized, template_family_signature(normalized))
                )
    # De-duplicate repeated material across raw/prepared/augmented sources.
    existing_records = sorted(set(existing_records))
    existing_exact = {normalized for normalized, _ in existing_records}
    existing_templates = {signature for _, signature in existing_records}

    selected: list[dict] = []
    selected_internal: list[tuple[str, str, str]] = []
    selected_records: list[tuple[str, str]] = []
    rejections: Counter[str] = Counter()
    rejected_family_ids: list[str] = []
    seen_family_ids: set[str] = set()

    for family_id, language, label in candidates:
        if not family_id or family_id in seen_family_ids:
            raise ValueError(f"duplicate or empty curated family id: {family_id!r}")
        seen_family_ids.add(family_id)
        if label not in CANONICAL_LABELS:
            raise ValueError(f"invalid candidate label: {label!r}")
        if not isinstance(language, str) or not language.strip():
            raise ValueError(f"{family_id}: language must be non-empty")
        normalized = normalize_text(language)
        signature = template_family_signature(normalized)
        reason: str | None = None
        if normalized in existing_exact:
            reason = "existing_exact"
        elif signature in existing_templates:
            reason = "existing_template_family"
        elif any(_near_duplicate(normalized, other) for other, _ in existing_records):
            reason = "existing_near_duplicate"
        elif any(normalized == other for other, _ in selected_records):
            reason = "candidate_exact"
        elif any(signature == other for _, other in selected_records):
            reason = "candidate_template_family"
        elif any(_near_duplicate(normalized, other) for other, _ in selected_records):
            reason = "candidate_near_duplicate"
        if reason is not None:
            rejections[reason] += 1
            rejected_family_ids.append(family_id)
            continue
        selected.append(
            {
                "language": language.strip(),
                "classification_label": label,
                "review_status": REVIEW_STATUS,
            }
        )
        selected_internal.append((family_id, normalized, label))
        selected_records.append((normalized, signature))

    counts = Counter(label for _, _, label in selected_internal)
    insufficient = {
        label: counts[label]
        for label in CANONICAL_LABELS
        if counts[label] < minimum_families_per_label
    }
    if insufficient:
        raise ValueError(
            "candidate pool has fewer than "
            f"{minimum_families_per_label} novel surface families: {insufficient}"
        )

    report = {
        "schema_version": 1,
        "provenance": REVIEW_STATUS,
        "human_authorship_claimed": False,
        "human_review_completed": False,
        "training_eligible": False,
        "evaluation_eligible": False,
        "model_or_classifier_used_to_assign_labels": False,
        "candidate_schema": {
            "exact_fields": [
                "language",
                "classification_label",
                "review_status",
            ],
            "canonical_labels": list(CANONICAL_LABELS),
            "required_review_status": REVIEW_STATUS,
        },
        "taxonomy": {
            "Evaluative": "assessment of a completed action or trajectory",
            "Imperative": "request or instruction for a future action",
            "Descriptive": "task-relevant state, feature, relation, or consequence",
        },
        "candidate_count": len(selected),
        "distinct_surface_family_count": len(selected_internal),
        "counts_by_label": {
            label: counts[label] for label in CANONICAL_LABELS
        },
        "minimum_distinct_families_per_label": minimum_families_per_label,
        "rejections": dict(sorted(rejections.items())),
        "rejected_family_ids": sorted(rejected_family_ids),
        "novelty_guard": {
            "existing_unique_texts_compared": len(existing_records),
            "exact_collision_in_emitted_pool": 0,
            "template_family_collision_in_emitted_pool": 0,
            "near_duplicate_in_emitted_pool": 0,
            "token_jaccard_threshold": NEAR_DUPLICATE_THRESHOLD,
            "sequence_similarity_threshold": SEQUENCE_SIMILARITY_THRESHOLD,
            "sources_compared": sorted(texts_by_source),
        },
        "accuracy_boundary": {
            "these_candidates_change_reported_human_accuracy": False,
            "reason": (
                "assistant-curated labels are not an independent human gold test; "
                "accepted examples must be human-reviewed and a held-out subset "
                "must be frozen before evaluation"
            ),
        },
    }
    return selected, report


def _write_jsonl(path: Path, rows: Iterable[dict]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = "".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows
    )
    path.write_text(payload, encoding="utf-8")
    # Hash the bytes that actually reached disk.  On Windows, text-mode newline
    # translation can otherwise make a pre-write payload hash incorrect.
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_blind_labeling_rows(rows: Iterable[dict]) -> list[dict]:
    """Hide suggested labels and deterministically scramble presentation order."""

    blind_rows = [
        {"language": row["language"], "classification_label": ""}
        for row in rows
    ]
    blind_rows.sort(
        key=lambda row: hashlib.sha256(
            (
                BLIND_SHUFFLE_VERSION
                + "\0"
                + normalize_text(row["language"])
            ).encode("utf-8")
        ).hexdigest()
    )
    return blind_rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--blind-output", type=Path, default=DEFAULT_BLIND_OUTPUT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args()

    texts_by_source, source_row_counts = load_existing_texts()
    rows, report = curate_candidate_pool(texts_by_source)
    blind_rows = build_blind_labeling_rows(rows)
    report["source_row_counts"] = source_row_counts
    report["output"] = str(args.output)
    report["output_sha256"] = _write_jsonl(args.output, rows)
    report["blind_labeling_template"] = {
        "output": str(args.blind_output),
        "output_sha256": _write_jsonl(args.blind_output, blind_rows),
        "row_count": len(blind_rows),
        "exact_fields": ["language", "classification_label"],
        "classification_label_value": "",
        "suggested_labels_visible": False,
        "order": "deterministic_sha256_shuffle",
        "order_version": BLIND_SHUFFLE_VERSION,
        "automatically_importable_before_labeling": False,
        "automatically_imported_or_trained": False,
        "intended_use_after_completion": (
            "freeze the entire completed file as a holdout benchmark"
        ),
        "must_not_be_merged_into": (
            "human feedback training annotations or model-selection data"
        ),
        "review_protocol": (
            "Annotate this file without opening the suggested-label file; "
            "freeze the completed file before running accuracy"
        ),
        "claim_after_blind_labeling": (
            "human-adjudicated labels on assistant-authored utterances; not "
            "natural human-player language"
        ),
    }
    report["suggested_label_boundary"] = {
        "allowed_after_explicit_opt_in": "assistant-curated training or regression",
        "never_valid_as": "human gold",
        "automatically_imported_or_trained": False,
    }
    write_json(args.report, report)
    print(
        "Prepared assistant-curated candidates pending human review: "
        + ", ".join(
            f"{label}={report['counts_by_label'][label]}"
            for label in CANONICAL_LABELS
        )
    )
    print("Training/evaluation eligibility: false until independent human review.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
