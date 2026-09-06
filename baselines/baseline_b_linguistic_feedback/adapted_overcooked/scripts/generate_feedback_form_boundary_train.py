"""Build a balanced, train-only feedback-form boundary corpus.

The corpus adds short conversational utterances and semantic minimal contrasts
for the three paper feedback forms. Development texts are used only as leakage
gates; test files are never opened. The frozen paper test is checked through
precomputed membership hashes, so the frozen file itself remains unopened.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from difflib import SequenceMatcher
import hashlib
import json
import sys
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.evaluation_splits import normalize_text  # noqa: E402
from src.feature_schema import write_json  # noqa: E402


VERSION = "feedback-form-boundary-train-v1"
FEEDBACK_TYPES = ("evaluative", "imperative", "descriptive")
CANONICAL_LABELS = {
    "evaluative": "Evaluative",
    "imperative": "Imperative",
    "descriptive": "Descriptive",
}
SOURCE = "feedback_form_hard_train"
LABEL_SOURCE = "direct_speech_act_contrastive_templates_train_only"
DEFAULT_BASE = ROOT / "data" / "feedback_form_hard_train.v1.json"
DEFAULT_OUTPUT = ROOT / "data" / "feedback_form_boundary_train.v1.json"
DEFAULT_REPORT = (
    ROOT / "outputs" / "synth" / "feedback_form_boundary_train.v1.report.json"
)
DEFAULT_PAPER_MANIFEST = (
    ROOT
    / "outputs"
    / "feedback_form_classifier"
    / "paper_human_benchmark.v1.manifest.json"
)
DEFAULT_HOLDOUT_SPECS = (
    (ROOT / "data" / "human_feedback_form_dev.json", False),
    (ROOT / "data" / "paper_feedback_form_human_dev.v1.json", False),
)
SCHEMA_FIELDS = frozenset(
    {
        "text",
        "normalized",
        "expected_feedback_type",
        "classification_label",
        "split",
        "source",
        "label_source",
        "group_id",
        "template_family",
    }
)


# Every scenario supplies grammatically audited verb forms.  Each is crossed
# with all twelve semantic contrast families below.
SCENARIOS = (
    {
        "base": "move the clean plate to the dish return",
        "past": "moved the clean plate to the dish return",
        "gerund": "moving the clean plate to the dish return",
        "state": "the clean plate is at the dish return",
    },
    {
        "base": "chop the tomato on the upper board",
        "past": "chopped the tomato on the upper board",
        "gerund": "chopping the tomato on the upper board",
        "state": "the tomato is on the upper chopping board",
    },
    {
        "base": "bring the onion to the lower counter",
        "past": "brought the onion to the lower counter",
        "gerund": "bringing the onion to the lower counter",
        "state": "the onion is on the lower counter",
    },
    {
        "base": "put the cooked soup by the serving window",
        "past": "put the cooked soup by the serving window",
        "gerund": "putting the cooked soup by the serving window",
        "state": "the cooked soup is beside the serving window",
    },
    {
        "base": "pick up the empty dish from the center table",
        "past": "picked up the empty dish from the center table",
        "gerund": "picking up the empty dish from the center table",
        "state": "the empty dish is on the center table",
    },
    {
        "base": "clear the spare tomato from the main aisle",
        "past": "cleared the spare tomato from the main aisle",
        "gerund": "clearing the spare tomato from the main aisle",
        "state": "the spare tomato is in the main aisle",
    },
    {
        "base": "leave the raw onion on the prep table",
        "past": "left the raw onion on the prep table",
        "gerund": "leaving the raw onion on the prep table",
        "state": "the raw onion is on the prep table",
    },
    {
        "base": "take the clean plate to the pot",
        "past": "took the clean plate to the pot",
        "gerund": "taking the clean plate to the pot",
        "state": "the clean plate is next to the pot",
    },
    {
        "base": "serve the finished soup at the window",
        "past": "served the finished soup at the window",
        "gerund": "serving the finished soup at the window",
        "state": "the finished soup is ready at the window",
    },
    {
        "base": "step away from the crowded center",
        "past": "stepped away from the crowded center",
        "gerund": "stepping away from the crowded center",
        "state": "the center lane is crowded",
    },
    {
        "base": "use the lower path to reach the onion crate",
        "past": "used the lower path to reach the onion crate",
        "gerund": "using the lower path to reach the onion crate",
        "state": "the lower path leads to the onion crate",
    },
    {
        "base": "keep the serving lane open",
        "past": "kept the serving lane open",
        "gerund": "keeping the serving lane open",
        "state": "the serving lane is open",
    },
)


# Rows at the same index are semantic minimal contrasts: a general state or
# property (Descriptive), a requested next action (Imperative), and a judgment
# about completed behavior (Evaluative).  Modal/action vocabulary deliberately
# appears in multiple classes so one cue word cannot solve the corpus.
CONTRAST_TEMPLATES = {
    "descriptive": (
        "{state}.",
        "In this layout, {gerund} usually saves time.",
        "{gerund_cap} can keep the workflow smooth.",
        "The team has enough space for {gerund}.",
        "The safest option is {gerund} before the timer ends.",
        "It is possible to {base} without blocking anyone.",
        "{gerund_cap} is unnecessary when the lane is already clear.",
        "A clear counter makes {gerund} easier.",
        "The current recipe allows us to {base}.",
        "{gerund_cap} reduces congestion in this situation.",
        "The route stays open while we are {gerund}.",
        "There is enough time to {base} before the order expires.",
    ),
    "imperative": (
        "{base_cap} now.",
        "Please {base}.",
        "Could you {base}?",
        "Can you {base}?",
        "You should {base} next.",
        "You need to {base}.",
        "Do not {base} while I cross.",
        "Try to {base}.",
        "I need you to {base}.",
        "Would you {base}?",
        "Remember to {base}.",
        "Why don't you {base}?",
    ),
    "evaluative": (
        "It was a good choice when you {past}.",
        "You did well when you {past}.",
        "I liked how you {past}.",
        "That last decision to {base} was helpful.",
        "You should have {past} earlier.",
        "You handled it efficiently when you {past}.",
        "There was no need to {base} on the last order.",
        "That attempt to {base} made the handoff worse.",
        "Thanks for the way you {past}.",
        "The result was better after you {past}.",
        "Your last attempt at {gerund} was excellent.",
        "Your last attempt at {gerund} was a mistake.",
    ),
}


SHORT_UTTERANCES = {
    "evaluative": (
        "That action was genuinely good.",
        "Great job on that move.",
        "Nice work just now.",
        "That was genuinely helpful.",
        "Your last move was excellent.",
        "You handled that perfectly.",
        "That route choice was efficient.",
        "The last handoff went well.",
        "Thanks for clearing the lane.",
        "I liked that decision.",
        "That move was a poor choice.",
        "That did not help the team.",
        "Your last trip wasted time.",
        "You handled that badly.",
        "That detour was unnecessary.",
        "I did not like that move.",
        "You should have served sooner.",
        "You should not have blocked the aisle.",
        "That could have been better.",
        "The previous action slowed us down.",
        "Well done on the last order.",
        "That was the right call.",
        "The last pickup was a mistake.",
        "Your timing on that delivery was great.",
    ),
    "imperative": (
        "Grab a tomato.",
        "Please chop the onion.",
        "Put the plate on the counter.",
        "Serve the soup now.",
        "Can you clear the center lane?",
        "Could you bring me a plate?",
        "Would you watch the pot?",
        "You should get another onion.",
        "You need to move left.",
        "Do not block the serving window.",
        "Let's finish this order.",
        "Try to keep the aisle clear.",
        "Remember to plate the soup.",
        "I need you to wash a dish.",
        "How about chopping the tomato?",
        "Why don't you take the lower route?",
        "Move away from the pot.",
        "Leave the onion on the table.",
        "Pick up the cooked soup.",
        "Help me with the next order.",
        "Get a clean plate next.",
        "Please use the upper counter.",
        "Keep the middle path open.",
        "Take the finished dish to the window.",
    ),
    "descriptive": (
        "The pot is ready.",
        "The soup is still cooking.",
        "The center lane is blocked.",
        "We have one clean plate left.",
        "An onion is waiting on the lower counter.",
        "The tomato crate is beside the sink.",
        "The serving window is open.",
        "The next order needs two tomatoes.",
        "The timer has ten seconds left.",
        "You are holding an onion.",
        "I am carrying a plate.",
        "The top counter is empty.",
        "The soup needs one more onion.",
        "There is a plate by the pot.",
        "The quickest route goes through the center.",
        "Keeping the aisle clear saves time.",
        "Chopped vegetables can go on the counter.",
        "A plate is required before serving.",
        "The dish rack has a clean plate.",
        "The current pot contains two tomatoes.",
        "The lower route is shorter.",
        "The board beside the onion crate is free.",
        "A finished soup can be served immediately.",
        "Moving through the center takes less time.",
    ),
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _text_hash(text: str) -> str:
    return hashlib.sha256(normalize_text(text).encode("utf-8")).hexdigest()


def _read_json_list(path: Path) -> list[dict]:
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(value, list) or not all(isinstance(row, dict) for row in value):
        raise ValueError(f"expected a JSON list of objects: {path}")
    return [dict(row) for row in value]


def _make_row(text: str, label: str, group_id: str, family: str) -> dict:
    normalized = normalize_text(text)
    if not normalized:
        raise ValueError("generated feedback text is empty")
    return {
        "text": text,
        "normalized": normalized,
        "expected_feedback_type": label,
        "classification_label": CANONICAL_LABELS[label],
        "split": "train",
        "source": SOURCE,
        "label_source": LABEL_SOURCE,
        "group_id": group_id,
        "template_family": family,
    }


def generate_boundary_rows(base_path: Path = DEFAULT_BASE) -> list[dict]:
    """Return the v1 hard set plus new contrastive and short-form rows."""

    base_rows = _read_json_list(base_path)
    rows = [dict(row) for row in base_rows]
    for family_index in range(len(CONTRAST_TEMPLATES["evaluative"])):
        for scenario_index, scenario in enumerate(SCENARIOS):
            values = {
                **scenario,
                "base_cap": scenario["base"][:1].upper() + scenario["base"][1:],
                "gerund_cap": scenario["gerund"][:1].upper()
                + scenario["gerund"][1:],
            }
            group_id = (
                f"{VERSION}:contrast_{family_index:02d}_{scenario_index:02d}"
            )
            for label in FEEDBACK_TYPES:
                text = CONTRAST_TEMPLATES[label][family_index].format(**values)
                rows.append(
                    _make_row(
                        text,
                        label,
                        group_id,
                        f"{VERSION}:{label}:contrast_{family_index:02d}",
                    )
                )
    short_lengths = {len(SHORT_UTTERANCES[label]) for label in FEEDBACK_TYPES}
    if len(short_lengths) != 1:
        raise ValueError("short-form utterance lists must be class balanced")
    for index in range(next(iter(short_lengths))):
        group_id = f"{VERSION}:short_{index:02d}"
        for label in FEEDBACK_TYPES:
            rows.append(
                _make_row(
                    SHORT_UTTERANCES[label][index],
                    label,
                    group_id,
                    f"{VERSION}:{label}:short_{index:02d}",
                )
            )
    return rows


def _iter_holdout_rows(path: Path, dev_split_only: bool) -> Iterable[dict]:
    if not path.exists() or path.stat().st_size <= 4:
        return ()
    rows = _read_json_list(path)
    if dev_split_only:
        return (row for row in rows if row.get("split") == "dev")
    return iter(rows)


def _load_holdout_gate(
    specs: Iterable[tuple[Path, bool]], paper_manifest_path: Path
) -> tuple[set[str], set[str], list[str], dict]:
    text_hashes: set[str] = set()
    group_tokens: set[str] = set()
    normalized_texts: list[str] = []
    inputs = []
    for path, dev_split_only in specs:
        if not path.exists():
            continue
        rows_seen = 0
        for row in _iter_holdout_rows(path, dev_split_only):
            text = row.get("text")
            if isinstance(text, str) and normalize_text(text):
                normalized = normalize_text(text)
                normalized_texts.append(normalized)
                text_hashes.add(hashlib.sha256(normalized.encode("utf-8")).hexdigest())
                rows_seen += 1
            for field in ("group_id", "paraphrase_family", "paper_task_uuid"):
                token = row.get(field)
                if isinstance(token, str) and token:
                    group_tokens.add(token)
        inputs.append(
            {
                "path": str(path),
                "sha256": _sha256(path),
                "split_filter": "dev_only" if dev_split_only else "all_rows",
                "text_rows_used_for_gate": rows_seen,
            }
        )

    manifest = json.loads(paper_manifest_path.read_text(encoding="utf-8-sig"))
    membership = manifest.get("frozen_test_membership") or {}
    text_hashes.update(membership.get("normalized_text_sha256") or [])
    group_hashes = set(membership.get("paper_task_uuid_sha256") or [])
    return text_hashes, group_tokens, normalized_texts, {
        "inputs": inputs,
        "paper_manifest": str(paper_manifest_path),
        "paper_manifest_sha256": _sha256(paper_manifest_path),
        "frozen_text_hash_count": len(
            membership.get("normalized_text_sha256") or []
        ),
        "frozen_group_hash_count": len(group_hashes),
        "frozen_file_opened": False,
    }


def _is_near_holdout(normalized: str, holdout_texts: list[str], threshold: float) -> bool:
    tokens = set(normalized.split())
    for held_out in holdout_texts:
        held_tokens = set(held_out.split())
        if not (tokens & held_tokens):
            continue
        length_ratio = min(len(normalized), len(held_out)) / max(
            len(normalized), len(held_out)
        )
        if length_ratio < threshold:
            continue
        union = tokens | held_tokens
        if union and len(tokens & held_tokens) / len(union) < 0.7:
            continue
        if SequenceMatcher(None, normalized, held_out, autojunk=False).ratio() >= threshold:
            return True
    return False


def build_boundary_corpus(
    base_path: Path = DEFAULT_BASE,
    holdout_specs: Iterable[tuple[Path, bool]] = DEFAULT_HOLDOUT_SPECS,
    paper_manifest_path: Path = DEFAULT_PAPER_MANIFEST,
    *,
    near_duplicate_threshold: float = 0.94,
) -> tuple[list[dict], dict]:
    """Generate, validate, and leakage-filter balanced contrast groups."""

    rows = generate_boundary_rows(base_path)
    by_normalized: dict[str, list[dict]] = defaultdict(list)
    by_group: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        if set(row) != SCHEMA_FIELDS:
            raise ValueError(f"unexpected generated schema: {sorted(row)}")
        by_normalized[row["normalized"]].append(row)
        by_group[row["group_id"]].append(row)
    duplicate_rows = sum(len(values) - 1 for values in by_normalized.values())
    cross_label_conflicts = sum(
        len({row["expected_feedback_type"] for row in values}) > 1
        for values in by_normalized.values()
    )
    if duplicate_rows or cross_label_conflicts:
        raise ValueError(
            "generated corpus has normalized collisions: "
            f"duplicates={duplicate_rows}, cross_label={cross_label_conflicts}"
        )
    if any(
        {row["expected_feedback_type"] for row in group} != set(FEEDBACK_TYPES)
        for group in by_group.values()
    ):
        raise ValueError("every contrast group must contain all three labels")

    text_hashes, holdout_groups, holdout_texts, gate = _load_holdout_gate(
        holdout_specs, paper_manifest_path
    )
    rejected = Counter({"exact_text": 0, "near_text": 0, "group_token": 0})
    accepted: list[dict] = []
    for group_id in sorted(by_group):
        group = by_group[group_id]
        reason = None
        if group_id in holdout_groups or any(
            row["template_family"] in holdout_groups for row in group
        ):
            reason = "group_token"
        elif any(_text_hash(row["text"]) in text_hashes for row in group):
            reason = "exact_text"
        elif any(
            _is_near_holdout(
                row["normalized"], holdout_texts, near_duplicate_threshold
            )
            for row in group
        ):
            reason = "near_text"
        if reason:
            rejected[reason] += 1
            continue
        accepted.extend(group)

    label_counts = Counter(row["expected_feedback_type"] for row in accepted)
    if len(set(label_counts.values())) != 1 or set(label_counts) != set(FEEDBACK_TYPES):
        raise AssertionError(f"leakage filtering broke class balance: {label_counts}")
    accepted_hashes = {_text_hash(row["text"]) for row in accepted}
    if accepted_hashes & text_hashes:
        raise AssertionError("accepted corpus has exact holdout text overlap")
    report = {
        "version": VERSION,
        "rows_before_leakage_filter": len(rows),
        "rows": len(accepted),
        "groups": len(accepted) // len(FEEDBACK_TYPES),
        "split": "train",
        "train_only": True,
        "generation": "deterministic_template_expansion",
        "runtime_generation_uses_llm": False,
        "template_authoring": "ai_assisted_codex",
        "independent_human_semantic_audit": False,
        "authoring_informed_by_model_error_analysis": True,
        "row_filtering_uses_model_predictions": False,
        "target_definition": "surface_feedback_form_fG_speech_act",
        "reference_type_used_as_target": False,
        "base_corpus": {
            "path": str(base_path),
            "sha256": _sha256(base_path),
            "rows": len(_read_json_list(base_path)),
        },
        "label_counts": dict(sorted(label_counts.items())),
        "canonical_label_counts": dict(
            sorted(Counter(row["classification_label"] for row in accepted).items())
        ),
        "normalized_duplicate_count": 0,
        "cross_label_exact_conflict_count": 0,
        "provenance_fields": ["source", "label_source", "group_id", "template_family"],
        "holdout_leakage_gate": {
            **gate,
            "labels_used": False,
            "metrics_computed": False,
            "heldout_contents_output": False,
            "development_text_used_for_leakage_filter": True,
            "test_text_used_for_leakage_filter": False,
            "test_files_opened": False,
            "exact_overlap_count": 0,
            "near_overlap_count": 0,
            "group_overlap_count": 0,
            "near_duplicate_threshold": near_duplicate_threshold,
            "groups_rejected": dict(sorted(rejected.items())),
        },
    }
    return accepted, report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, default=DEFAULT_BASE)
    parser.add_argument("--paper-manifest", type=Path, default=DEFAULT_PAPER_MANIFEST)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--near-duplicate-threshold", type=float, default=0.94)
    args = parser.parse_args()
    rows, report = build_boundary_corpus(
        args.base,
        DEFAULT_HOLDOUT_SPECS,
        args.paper_manifest,
        near_duplicate_threshold=args.near_duplicate_threshold,
    )
    write_json(args.output, rows)
    report["output"] = str(args.output)
    report["output_sha256"] = _sha256(args.output)
    write_json(args.report, report)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
