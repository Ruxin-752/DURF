"""Generate the clean-v5 five-bank direct-fG surface benchmark.

The corpus is generated from the current-game typed scenarios only.  It opens
no corpus, human, legacy, development, test, frozen, or membership data file.
Five natural lexical banks express the same typed task payloads with different
surface forms.  Bank identifiers are metadata only and are never rendered into
the feedback text.

This synthetic corpus is designed only for a surface-heldout benchmark.  It is
not human gold, an independent semantic benchmark, or player-domain evidence.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
import sys
import unicodedata

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts import generate_direct_fg_clean_v3 as ontology  # noqa: E402


DEFAULT_OUTPUT = (
    ROOT
    / "outputs"
    / "feedback_form_classifier_candidates"
    / "direct_fg_clean_v5"
    / "generated_corpus.json"
)
VERSION = "direct-fg-clean-v5-five-surface-banks-v1"
SOURCE = "direct_fg_current_ontology_surface_banks_v5"
LABEL_SOURCE = "direct_surface_semantic_contract"
LABELS = ontology.LABELS
CANONICAL_LABELS = ontology.CANONICAL_LABELS
GROUNDINGS_PER_SCENARIO = 6


@dataclass(frozen=True)
class LexicalBank:
    bank_id: str
    split_role: str
    evaluative_positive: str
    evaluative_negative: str
    imperative: str
    descriptive: str


BANKS = (
    LexicalBank(
        "bank_01",
        "train",
        "For {context}, that was a good move just now: {past}.",
        "For {context}, that was a bad move just now: {past}.",
        "For {context}, please {command} as your next move.",
        "A kitchen-state snapshot for {context} shows this condition: {fact}.",
    ),
    LexicalBank(
        "bank_02",
        "train",
        "I approve of your recent action during {context}: {past}.",
        "I disapprove of your recent action during {context}: {past}.",
        "During {context}, you need to {command} for the current task.",
        "During {context}, the presently observed condition is that {fact}.",
    ),
    LexicalBank(
        "bank_03",
        "train",
        "The last move in {context} was handled well when {past}.",
        "The last move in {context} was handled poorly when {past}.",
        "Could you {command} right away for {context}?",
        "This neutral status applies to {context} as things stand: {fact}.",
    ),
    LexicalBank(
        "bank_04",
        "calibration",
        "A moment ago in {context}, it was the right choice when {past}.",
        "A moment ago in {context}, it was the wrong choice when {past}.",
        "Make this your immediate priority in {context}: {command}.",
        "The live game record lists the following under {context}: {fact}.",
    ),
    LexicalBank(
        "bank_05",
        "final_eval",
        "What you did recently for {context} was helpful: {past}.",
        "What you did recently for {context} was unhelpful: {past}.",
        "Coordinate this kitchen task now for {context}: {command}.",
        "Right now, {fact}; this is the relevant fact for {context}.",
    ),
)


# These phrases are meaningful task contexts, not split markers or nonces.
SCENARIO_CONTEXT = {
    "get_tomato_dispenser": "the tomato pickup at the tomato dispenser",
    "get_onion_dispenser": "the onion pickup at the onion dispenser",
    "get_dish_dispenser": "the dish pickup at the dish dispenser",
    "get_tomato_counter": "the tomato pickup at an accessible counter",
    "get_onion_counter": "the onion pickup at an accessible counter",
    "get_dish_counter": "the dish pickup at an accessible counter",
    "get_soup_counter": "the soup pickup at an accessible counter",
    "put_tomato_empty_pot": "the tomato step with an empty pot",
    "put_second_tomato": "the second-tomato step at the pot",
    "put_onion_empty_pot": "the onion step with an empty pot",
    "put_onion_after_tomato": "the onion step after one tomato",
    "complete_with_tomato": "the final-tomato step for the soup",
    "complete_with_onion": "the final-onion step for the soup",
    "wait_while_cooking": "the wait while the soup cooks",
    "plate_ready_soup": "the ready-soup pickup at the pot",
    "serve_held_soup": "the soup delivery at the serving station",
    "stash_tomato": "the tomato placement on an accessible counter",
    "stash_onion": "the onion placement on an accessible counter",
    "stash_dish": "the dish placement on an accessible counter",
    "stash_soup": "the soup placement on an accessible counter",
    "yield_upper_corridor": "the yield in the upper corridor",
    "yield_lower_corridor": "the yield in the lower corridor",
    "clear_pot_access": "the clear access tile in front of the pot",
    "clear_serving_access": "the clear access tile at the serving station",
}


EVALUATIVE_PATTERN = re.compile(
    r"\b(?:good move|bad move|approve|disapprove|handled well|handled poorly|"
    r"right choice|wrong choice|helpful|unhelpful)\b",
    re.IGNORECASE,
)
COMPLETION_PATTERN = re.compile(
    r"\b(?:just now|recent action|last move|a moment ago|did recently)\b",
    re.IGNORECASE,
)
IMPERATIVE_PATTERNS = (
    re.compile(r"^For .+, please .+ as your next move\.$", re.IGNORECASE),
    re.compile(r"^During .+, you need to .+ for the current task\.$", re.IGNORECASE),
    re.compile(r"^Could you .+ right away for .+\?$", re.IGNORECASE),
    re.compile(r"^Make this your immediate priority in .+: .+\.$", re.IGNORECASE),
    re.compile(
        r"^Coordinate this kitchen task now for .+: .+\.$",
        re.IGNORECASE,
    ),
)
DESCRIPTIVE_PATTERNS = (
    re.compile(
        r"^A kitchen-state snapshot for .+ shows this condition: .+\.$",
        re.IGNORECASE,
    ),
    re.compile(
        r"^During .+, the presently observed condition is that .+\.$",
        re.IGNORECASE,
    ),
    re.compile(
        r"^This neutral status applies to .+ as things stand: .+\.$",
        re.IGNORECASE,
    ),
    re.compile(
        r"^The live game record lists the following under .+: .+\.$",
        re.IGNORECASE,
    ),
    re.compile(
        r"^Right now, .+; this is the relevant fact for .+\.$",
        re.IGNORECASE,
    ),
)


def normalize_text(text: str | None) -> str:
    return ontology.normalize_text(text)


def speech_act_matches(text: str) -> tuple[str, ...]:
    matches = []
    if EVALUATIVE_PATTERN.search(text) and COMPLETION_PATTERN.search(text):
        matches.append("evaluative")
    if any(pattern.fullmatch(text) for pattern in IMPERATIVE_PATTERNS):
        matches.append("imperative")
    if any(pattern.fullmatch(text) for pattern in DESCRIPTIVE_PATTERNS):
        matches.append("descriptive")
    return tuple(matches)


def _semantic_payload(scenario: ontology.TypedScenario, variant_index: int) -> dict:
    # Deliberately label- and bank-independent.  Temporal/speech-act information
    # belongs in speech_act_contract, so the E/I/D contrast shares this payload.
    return {
        "actor": "AI chef",
        "action": scenario.canonical_action,
        "object": scenario.object_name,
        "source": scenario.source,
        "target": scenario.target,
        "precondition": scenario.precondition,
        "result": scenario.result,
        "grounding_variant_index": variant_index,
    }


def _render(
    bank: LexicalBank,
    scenario: ontology.TypedScenario,
    variant: ontology.GroundingVariant,
    variant_index: int,
    label: str,
) -> tuple[str, str]:
    context = SCENARIO_CONTEXT[scenario.scenario_id]
    stance = "neutral"
    if label == "evaluative":
        stance = "positive" if variant_index % 2 == 0 else "negative"
        frame = (
            bank.evaluative_positive if stance == "positive" else bank.evaluative_negative
        )
    elif label == "imperative":
        frame = bank.imperative
    elif label == "descriptive":
        frame = bank.descriptive
    else:
        raise ValueError(f"unsupported direct-fG label: {label}")
    return frame.format(
        context=context,
        command=variant.command,
        past=variant.past,
        fact=variant.fact,
    ), stance


def generate_rows() -> list[dict]:
    ontology.validate_scenarios()
    if set(SCENARIO_CONTEXT) != {scenario.scenario_id for scenario in ontology.SCENARIOS}:
        raise ValueError("scenario context map must exactly cover the typed scenarios")
    rows = []
    for bank in BANKS:
        for scenario in ontology.SCENARIOS:
            context = SCENARIO_CONTEXT[scenario.scenario_id]
            for variant_index, variant in enumerate(
                scenario.variants[:GROUNDINGS_PER_SCENARIO]
            ):
                payload = _semantic_payload(scenario, variant_index)
                grounding_id = f"clean-v5:grounding:{scenario.scenario_id}:{variant_index:02d}"
                contrast_triad_id = (
                    f"clean-v5:{bank.bank_id}:contrast:{scenario.scenario_id}:{variant_index:02d}"
                )
                surface_frame_triad_id = f"clean-v5:{bank.bank_id}:surface-bank"
                for label in LABELS:
                    text, stance = _render(bank, scenario, variant, variant_index, label)
                    normalized = normalize_text(text)
                    feedback_id = hashlib.sha256(
                        (
                            f"{VERSION}|{bank.bank_id}|{scenario.scenario_id}|"
                            f"{variant_index}|{label}|{normalized}"
                        ).encode("utf-8")
                    ).hexdigest()[:24]
                    rows.append(
                        {
                            "feedback_id": feedback_id,
                            "text": text,
                            "normalized_text": normalized,
                            "expected_feedback_type": label,
                            "classification_label": CANONICAL_LABELS[label],
                            "source": SOURCE,
                            "label_source": LABEL_SOURCE,
                            "bank_id": bank.bank_id,
                            "bank_split_role": bank.split_role,
                            "scenario_id": scenario.scenario_id,
                            "scenario_group": f"clean-v5:scenario:{scenario.scenario_id}",
                            "grounding_id": grounding_id,
                            "contrast_triad_id": contrast_triad_id,
                            "surface_frame_triad_id": surface_frame_triad_id,
                            "surface_family": f"clean-v5:{bank.bank_id}:{label}:surface-bank",
                            "paraphrase_group": contrast_triad_id,
                            "frame_stance": stance,
                            "semantic_payload": dict(payload),
                            "speech_act_contract": {
                                "single_speech_act": True,
                                "direct_target": True,
                                "validator_expected_match": label,
                                "temporal_frame": {
                                    "evaluative": "recent_completed_event_with_judgment",
                                    "imperative": "current_or_future_request",
                                    "descriptive": "neutral_current_state_or_fact",
                                }[label],
                            },
                            "generation_provenance": {
                                "ai_assisted": True,
                                "human_gold": False,
                                "independent_human_semantic_audit": False,
                                "real_player_data": False,
                                "prior_error_informed": True,
                                "five_bank_single_eval_style_design": True,
                            },
                            "semantic_context_phrase": context,
                        }
                    )
    audit_rows(rows)
    return rows


def _canonical_payload_hash(value: dict) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def audit_rows(rows: list[dict]) -> dict:
    expected_rows = len(BANKS) * len(ontology.SCENARIOS) * GROUNDINGS_PER_SCENARIO * len(LABELS)
    if len(rows) != expected_rows or expected_rows != 2160:
        raise ValueError(f"clean-v5 requires exactly 2160 rows, got {len(rows)}")
    normalized = [str(row.get("normalized_text") or "") for row in rows]
    if len(set(normalized)) != len(normalized):
        raise ValueError(f"clean-v5 normalized exact duplicates: {len(normalized) - len(set(normalized))}")
    if any("reference_type" in row for row in rows):
        raise ValueError("reference_type is forbidden in clean-v5")

    forbidden_hits = []
    speech_act_failures = []
    grammar_failures = []
    hidden_identifier_hits = []
    invisible_character_hits = []
    bank_ids = {bank.bank_id.casefold() for bank in BANKS}
    for row in rows:
        text = str(row["text"])
        expected = str(row["expected_feedback_type"])
        matches = speech_act_matches(text)
        if matches != (expected,):
            speech_act_failures.append({"text": text, "expected": expected, "matches": matches})
        for pattern, compiled in zip(ontology.FORBIDDEN_PATTERNS, ontology.FORBIDDEN_RE):
            if compiled.search(text):
                forbidden_hits.append({"text": text, "pattern": pattern})
        if "{" in text or "}" in text:
            grammar_failures.append({"text": text, "reason": "unresolved_placeholder"})
        if ontology.BAD_GERUND_CONSTRUCTION_RE.search(text):
            grammar_failures.append({"text": text, "reason": "by_plus_base_verb"})
        if len(re.findall(r"[.!?]", text)) != 1 or text[-1:] not in ".!?":
            grammar_failures.append({"text": text, "reason": "not_one_terminal_sentence"})
        normalized_words = set(normalize_text(text).split())
        if normalized_words & bank_ids:
            hidden_identifier_hits.append(text)
        for character in text:
            category = unicodedata.category(character)
            if category.startswith("C") or (category.startswith("Z") and character != " "):
                invisible_character_hits.append(
                    {"feedback_id": row["feedback_id"], "codepoint": f"U+{ord(character):04X}"}
                )
                break
    if forbidden_hits:
        raise ValueError(f"forbidden ontology terms found: {forbidden_hits[:3]}")
    if speech_act_failures:
        raise ValueError(f"single speech-act validation failed: {speech_act_failures[:3]}")
    if grammar_failures:
        raise ValueError(f"surface grammar validation failed: {grammar_failures[:3]}")
    if hidden_identifier_hits:
        raise ValueError(f"bank identifiers leaked into text: {hidden_identifier_hits[:3]}")
    if invisible_character_hits:
        raise ValueError(f"invisible/control characters found: {invisible_character_hits[:3]}")

    expected_scenarios = {scenario.scenario_id for scenario in ontology.SCENARIOS}
    expected_actions = set(ontology.ALLOWED_ACTIONS)
    bank_audits = {}
    for bank in BANKS:
        bank_rows = [row for row in rows if row["bank_id"] == bank.bank_id]
        label_counts = Counter(row["expected_feedback_type"] for row in bank_rows)
        scenario_ids = {row["scenario_id"] for row in bank_rows}
        actions = {row["semantic_payload"]["action"] for row in bank_rows}
        action_label_support = Counter(
            (row["semantic_payload"]["action"], row["expected_feedback_type"])
            for row in bank_rows
        )
        if len(bank_rows) != 432 or label_counts != Counter({label: 144 for label in LABELS}):
            raise ValueError(f"unbalanced bank {bank.bank_id}: {label_counts}")
        if scenario_ids != expected_scenarios or actions != expected_actions:
            raise ValueError(f"incomplete scenario/action coverage in {bank.bank_id}")
        if any(action_label_support[(action, label)] <= 0 for action in expected_actions for label in LABELS):
            raise ValueError(f"incomplete action x label support in {bank.bank_id}")
        bank_audits[bank.bank_id] = {
            "split_role": bank.split_role,
            "rows": len(bank_rows),
            "label_counts": dict(sorted(label_counts.items())),
            "scenario_count": len(scenario_ids),
            "canonical_action_count": len(actions),
            "action_x_label_cells": len(action_label_support),
            "surface_family_count": len({row["surface_family"] for row in bank_rows}),
        }

    by_contrast: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_contrast[str(row["contrast_triad_id"])].append(row)
    for triad_id, triad_rows in by_contrast.items():
        labels = Counter(row["expected_feedback_type"] for row in triad_rows)
        payload_hashes = {_canonical_payload_hash(row["semantic_payload"]) for row in triad_rows}
        if len(triad_rows) != 3 or labels != Counter({label: 1 for label in LABELS}):
            raise ValueError(f"invalid E/I/D contrast triad {triad_id}: {labels}")
        if len(payload_hashes) != 1:
            raise ValueError(f"semantic payload differs inside triad {triad_id}")

    cross_bank_payloads: dict[tuple[str, str], set[str]] = defaultdict(set)
    for row in rows:
        key = (str(row["scenario_id"]), str(row["grounding_id"]))
        cross_bank_payloads[key].add(_canonical_payload_hash(row["semantic_payload"]))
    if any(len(hashes) != 1 for hashes in cross_bank_payloads.values()):
        raise ValueError("semantic payload differs across lexical banks")

    families_by_bank = {
        bank.bank_id: {row["surface_family"] for row in rows if row["bank_id"] == bank.bank_id}
        for bank in BANKS
    }
    for left_index, left in enumerate(BANKS):
        for right in BANKS[left_index + 1 :]:
            if families_by_bank[left.bank_id] & families_by_bank[right.bank_id]:
                raise ValueError("surface families must be bank-exclusive")

    metadata_cells = Counter(
        (
            row["bank_id"],
            row["scenario_id"],
            row["semantic_payload"]["action"],
            row["grounding_id"],
            row["expected_feedback_type"],
        )
        for row in rows
    )
    metadata_only_accuracy = sum(
        max(metadata_cells[(bank.bank_id, scenario.scenario_id, scenario.canonical_action, f"clean-v5:grounding:{scenario.scenario_id}:{variant_index:02d}", label)] for label in LABELS)
        for bank in BANKS
        for scenario in ontology.SCENARIOS
        for variant_index in range(GROUNDINGS_PER_SCENARIO)
    ) / len(rows)
    if abs(metadata_only_accuracy - 1.0 / 3.0) > 1e-12:
        raise ValueError(f"metadata-only empirical majority baseline is not chance: {metadata_only_accuracy}")

    return {
        "version": VERSION,
        "rows": len(rows),
        "normalized_unique_rows": len(set(normalized)),
        "label_counts": dict(sorted(Counter(row["expected_feedback_type"] for row in rows).items())),
        "bank_count": len(BANKS),
        "bank_audits": bank_audits,
        "contrast_triad_count": len(by_contrast),
        "contrast_triad_rows": 3,
        "semantic_payload_identical_inside_each_eid_triad": True,
        "semantic_payload_intentionally_shared_across_banks": True,
        "surface_families_bank_exclusive": True,
        "forbidden_ontology_hits": 0,
        "single_speech_act_failures": 0,
        "surface_grammar_failures": 0,
        "bank_identifier_or_nonce_text_hits": 0,
        "invisible_character_hits": 0,
        "metadata_only_empirical_majority_accuracy": metadata_only_accuracy,
        "metadata_only_baseline_is_chance": True,
        "external_input_files_opened": 0,
        "old_human_dev_test_frozen_rows_read": 0,
        "reference_type_used": False,
        "ai_assisted": True,
        "human_gold": False,
        "independent_human_semantic_audit": False,
        "real_player_accuracy_claim": False,
        "benchmark_scope": "five_bank_surface_heldout_synthetic_only",
        "style_limit": "five finite lexical banks with one held-out evaluation bank",
    }


def _write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--audit-only", action="store_true")
    args = parser.parse_args()
    rows = generate_rows()
    audit = audit_rows(rows)
    if not args.audit_only:
        _write_json(args.output.resolve(), rows)
    print(json.dumps(audit, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
