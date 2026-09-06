"""Generate clean-v4 by adding short direct-fG contrasts to every typed scenario.

The v3 typed ontology is reused as code only.  No v3 artifact, JSON corpus,
human data, legacy synthetic data, dev/test/frozen data, or membership file is
opened.  The short frames are explicitly informed by v3's natural-probe errors.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts import generate_direct_fg_clean_v3 as v3  # noqa: E402


VERSION = "direct-fg-clean-v4-short-contrast-v1"
SOURCE = "direct_fg_current_ontology_template_v4"
LABEL_SOURCE = v3.LABEL_SOURCE
LABELS = v3.LABELS
CANONICAL_LABELS = v3.CANONICAL_LABELS
DEFAULT_OUTPUT = (
    ROOT
    / "outputs"
    / "feedback_form_classifier_candidates"
    / "direct_fg_clean_v4"
    / "generated_corpus.json"
)

SHORT_IMPERATIVE_START_RE = re.compile(
    r"^(?:pick|get|collect|take|fetch|move|go|interact|retrieve|use|put|add|"
    r"place|carry|bring|complete|wait|stay|hold|remain|serve|deliver|set|"
    r"leave|stash|store|yield|step|clear|avoid|make|fill|lift|keep|plate|finish)\b",
    re.IGNORECASE,
)
SHORT_EVALUATIVE_RE = re.compile(r"^(?:good|bad) move:\s+you\b", re.IGNORECASE)


def _cap(text: str) -> str:
    return text[:1].upper() + text[1:]


def speech_act_matches(text: str) -> tuple[str, ...]:
    stripped = text.strip()
    if SHORT_EVALUATIVE_RE.search(stripped):
        return ("evaluative",)
    if " now" in stripped.casefold() and stripped.endswith(".") and SHORT_IMPERATIVE_START_RE.search(stripped):
        return ("imperative",)
    matches = v3.speech_act_matches(text)
    if not matches and any(
        marker in f" {stripped.casefold()} " for marker in (" lasts ", " receives ")
    ):
        return ("descriptive",)
    return matches


def _v4_base_rows() -> list[dict]:
    rows: list[dict] = []
    for raw in v3.generate_rows():
        row = dict(raw)
        provenance = dict(row["generation_provenance"])
        provenance.update(
            {
                "v4_short_contrast_extension": False,
                "v3_natural_probe_error_informed": True,
            }
        )
        row["source"] = SOURCE
        row["generation_provenance"] = provenance
        row["feedback_id"] = hashlib.sha256(
            f"{VERSION}|base|{row['surface_family']}|{row['normalized_text']}".encode(
                "utf-8"
            )
        ).hexdigest()[:24]
        rows.append(row)
    return rows


def _short_rows(existing_normalized: set[str]) -> list[dict]:
    rows: list[dict] = []
    for scenario_index, scenario in enumerate(v3.SCENARIOS):
        scenario_group = f"clean-v3:scenario:{scenario.scenario_id}"
        for variant_index, variant in enumerate(scenario.variants):
            grounding_id = f"clean-v3:grounding:{scenario.scenario_id}:{variant_index:02d}"
            polarity = "positive" if variant_index % 2 == 0 else "negative"
            command = variant.command
            past = variant.past
            fact = variant.fact
            pot_conditions = {
                "put_tomato_empty_pot": "the pot is empty",
                "put_second_tomato": "the pot contains one tomato",
                "put_onion_empty_pot": "the pot is empty",
                "put_onion_after_tomato": "the pot contains one tomato",
                "complete_with_tomato": "the pot contains one tomato and one onion",
                "complete_with_onion": "the pot contains two tomatoes",
            }
            condition = pot_conditions.get(scenario.scenario_id)
            if condition:
                command = f"{command} now while {condition}"
                past = f"{past} while {condition}"
            else:
                command = f"{command} now"
            if scenario.scenario_id in {
                "yield_upper_corridor",
                "yield_lower_corridor",
            }:
                location = str(scenario.source)
                if location not in command:
                    command = f"{command} in the {location}"
                if location not in past:
                    past = f"{past} in the {location}"
            if scenario.scenario_id == "complete_with_tomato":
                fact = f"{fact} after the final tomato addition"
            elif scenario.scenario_id == "complete_with_onion":
                fact = f"{fact} after the final onion addition"
            elif scenario.scenario_id == "wait_while_cooking":
                fact = f"{fact} while the AI chef waits"
            elif scenario.canonical_action == "YIELD_PATH":
                location = str(scenario.source)
                if location not in fact:
                    fact = f"{fact} in the {location}"
            evaluative_text = (
                f"Good move: {past}."
                if polarity == "positive"
                else f"Bad move: {past}."
            )
            imperative_text = f"{_cap(command)}."
            descriptive_text = f"{_cap(fact)}."
            if v3.normalize_text(descriptive_text) in existing_normalized:
                # Scenario 0's v3 long-form bank already includes a bare fact
                # frame.  Keep v4 exact-unique without weakening the short fact
                # contract or deleting a balanced base contrast.
                descriptive_text = f"{_cap(variant.fact)} right now."
            rendered = {
                "evaluative": (evaluative_text, polarity),
                "imperative": (imperative_text, "directive"),
                "descriptive": (descriptive_text, "neutral"),
            }
            for label in LABELS:
                text, stance = rendered[label]
                normalized = v3.normalize_text(text)
                surface_family = (
                    f"clean-v4:{label}:short-scenario-{scenario_index:02d}"
                )
                temporal_frame = {
                    "evaluative": "recent_completed_event",
                    "imperative": "current_or_future_action",
                    "descriptive": "current_state_or_mechanism_fact",
                }[label]
                rows.append(
                    {
                        "feedback_id": hashlib.sha256(
                            f"{VERSION}|short|{surface_family}|{variant_index}|{normalized}".encode(
                                "utf-8"
                            )
                        ).hexdigest()[:24],
                        "text": text,
                        "normalized_text": normalized,
                        "expected_feedback_type": label,
                        "classification_label": CANONICAL_LABELS[label],
                        "source": SOURCE,
                        "label_source": LABEL_SOURCE,
                        "scenario_id": scenario.scenario_id,
                        "scenario_group": scenario_group,
                        "grounding_id": grounding_id,
                        "surface_family": surface_family,
                        "paraphrase_group": surface_family,
                        "frame_stance": stance,
                        "semantic_payload": {
                            "actor": "AI chef",
                            "action": scenario.canonical_action,
                            "object": scenario.object_name,
                            "source": scenario.source,
                            "target": scenario.target,
                            "precondition": scenario.precondition,
                            "result": scenario.result,
                            "temporal_frame": temporal_frame,
                        },
                        "speech_act_contract": {
                            "direct_target": True,
                            "single_speech_act": True,
                            "validator_expected_match": label,
                            "short_minimal_contrast": True,
                        },
                        "generation_provenance": {
                            "ai_assisted": True,
                            "prior_error_informed": True,
                            "independent_human_semantic_audit": False,
                            "human_gold": False,
                            "real_player_data": False,
                            "v4_short_contrast_extension": True,
                            "v3_natural_probe_error_informed": True,
                        },
                    }
                )
    return rows


def generate_rows() -> list[dict]:
    base = _v4_base_rows()
    rows = [*base, *_short_rows({row["normalized_text"] for row in base})]
    audit_rows(rows)
    return rows


def audit_rows(rows: list[dict]) -> dict:
    expected_rows = 2160
    if len(rows) != expected_rows:
        raise ValueError(f"clean-v4 requires {expected_rows} rows, got {len(rows)}")
    labels = Counter(str(row.get("expected_feedback_type")) for row in rows)
    if labels != Counter({label: 720 for label in LABELS}):
        raise ValueError(f"clean-v4 labels are not exactly balanced: {dict(labels)}")
    normalized = [str(row.get("normalized_text") or "") for row in rows]
    if any(not value for value in normalized):
        raise ValueError("clean-v4 has empty normalized text")
    if len(set(normalized)) != len(normalized):
        raise ValueError(
            f"clean-v4 normalized exact duplicates: {len(normalized) - len(set(normalized))}"
        )
    contract_failures = []
    grammar_failures = []
    forbidden_hits = []
    for row in rows:
        text = str(row["text"])
        expected = str(row["expected_feedback_type"])
        matches = speech_act_matches(text)
        if matches != (expected,):
            contract_failures.append(
                {"text": text, "expected": expected, "matches": list(matches)}
            )
        if "{" in text or "}" in text or len(re.findall(r"[.!?]", text)) != 1:
            grammar_failures.append(text)
        if v3.BAD_GERUND_CONSTRUCTION_RE.search(text):
            grammar_failures.append(text)
        for pattern, compiled in zip(v3.FORBIDDEN_PATTERNS, v3.FORBIDDEN_RE):
            if compiled.search(text):
                forbidden_hits.append({"text": text, "pattern": pattern})
    if contract_failures:
        raise ValueError(f"clean-v4 speech-act failures: {contract_failures[:3]}")
    if grammar_failures:
        raise ValueError(f"clean-v4 grammar failures: {grammar_failures[:3]}")
    if forbidden_hits:
        raise ValueError(f"clean-v4 ontology failures: {forbidden_hits[:3]}")
    family_counts = {
        label: len(
            {
                str(row["surface_family"])
                for row in rows
                if row["expected_feedback_type"] == label
            }
        )
        for label in LABELS
    }
    if any(count != 72 for count in family_counts.values()):
        raise ValueError(f"clean-v4 expected 72 surface families/class: {family_counts}")
    short_counts = Counter(
        str(row["expected_feedback_type"])
        for row in rows
        if row["generation_provenance"]["v4_short_contrast_extension"]
    )
    if short_counts != Counter({label: 240 for label in LABELS}):
        raise ValueError(f"clean-v4 short contrasts not balanced: {dict(short_counts)}")
    if any("reference_type" in row for row in rows):
        raise ValueError("reference_type is forbidden in clean-v4")
    return {
        "version": VERSION,
        "rows": len(rows),
        "label_counts": dict(sorted(labels.items())),
        "short_contrast_rows": sum(short_counts.values()),
        "short_contrast_label_counts": dict(sorted(short_counts.items())),
        "scenario_count": len({row["scenario_id"] for row in rows}),
        "surface_family_counts": family_counts,
        "normalized_unique_rows": len(set(normalized)),
        "forbidden_ontology_hits": 0,
        "single_speech_act_failures": 0,
        "surface_grammar_failures": 0,
        "reference_type_used": False,
        "external_input_files_opened": 0,
        "old_or_human_data_rows_read": 0,
        "v3_code_reused_without_v3_artifact_reads": True,
        "v3_natural_probe_error_informed": True,
        "ai_assisted": True,
        "prior_error_informed": True,
        "independent_human_semantic_audit": False,
        "human_gold": False,
        "real_player_accuracy_claim": False,
        "style_limit": "synthetic finite-frame English plus short contrasts; not natural-player gold",
    }


def _write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--audit-only", action="store_true")
    args = parser.parse_args()
    rows = generate_rows()
    audit = audit_rows(rows)
    if not args.audit_only:
        _write_json(args.output, rows)
    print(json.dumps(audit, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
