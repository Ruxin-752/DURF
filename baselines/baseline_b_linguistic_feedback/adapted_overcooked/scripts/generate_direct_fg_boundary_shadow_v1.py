"""Generate a small train-only direct-fG boundary augmentation candidate.

This is targeted synthetic training data, not an accuracy benchmark.  It uses
only the current tomato/onion game ontology and direct three-way labels.  No
legacy corpus, reference-type target, human row, development set, test set, or
frozen diagnostic is opened.  Known diagnostics are explicitly marked as
training-informed regressions and are never included in model fitting.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
import unicodedata


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = (
    ROOT
    / "outputs"
    / "feedback_form_classifier_candidates"
    / "direct_fg_boundary_shadow_v1"
)
VERSION = "direct-fg-boundary-shadow-v1"
SOURCE = "direct_fg_current_ontology_boundary_synthetic_train_v1"
LABEL_SOURCE = "direct_surface_semantic_contract"
LABELS = ("evaluative", "imperative", "descriptive")
CANONICAL_LABELS = {
    "evaluative": "Evaluative",
    "imperative": "Imperative",
    "descriptive": "Descriptive",
}
VARIANTS_PER_SCENARIO = 4

ALLOWED_ACTIONS = frozenset(
    {
        "GET_TOMATO",
        "GET_ONION",
        "GET_DISH",
        "PUT_TOMATO_IN_POT",
        "PUT_ONION_IN_POT",
        "PICKUP_SOUP",
        "SERVE_SOUP",
        "STASH_HELD_OBJECT",
        "YIELD_PATH",
        "WAIT",
    }
)
FORBIDDEN_PATTERNS = (
    r"\bchop(?:ped|ping)?\b",
    r"\bcut(?:ting)?\b",
    r"\bknife\b",
    r"\bboard\b",
    r"\bsink\b",
    r"\bwash(?:ed|ing)?\b",
    r"\bdirty\b",
    r"\b(?:fridge|refrigerator|oven|grill|stove|trash|bin)\b",
    r"\b(?:lettuce|meat|burger|rice|fish|cheese|bun)\b",
    r"\b(?:worktop|prep table|center table|central aisle|main aisle|middle path)\b",
    r"\bplate\b",
    r"\breference[_ -]?type\b",
)
FORBIDDEN_RE = tuple(re.compile(pattern, re.IGNORECASE) for pattern in FORBIDDEN_PATTERNS)


def normalize_text(text: str | None) -> str:
    value = unicodedata.normalize("NFKC", str(text or "")).casefold()
    return " ".join(re.sub(r"[^\w]+", " ", value, flags=re.UNICODE).split())


def _cap(text: str) -> str:
    return text[:1].upper() + text[1:]


@dataclass(frozen=True)
class BoundaryScenario:
    scenario_id: str
    canonical_action: str
    event: str
    command: str
    fact: str
    prohibited_action: str | None = None
    prohibited_gerund: str | None = None


SCENARIOS: tuple[BoundaryScenario, ...] = (
    BoundaryScenario("get_tomato_dispenser", "GET_TOMATO", "you picked up a tomato from the tomato dispenser", "pick up a tomato from the tomato dispenser", "the AI chef holds a tomato from the tomato dispenser"),
    BoundaryScenario("get_onion_dispenser", "GET_ONION", "you collected an onion at the onion dispenser", "collect an onion at the onion dispenser", "the AI chef holds an onion from the onion dispenser"),
    BoundaryScenario("get_dish_dispenser", "GET_DISH", "you took a dish from the dish dispenser", "take a dish from the dish dispenser", "the AI chef is holding a dish from the dish dispenser"),
    BoundaryScenario("get_tomato_counter", "GET_TOMATO", "you retrieved a tomato from an accessible counter", "retrieve a tomato from an accessible counter", "the AI chef carries a tomato taken from an accessible counter"),
    BoundaryScenario("get_onion_counter", "GET_ONION", "you got an onion from an accessible counter", "get an onion from an accessible counter", "an onion from an accessible counter is with the AI chef"),
    BoundaryScenario("put_tomato_empty_pot", "PUT_TOMATO_IN_POT", "you added the held tomato to the empty pot", "add the held tomato to the empty pot", "the pot contains one tomato"),
    BoundaryScenario("put_onion_after_tomato", "PUT_ONION_IN_POT", "you put the held onion into the pot after one tomato", "put the held onion into the pot after one tomato", "the pot contains one tomato and one onion"),
    BoundaryScenario("complete_with_tomato", "PUT_TOMATO_IN_POT", "you completed the recipe by adding the second tomato", "add the second tomato to complete the recipe", "the pot is cooking two tomatoes and one onion"),
    BoundaryScenario("wait_while_cooking", "WAIT", "you waited while the soup cooked", "wait while the soup cooks", "the pot is currently cooking", "move before the soup is ready", "moving before the soup is ready"),
    BoundaryScenario("dish_ready_soup", "PICKUP_SOUP", "you used the dish to collect ready soup from the pot", "use the dish to collect ready soup from the pot", "the AI chef holds ready soup in a dish"),
    BoundaryScenario("serve_held_soup", "SERVE_SOUP", "you delivered the held soup at the serving station", "deliver the held soup at the serving station", "the serving station has received the soup"),
    BoundaryScenario("stash_tomato", "STASH_HELD_OBJECT", "you put the tomato on an empty accessible counter", "put the tomato on an empty accessible counter", "an accessible counter holds a tomato"),
    BoundaryScenario("stash_onion", "STASH_HELD_OBJECT", "you set the onion on an empty accessible counter", "set the onion on an empty accessible counter", "an accessible counter holds an onion"),
    BoundaryScenario("stash_dish", "STASH_HELD_OBJECT", "you left the dish on an empty accessible counter", "leave the dish on an empty accessible counter", "an accessible counter holds a dish"),
    BoundaryScenario("stash_soup", "STASH_HELD_OBJECT", "you placed the soup on an empty accessible counter", "place the soup on an empty accessible counter", "an accessible counter holds soup"),
    BoundaryScenario("yield_upper_corridor", "YIELD_PATH", "you moved aside in the upper corridor", "move aside in the upper corridor", "the upper corridor is open for the other chef", "block the upper corridor", "blocking the upper corridor"),
    BoundaryScenario("yield_lower_corridor", "YIELD_PATH", "you cleared the lower corridor for the teammate", "clear the lower corridor for the teammate", "the teammate can pass through the lower corridor", "block the lower corridor", "blocking the lower corridor"),
    BoundaryScenario("clear_pot_access", "YIELD_PATH", "you stepped away from the access tile in front of the pot", "step away from the access tile in front of the pot", "the access tile in front of the pot is clear", "block the access tile in front of the pot", "blocking the access tile in front of the pot"),
    BoundaryScenario("clear_serving_access", "YIELD_PATH", "you opened the access tile in front of the serving station", "open the access tile in front of the serving station", "the access tile in front of the serving station is open", "block the access tile in front of the serving station", "blocking the access tile in front of the serving station"),
    BoundaryScenario("get_soup_counter", "PICKUP_SOUP", "you picked up soup from an accessible counter", "pick up soup from an accessible counter", "the AI chef carries soup from an accessible counter"),
)


EVALUATIVE_FRAMES = (
    ("positive", "Good move: {event}."),
    ("positive", "That last step was helpful because {event}."),
    ("positive", "You handled that well when {event}."),
    ("positive", "Nice work just now: {event}."),
    ("positive", "That was the right move; {event}."),
    ("positive", "Your completed action was effective because {event}."),
    ("positive", "The last move was useful: {event}."),
    ("positive", "That recent step was well timed because {event}."),
    ("positive", "Good decision a moment ago: {event}."),
    ("positive", "What you just did was helpful: {event}."),
    ("negative", "Bad move: {event}."),
    ("negative", "That last step was unhelpful because {event}."),
    ("negative", "You handled that poorly when {event}."),
    ("negative", "That was a mistake just now: {event}."),
    ("negative", "The completed move was wrong; {event}."),
    ("negative", "Your recent action was ineffective because {event}."),
    ("negative", "The last move was harmful: {event}."),
    ("negative", "That recent step was poorly timed because {event}."),
    ("negative", "Poor decision a moment ago: {event}."),
    ("negative", "What you just did was unhelpful: {event}."),
)

IMPERATIVE_FRAMES = (
    "{command_cap}.",
    "Please {command}.",
    "Could you {command}?",
    "Would you please {command}?",
    "Why don't you {command}?",
    "Can you {command} next?",
    "You should {command}.",
    "I need you to {command}.",
    "Please make your next move: {command}.",
    "The next thing to do is {command}.",
    "Go ahead and {command}.",
    "For this step, please {command}.",
    "Would you {command} now?",
    "Your next task is to {command}.",
    "Right now, {command}.",
    "Could you please {command}?",
    "I would like you to {command}.",
    "Make this your next move: {command}.",
    "Why not {command} now?",
    "Be sure to {command}.",
)

NEGATIVE_IMPERATIVE_FRAMES = (
    "Don't {prohibited_action}.",
    "Please don't {prohibited_action}.",
    "You should not {prohibited_action}.",
    "Would you please avoid {prohibited_gerund}?",
)

DESCRIPTIVE_FRAMES = (
    "{fact_cap}.",
    "Right now, {fact}.",
    "The current state is that {fact}.",
    "At the moment, {fact}.",
    "The kitchen shows that {fact}.",
    "As things stand, {fact}.",
    "The visible state is that {fact}.",
    "Currently, {fact}.",
    "The present fact is that {fact}.",
    "In the kitchen, {fact}.",
    "The status now is that {fact}.",
    "The current layout shows that {fact}.",
    "At present, {fact}.",
    "The game state shows that {fact}.",
    "For this order, {fact}.",
    "The situation right now is that {fact}.",
    "The kitchen currently has this fact: {fact}.",
    "The observed condition is that {fact}.",
    "As the round stands, {fact}.",
    "The present kitchen state is that {fact}.",
)


def _frame_index(scenario_index: int, variant_index: int) -> int:
    return (7 * scenario_index + 5 * variant_index) % 20


def _render(
    scenario: BoundaryScenario,
    scenario_index: int,
    variant_index: int,
    label: str,
) -> tuple[str, str, str]:
    frame_index = _frame_index(scenario_index, variant_index)
    if label == "evaluative":
        stance, frame = EVALUATIVE_FRAMES[frame_index]
        return frame.format(event=scenario.event), stance, f"eval-{frame_index:02d}"
    if label == "imperative":
        if scenario.prohibited_action is not None:
            frame = NEGATIVE_IMPERATIVE_FRAMES[variant_index]
            return (
                frame.format(
                    prohibited_action=scenario.prohibited_action,
                    prohibited_gerund=scenario.prohibited_gerund,
                ),
                "negative_command",
                f"negative-command-{variant_index:02d}",
            )
        frame = IMPERATIVE_FRAMES[frame_index]
        return (
            frame.format(command=scenario.command, command_cap=_cap(scenario.command)),
            "request",
            f"imperative-{frame_index:02d}",
        )
    if label == "descriptive":
        frame = DESCRIPTIVE_FRAMES[frame_index]
        return (
            frame.format(fact=scenario.fact, fact_cap=_cap(scenario.fact)),
            "neutral",
            f"descriptive-{frame_index:02d}",
        )
    raise ValueError(f"unknown label: {label}")


def _payload(scenario: BoundaryScenario) -> dict:
    return {
        "scenario_id": scenario.scenario_id,
        "canonical_action": scenario.canonical_action,
        "event": scenario.event,
        "command": scenario.command,
        "fact": scenario.fact,
    }


def generate_train_rows() -> list[dict]:
    rows: list[dict] = []
    for scenario_index, scenario in enumerate(SCENARIOS):
        for variant_index in range(VARIANTS_PER_SCENARIO):
            frame_index = _frame_index(scenario_index, variant_index)
            triad_id = f"boundary-shadow:{scenario.scenario_id}:{variant_index:02d}"
            for label in LABELS:
                text, stance, surface_name = _render(
                    scenario, scenario_index, variant_index, label
                )
                normalized = normalize_text(text)
                feedback_id = hashlib.sha256(
                    f"{VERSION}|{triad_id}|{label}|{normalized}".encode("utf-8")
                ).hexdigest()[:24]
                rows.append(
                    {
                        "feedback_id": feedback_id,
                        "text": text,
                        "normalized_text": normalized,
                        "expected_feedback_type": label,
                        "classification_label": CANONICAL_LABELS[label],
                        "split": "train",
                        "source": SOURCE,
                        "label_source": LABEL_SOURCE,
                        "scenario_id": scenario.scenario_id,
                        "canonical_action": scenario.canonical_action,
                        "variant_index": variant_index,
                        "contrast_triad_id": triad_id,
                        "surface_bundle_id": f"boundary-shadow:surface-{frame_index:02d}",
                        "surface_family": f"boundary-shadow:{label}:{surface_name}",
                        "stance": stance,
                        "semantic_payload": _payload(scenario),
                        "provenance": {
                            "ai_assisted": True,
                            "targeted_boundary_augmentation": True,
                            "training_informed": True,
                            "human_gold": False,
                            "real_player_data": False,
                            "independent_human_semantic_audit": False,
                        },
                    }
                )
    audit_train_rows(rows)
    return rows


KNOWN_REGRESSION: tuple[tuple[str, str], ...] = (
    ("Good move: you served the soup.", "evaluative"),
    ("That last step was helpful.", "evaluative"),
    ("You handled the pickup well.", "evaluative"),
    ("Bad move: you blocked the corridor.", "evaluative"),
    ("That recent action was unhelpful.", "evaluative"),
    ("The completed move was wrong.", "evaluative"),
    ("Nice work just now.", "evaluative"),
    ("Poor decision a moment ago.", "evaluative"),
    ("What you just did was effective.", "evaluative"),
    ("That was a mistake on the last step.", "evaluative"),
    ("Please get an onion.", "imperative"),
    ("Could you take a dish?", "imperative"),
    ("Would you please serve the soup?", "imperative"),
    ("Why don't you clear the upper corridor?", "imperative"),
    ("Can you wait while the soup cooks?", "imperative"),
    ("Could you leave the access tile in front of the pot open?", "imperative"),
    ("Would you leave the access tile in front of the serving station open?", "imperative"),
    ("You should put the tomato in the pot.", "imperative"),
    ("I need you to use the dish for the ready soup.", "imperative"),
    ("Pick up the tomato now.", "imperative"),
    ("At present, one tomato is inside the pot.", "descriptive"),
    ("The AI chef is holding a dish.", "descriptive"),
    ("Right now, the soup is cooking.", "descriptive"),
    ("The serving station has received soup.", "descriptive"),
    ("An accessible counter holds an onion.", "descriptive"),
    ("The upper corridor is open.", "descriptive"),
    ("The pot is ready.", "descriptive"),
    ("The AI chef carries soup in a dish.", "descriptive"),
    ("The current state is that the lower corridor is clear.", "descriptive"),
    ("Two tomatoes and one onion are cooking in the pot.", "descriptive"),
)


def generate_known_regression_rows() -> list[dict]:
    rows = []
    for index, (text, label) in enumerate(KNOWN_REGRESSION):
        rows.append(
            {
                "probe_id": f"known-regression-{index:02d}",
                "text": text,
                "expected_feedback_type": label,
                "role": "training_informed_regression_only",
                "independent_accuracy_gate": False,
            }
        )
    return rows


def generate_mixed_rejection_rows(train_rows: list[dict] | None = None) -> list[dict]:
    rows = []
    train = list(train_rows) if train_rows is not None else generate_train_rows()
    by_triad: dict[str, dict[str, dict]] = defaultdict(dict)
    for row in train:
        by_triad[row["contrast_triad_id"]][row["expected_feedback_type"]] = row
    for index, triad in enumerate(list(by_triad.values())[:24]):
        if index % 3 == 0:
            labels = ("evaluative", "imperative")
        elif index % 3 == 1:
            labels = ("evaluative", "descriptive")
        else:
            labels = ("imperative", "descriptive")
        left = triad[labels[0]]["text"].rstrip(".!?")
        right = triad[labels[1]]["text"]
        right = right[:1].lower() + right[1:]
        text = f"{left}; {right}"
        rows.append(
            {
                "probe_id": f"mixed-reject-{index:02d}",
                "text": text,
                "expected_behavior": "reject_mixed_speech_act",
                "speech_acts": list(labels),
                "role": "synthetic_rule_rejection_diagnostic",
            }
        )
    return rows


EVALUATION_RE = re.compile(
    r"\b(?:good|bad|helpful|unhelpful|well|poorly|nice work|right move|"
    r"wrong|effective|ineffective|useful|harmful|well timed|poorly timed|"
    r"good decision|poor decision|mistake)\b",
    re.IGNORECASE,
)
REQUEST_RE = re.compile(
    r"\b(?:please|could you|would you|why don'?t you|can you|you should|"
    r"i need you|next move|next thing to do|go ahead|next task|"
    r"i would like you|why not|be sure|don'?t|should not|avoid)\b",
    re.IGNORECASE,
)
DESCRIPTIVE_RE = re.compile(
    r"\b(?:current state|at the moment|kitchen shows|as things stand|"
    r"visible state|currently|present fact|in the kitchen|status now|"
    r"current layout|at present|game state|for this order|observed condition|"
    r"round stands|present kitchen state|situation right now)\b",
    re.IGNORECASE,
)
DIRECT_COMMAND_RE = re.compile(
    r"^(?:right now,\s*)?(?:pick up|collect|take|get|retrieve|add|put|wait|use|deliver|set|"
    r"leave|place|move|clear|step|open)\b",
    re.IGNORECASE,
)
BARE_FACT_RE = re.compile(
    r"^(?:right now,\s*)?(?:the AI chef|the soup|the pot|the serving station|the upper corridor|"
    r"the access tile|the teammate|an accessible counter|an onion|"
    r"two tomatoes and one onion)\b.*\b(?:is|are|has|holds|carries|contains|"
    r"can pass|has received)\b",
    re.IGNORECASE,
)


def detect_speech_acts(text: str) -> tuple[str, ...]:
    """Detect explicit acts for the shadow-only mixed-sentence reject policy."""
    acts: set[str] = set()
    clauses = [part.strip(" .?!") for part in re.split(r"[;\n]", str(text))]
    for clause in clauses:
        if not clause:
            continue
        if EVALUATION_RE.search(clause):
            acts.add("evaluative")
        if REQUEST_RE.search(clause) or DIRECT_COMMAND_RE.search(clause):
            acts.add("imperative")
        if DESCRIPTIVE_RE.search(clause) or BARE_FACT_RE.search(clause):
            acts.add("descriptive")
    return tuple(label for label in LABELS if label in acts)


def audit_train_rows(rows: list[dict]) -> dict:
    failures: list[str] = []
    labels = Counter(row.get("expected_feedback_type") for row in rows)
    if len(rows) != 240 or labels != Counter({label: 80 for label in LABELS}):
        failures.append("train_rows_and_labels_exact")
    normalized = [row.get("normalized_text") for row in rows]
    if len(set(normalized)) != 240 or any(
        row.get("normalized_text") != normalize_text(row.get("text")) for row in rows
    ):
        failures.append("normalized_text_unique_and_recomputed")
    if any(row.get("split") != "train" for row in rows):
        failures.append("train_only")
    if any("reference_type" in row or "reference_type" in row.get("semantic_payload", {}) for row in rows):
        failures.append("direct_three_class_schema_only")
    if any(row.get("canonical_action") not in ALLOWED_ACTIONS for row in rows):
        failures.append("current_ontology_actions_only")
    if any(
        compiled.search(str(row.get("text") or ""))
        for row in rows
        for compiled in FORBIDDEN_RE
    ):
        failures.append("forbidden_ontology_zero")
    triads: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        triads[row["contrast_triad_id"]].append(row)
    if len(triads) != 80 or any(
        len(group) != 3
        or Counter(row["expected_feedback_type"] for row in group)
        != Counter({label: 1 for label in LABELS})
        or len(
            {
                json.dumps(row["semantic_payload"], sort_keys=True)
                for row in group
            }
        )
        != 1
        for group in triads.values()
    ):
        failures.append("eid_triad_contract")
    scenario_label = Counter(
        (row["scenario_id"], row["expected_feedback_type"]) for row in rows
    )
    if len(scenario_label) != 60 or set(scenario_label.values()) != {4}:
        failures.append("scenario_label_support_exact")
    bundle_label = Counter(
        (row["surface_bundle_id"], row["expected_feedback_type"]) for row in rows
    )
    if len(bundle_label) != 60 or set(bundle_label.values()) != {4}:
        failures.append("surface_bundle_label_support_exact")
    negative_commands = sum(row.get("stance") == "negative_command" for row in rows)
    if negative_commands != 20:
        failures.append("negative_command_count_exact")
    scenario_index = {scenario.scenario_id: index for index, scenario in enumerate(SCENARIOS)}
    frozen_renderer_ok = True
    direct_act_ok = True
    for row in rows:
        scenario = next(
            (value for value in SCENARIOS if value.scenario_id == row.get("scenario_id")),
            None,
        )
        variant = row.get("variant_index")
        label = row.get("expected_feedback_type")
        if scenario is None or type(variant) is not int or label not in LABELS:
            frozen_renderer_ok = False
            direct_act_ok = False
            continue
        expected_text, expected_stance, expected_surface = _render(
            scenario, scenario_index[scenario.scenario_id], variant, label
        )
        frozen_renderer_ok = frozen_renderer_ok and (
            row.get("text") == expected_text
            and row.get("stance") == expected_stance
            and row.get("surface_family")
            == f"boundary-shadow:{label}:{expected_surface}"
            and row.get("semantic_payload") == _payload(scenario)
        )
        direct_act_ok = direct_act_ok and detect_speech_acts(expected_text) == (label,)
    if not frozen_renderer_ok:
        failures.append("row_equals_frozen_direct_renderer")
    if not direct_act_ok:
        failures.append("single_direct_speech_act_matches_label")
    known = generate_known_regression_rows()
    if Counter(row["expected_feedback_type"] for row in known) != Counter(
        {label: 10 for label in LABELS}
    ) or any(row["role"] != "training_informed_regression_only" for row in known):
        failures.append("known_regression_is_balanced_and_non_gate")
    known_normalized = [normalize_text(row["text"]) for row in known]
    train_normalized = {row["normalized_text"] for row in rows}
    if len(set(known_normalized)) != len(known_normalized) or train_normalized.intersection(
        known_normalized
    ):
        failures.append("known_regression_normalized_unique_and_train_disjoint")
    if any(
        detect_speech_acts(row["text"]) != (row["expected_feedback_type"],)
        for row in known
    ):
        failures.append("known_regression_single_direct_speech_act_matches_label")
    mixed = generate_mixed_rejection_rows(rows)
    if len(mixed) != 24 or any(
        set(detect_speech_acts(row["text"])) != set(row["speech_acts"])
        or len(row["speech_acts"]) != 2
        for row in mixed
    ):
        failures.append("mixed_speech_act_rejection_contract")
    if failures:
        raise ValueError("boundary-shadow generation audit failed: " + ", ".join(failures))
    return {
        "version": VERSION,
        "status": "passed_train_data_contract",
        "rows": len(rows),
        "label_counts": dict(sorted(labels.items())),
        "scenario_count": len(SCENARIOS),
        "surface_bundle_count": len({row["surface_bundle_id"] for row in rows}),
        "contrast_triad_count": len(triads),
        "negative_command_rows": negative_commands,
        "known_regression_rows": len(KNOWN_REGRESSION),
        "known_regression_train_normalized_exact_overlap": 0,
        "known_regression_single_direct_speech_act_passed": True,
        "mixed_rejection_rows": 24,
        "mixed_rejection_detector_passed": True,
        "external_input_files_opened": 0,
        "old_human_dev_test_frozen_rows_read": 0,
        "reference_type_used_as_target": False,
        "accuracy_claim_role": "targeted_train_augmentation_only",
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
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--audit-only", action="store_true")
    args = parser.parse_args()
    train_rows = generate_train_rows()
    report = audit_train_rows(train_rows)
    if not args.audit_only:
        output_dir = args.output_dir.resolve()
        _write_json(output_dir / "train.json", train_rows)
        _write_json(output_dir / "known_regression.json", generate_known_regression_rows())
        _write_json(output_dir / "mixed_rejection.json", generate_mixed_rejection_rows(train_rows))
        _write_json(output_dir / "data_audit.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
