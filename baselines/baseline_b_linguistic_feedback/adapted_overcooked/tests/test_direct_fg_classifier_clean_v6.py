from __future__ import annotations

from collections import Counter, defaultdict
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import unicodedata
import unittest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts import audit_direct_fg_clean_v6_split as split_audit
from scripts import generate_direct_fg_clean_v6 as generator


EXPECTED_SIMILARITY_COUNTS = {
    "exact": 0,
    "sequence": 0,
    "jaccard": 0,
    "either": 0,
}
LOOKUP_VIEWS = (
    "full_text_raw",
    "full_text_entity_masked",
    "core_text_raw",
    "core_text_entity_masked",
)
MODEL_SUFFIXES = {".joblib", ".pkl", ".pickle", ".pt", ".pth", ".onnx"}


def _canonical_json_sha256(value) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _terminal_punctuation(text: str) -> str:
    return text[-1] if text[-1:] in ".!?" else "<none>"


def _initial_case(text: str) -> str:
    first = text[:1]
    if first.isupper():
        return "upper"
    if first.islower():
        return "lower"
    return "other"


def _first_token(text: str) -> str:
    normalized = generator.normalize_text(text)
    return normalized.split()[0] if normalized else "<empty>"


def _majority_lookup_accuracy(rows: list[dict], feature) -> float:
    groups: dict[object, Counter] = defaultdict(Counter)
    for row in rows:
        groups[feature(row)][row["expected_feedback_type"]] += 1
    return sum(max(counts.values()) for counts in groups.values()) / len(rows)


def _empirical_ks(left: list[int], right: list[int]) -> float:
    points = sorted(set(left) | set(right))
    return max(
        abs(
            sum(value <= point for value in left) / len(left)
            - sum(value <= point for value in right) / len(right)
        )
        for point in points
    )


def _matching_cross_partition_pair(rows: list[dict]) -> tuple[int, int]:
    """Return equal-label/equal-payload rows from train and final-eval banks."""

    candidates: dict[tuple[str, str, str], int] = {}
    for index, row in enumerate(rows):
        key = (
            row["scenario_id"],
            row["grounding_id"],
            row["expected_feedback_type"],
        )
        if row["bank_split_role"] == "train":
            candidates.setdefault(key, index)
        elif row["bank_split_role"] == "final_eval" and key in candidates:
            return candidates[key], index
    raise AssertionError("no matching train/final-eval semantic pair")


def _set_core_text(row: dict, core_text: str) -> None:
    row["core_text"] = core_text
    if "normalized_core_text" in row:
        row["normalized_core_text"] = generator.normalize_text(core_text)


def _set_text(row: dict, text: str) -> None:
    row["text"] = text
    row["normalized_text"] = generator.normalize_text(text)


def _append_before_terminal(text: str, suffix: str) -> str:
    terminal = text[-1:] if text[-1:] in ".!?" else ""
    stem = text[:-1] if terminal else text
    return f"{stem.rstrip()} {suffix}{terminal}"


def _combine_sentences(left: str, right: str) -> str:
    left_stem = left.rstrip().rstrip(".!?")
    right_stem = right.strip()
    if right_stem:
        right_stem = right_stem[:1].lower() + right_stem[1:]
    return f"{left_stem}; {right_stem}"


def _row_for(rows: list[dict], *, label: str, scenario_fragment: str = "") -> dict:
    return next(
        row
        for row in rows
        if row["expected_feedback_type"] == label
        and scenario_fragment in row["scenario_id"]
    )


def _sequence_only_variant(text: str) -> str:
    tokens = generator.normalize_text(text).split()
    for changed_count in range(1, len(tokens) + 1):
        changed = [
            token + "s" if index < changed_count else token
            for index, token in enumerate(tokens)
        ]
        candidate = " ".join(changed)
        if split_audit.similarity_flags(
            generator.normalize_text(text), candidate
        ) == (False, True, False):
            return candidate
    raise AssertionError("could not construct a sequence-only core mutation")


def _jaccard_only_variant(text: str) -> str:
    tokens = generator.normalize_text(text).split()
    candidates = (
        list(reversed(tokens)),
        tokens[::2] + tokens[1::2],
        tokens[len(tokens) // 2 :] + tokens[: len(tokens) // 2],
    )
    for candidate_tokens in candidates:
        candidate = " ".join(candidate_tokens)
        if split_audit.similarity_flags(
            generator.normalize_text(text), candidate
        ) == (False, False, True):
            return candidate
    raise AssertionError("could not construct a jaccard-only core mutation")


class CleanV6GenerationAndSplitTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rows = generator.generate_rows()
        cls.generation = generator.audit_rows(cls.rows)
        cls.report = split_audit.audit_split(cls.rows)

    def _run_rejected_case(
        self,
        records: list[dict],
        mutation_id: str,
        variant: str,
        targets: list[str],
        expected_failure_codes: tuple[str, ...],
        mutate,
    ) -> None:
        mutated = deepcopy(self.rows)
        before_rows = deepcopy(mutated)
        pre_hash = _canonical_json_sha256(mutated)
        mutate(mutated)
        post_hash = _canonical_json_sha256(mutated)
        self.assertNotEqual(pre_hash, post_hash, f"{mutation_id}/{variant} made no change")
        target_rows = []
        for index in range(max(len(before_rows), len(mutated))):
            before = before_rows[index] if index < len(before_rows) else None
            after = mutated[index] if index < len(mutated) else None
            if before != after:
                feedback_id = (before or after or {}).get("feedback_id", "<missing>")
                target_rows.append(f"{index}:{feedback_id}")
        self.assertTrue(target_rows, f"{mutation_id}/{variant} has no target rows")
        with self.assertRaises(split_audit.AuditFailure) as caught:
            split_audit.audit_split(mutated, raise_on_failure=True)
        observed = tuple(caught.exception.failure_codes)
        self.assertTrue(caught.exception.report)
        for code in expected_failure_codes:
            self.assertIn(code, observed, f"{mutation_id}/{variant}: {observed}")
        records.append(
            {
                "mutation_id": mutation_id,
                "variant": variant,
                "targets": list(targets),
                "target_rows": target_rows,
                "target_row_count": len(target_rows),
                "target_rows_sha256": _canonical_json_sha256(target_rows),
                "pre_sha256": pre_hash,
                "post_sha256": post_hash,
                "expected_failure_codes": list(expected_failure_codes),
                "observed_failure_codes": list(observed),
                "passed": True,
            }
        )

    def _run_control_case(
        self,
        records: list[dict],
        control_id: str,
        targets: list[str],
        mutate=lambda rows: None,
    ) -> None:
        controlled = deepcopy(self.rows)
        pre_hash = _canonical_json_sha256(controlled)
        mutate(controlled)
        post_hash = _canonical_json_sha256(controlled)
        report = split_audit.audit_split(controlled, raise_on_failure=True)
        self.assertTrue(report["quality_gate_passed"], control_id)
        records.append(
            {
                "mutation_id": control_id,
                "variant": "control",
                "targets": list(targets),
                "target_rows": [],
                "target_row_count": 0,
                "target_rows_sha256": _canonical_json_sha256([]),
                "pre_sha256": pre_hash,
                "post_sha256": post_hash,
                "expected_failure_codes": [],
                "observed_failure_codes": [],
                "passed": True,
            }
        )

    def test_generation_is_balanced_unique_and_raw_only(self):
        self.assertEqual(len(self.rows), 2160)
        self.assertEqual(
            Counter(row["expected_feedback_type"] for row in self.rows),
            Counter({"evaluative": 720, "imperative": 720, "descriptive": 720}),
        )
        self.assertEqual(len({row["normalized_text"] for row in self.rows}), 2160)
        self.assertEqual(self.generation["external_input_files_opened"], 0)
        self.assertEqual(self.generation["old_human_dev_test_frozen_rows_read"], 0)
        metadata_groups = defaultdict(Counter)
        for row in self.rows:
            metadata_groups[
                (
                    row["bank_id"],
                    row["scenario_id"],
                    row["grounding_id"],
                    row["semantic_payload"]["action"],
                )
            ][row["expected_feedback_type"]] += 1
        metadata_accuracy = sum(
            max(counts.values()) for counts in metadata_groups.values()
        ) / len(self.rows)
        self.assertEqual(metadata_accuracy, 1.0 / 3.0)
        self.assertTrue(
            self.generation["checks"]["model_feature_allowlist_raw_text_only"]
        )

        for row in self.rows:
            self.assertNotIn("reference_type", row)
            self.assertNotIn("reference_type", row["semantic_payload"])
            self.assertNotIn("classification_label", row["semantic_payload"])
            self.assertNotIn("expected_feedback_type", row["semantic_payload"])
            self.assertFalse(row["generation_provenance"]["human_gold"])
            self.assertFalse(row["generation_provenance"]["real_player_data"])

        source = Path(generator.__file__).read_text(encoding="utf-8")
        self.assertNotRegex(
            source,
            r"(?:from|import)\s+[^\n]*generate_direct_fg_clean_v5",
        )
        self.assertNotRegex(source, r"\.variants\b")
        self.assertNotRegex(source, r"\bvariant\.(?:command|past|fact)\b")
        self.assertNotIn("reference_classifier_feedback", source)

    def test_five_banks_have_12_balanced_outer_frames_per_label(self):
        self.assertEqual(len(generator.BANKS), 5)
        self.assertEqual(
            Counter(bank.split_role for bank in generator.BANKS),
            Counter({"train": 3, "calibration": 1, "final_eval": 1}),
        )
        for bank in generator.BANKS:
            bank_rows = [row for row in self.rows if row["bank_id"] == bank.bank_id]
            self.assertEqual(len(bank_rows), 432)
            self.assertEqual(
                Counter(row["expected_feedback_type"] for row in bank_rows),
                Counter({label: 144 for label in generator.LABELS}),
            )
            self.assertEqual(len({row["scenario_id"] for row in bank_rows}), 24)
            self.assertEqual(
                {row["semantic_payload"]["action"] for row in bank_rows},
                set(generator.ALLOWED_ACTIONS),
            )

            for label in generator.LABELS:
                selected = [
                    row
                    for row in bank_rows
                    if row["expected_feedback_type"] == label
                ]
                frames = Counter(row["outer_frame_id"] for row in selected)
                self.assertEqual(len(frames), 12)
                self.assertEqual(set(frames.values()), {12})
                slots = Counter(row["outer_frame_slot"] for row in selected)
                self.assertEqual(slots, Counter({slot: 12 for slot in range(12)}))
                first_tokens = Counter(_first_token(row["text"]) for row in selected)
                self.assertEqual(len(first_tokens), 12)
                self.assertEqual(set(first_tokens.values()), {12})
                self.assertEqual(
                    Counter(_terminal_punctuation(row["text"]) for row in selected),
                    Counter({".": 120, "<none>": 24}),
                )
                self.assertEqual(
                    Counter(_initial_case(row["text"]) for row in selected),
                    Counter({"upper": 108, "lower": 36}),
                )

            evaluative_stances = Counter(
                row["frame_stance"]
                for row in bank_rows
                if row["expected_feedback_type"] == "evaluative"
            )
            self.assertEqual(evaluative_stances, Counter({"positive": 72, "negative": 72}))

    def test_720_contrast_triads_share_raw_semantic_payload(self):
        by_triad: dict[str, list[dict]] = defaultdict(list)
        for row in self.rows:
            by_triad[row["contrast_triad_id"]].append(row)
        self.assertEqual(len(by_triad), 720)
        for triad_rows in by_triad.values():
            self.assertEqual(len(triad_rows), 3)
            self.assertEqual(
                {row["expected_feedback_type"] for row in triad_rows},
                set(generator.LABELS),
            )
            self.assertEqual(
                len({_canonical_json_sha256(row["semantic_payload"]) for row in triad_rows}),
                1,
            )

        across_banks: dict[tuple[str, str], set[str]] = defaultdict(set)
        for row in self.rows:
            across_banks[(row["scenario_id"], row["grounding_id"])].add(
                _canonical_json_sha256(row["semantic_payload"])
            )
        self.assertEqual(len(across_banks), 144)
        self.assertTrue(all(len(payloads) == 1 for payloads in across_banks.values()))

    def test_core_text_is_present_and_bank_specific(self):
        cores_by_bank = {}
        for bank in generator.BANKS:
            selected = [row for row in self.rows if row["bank_id"] == bank.bank_id]
            self.assertTrue(all(str(row["core_text"]).strip() for row in selected))
            cores_by_bank[bank.bank_id] = {
                generator.normalize_text(row["core_text"]) for row in selected
            }
        bank_ids = [bank.bank_id for bank in generator.BANKS]
        for left_index, left in enumerate(bank_ids):
            for right in bank_ids[left_index + 1 :]:
                self.assertFalse(
                    cores_by_bank[left] & cores_by_bank[right],
                    f"core text reused across {left} and {right}",
                )

    def test_text_and_core_obey_ontology_and_no_identifier_contract(self):
        identifier_pattern = re.compile(
            r"\b(?:bank|split|label|classifier|evaluative|imperative|descriptive|nonce)\b",
            re.IGNORECASE,
        )
        for row in self.rows:
            self.assertEqual(
                generator.speech_act_matches(row["text"]),
                (row["expected_feedback_type"],),
            )
            for field in ("text", "core_text"):
                value = str(row[field])
                self.assertNotRegex(value, identifier_pattern)
                for compiled in generator.FORBIDDEN_RE:
                    self.assertIsNone(compiled.search(value), value)
                self.assertFalse(
                    any(
                        unicodedata.category(character).startswith("C")
                        or (
                            unicodedata.category(character).startswith("Z")
                            and character != " "
                        )
                        for character in value
                    )
                )

    def test_full_and_core_similarity_gates_are_zero(self):
        self.assertTrue(self.report["quality_gate_passed"])
        self.assertEqual(self.report["status"], "passed_split_audit")
        self.assertTrue(all(self.report["checks"].values()))
        for report_key in ("cross_bank_similarity", "cross_partition_similarity"):
            for text_key in ("full_text", "core_text"):
                stats = self.report[report_key][text_key]
                self.assertEqual(stats["counts"], EXPECTED_SIMILARITY_COUNTS)
                self.assertLess(
                    stats["maximum_sequence_ratio"],
                    split_audit.SEQUENCE_THRESHOLD,
                )
                self.assertLess(
                    stats["maximum_token_jaccard"],
                    split_audit.JACCARD_THRESHOLD,
                )

    def test_masked_lexical_and_ngram_lookup_gates_are_binding(self):
        leakage = self.report["lexical_leakage"]
        masked = leakage["entity_masked_overlap"]
        self.assertEqual(masked["shared_4gram_count"], 0)
        self.assertEqual(masked["shared_5gram_count"], 0)
        self.assertEqual(masked["label_predictive_shared_3gram_count"], 0)
        self.assertLessEqual(masked["maximum_contiguous_token_span"], 3)
        self.assertEqual(
            leakage["raw_overlap"]["non_entity_shared_span_over_3_count"], 0
        )

        lookup = leakage["ngram_lookup_baselines"]
        self.assertEqual(set(lookup["views"]), set(LOOKUP_VIEWS))
        for view_name in LOOKUP_VIEWS:
            metrics = lookup["views"][view_name]["train_to_final_eval"]
            self.assertLessEqual(metrics["accuracy"], 0.40, view_name)
            self.assertLessEqual(metrics["macro_f1"], 0.40, view_name)
            self.assertIn("coverage", metrics)
            self.assertGreaterEqual(metrics["coverage"], 0.0)
            self.assertLessEqual(metrics["coverage"], 1.0)
            self.assertLessEqual(
                lookup["views"][view_name]["leave_one_bank_out_max_accuracy"],
                0.42,
                view_name,
            )

    def test_nuisance_features_are_chance_and_length_style_gates_pass(self):
        self.assertAlmostEqual(
            _majority_lookup_accuracy(self.rows, lambda row: _first_token(row["text"])),
            1.0 / 3.0,
        )
        self.assertAlmostEqual(
            _majority_lookup_accuracy(
                self.rows, lambda row: _terminal_punctuation(row["text"])
            ),
            1.0 / 3.0,
        )
        self.assertAlmostEqual(
            _majority_lookup_accuracy(self.rows, lambda row: _initial_case(row["text"])),
            1.0 / 3.0,
        )

        token_lengths = {
            label: [
                len(generator.normalize_text(row["text"]).split())
                for row in self.rows
                if row["expected_feedback_type"] == label
            ]
            for label in generator.LABELS
        }
        means = {
            label: sum(values) / len(values) for label, values in token_lengths.items()
        }
        self.assertLessEqual(max(means.values()) - min(means.values()), 1.0)
        pairwise_ks = []
        for left_index, left in enumerate(generator.LABELS):
            for right in generator.LABELS[left_index + 1 :]:
                pairwise_ks.append(_empirical_ks(token_lengths[left], token_lengths[right]))
        self.assertLessEqual(max(pairwise_ks), 0.10)

        nuisance = self.report["nuisance_baselines"]
        for key in ("first_token_lookup", "punctuation_only", "initial_case_only"):
            self.assertAlmostEqual(nuisance[key]["accuracy"], 1.0 / 3.0)
        self.assertLessEqual(nuisance["length_only"]["accuracy"], 0.40)
        self.assertLessEqual(nuisance["shallow_style_tree"]["accuracy"], 0.40)
        self.assertLessEqual(nuisance["label_token_length_mean_max_gap"], 1.0)
        self.assertLessEqual(nuisance["maximum_pairwise_token_length_ks"], 0.10)
        self.assertTrue(nuisance["included_in_quality_gate"])

    def test_precommitted_51_id_mutation_and_control_matrix(self):
        records: list[dict] = []
        first_triad = self.rows[0]["contrast_triad_id"]
        second_triad = next(
            row["contrast_triad_id"]
            for row in self.rows
            if row["contrast_triad_id"] != first_triad
        )

        # S01-S12: structural invariants.
        self._run_rejected_case(
            records,
            "S01",
            "delete_one_row",
            ["rows"],
            ("row_count_exact",),
            lambda rows: rows.pop(),
        )

        def s02(rows):
            rows[1]["feedback_id"] = rows[0]["feedback_id"]

        self._run_rejected_case(
            records,
            "S02",
            "duplicate_feedback_id",
            ["feedback_id"],
            ("feedback_id_unique",),
            s02,
        )

        def s03(rows):
            rows[0]["text"] = _append_before_terminal(rows[0]["text"], "for this turn")

        self._run_rejected_case(
            records,
            "S03",
            "raw_text_without_normalized_update",
            ["text", "normalized_text"],
            ("normalized_text_matches_recomputed",),
            s03,
        )

        def s04(rows):
            same_bank = [row for row in rows if row["bank_id"] == rows[0]["bank_id"]]
            same_bank[1]["text"] = same_bank[0]["text"]
            same_bank[1]["normalized_text"] = same_bank[0]["normalized_text"]

        self._run_rejected_case(
            records,
            "S04",
            "same_bank_normalized_duplicate",
            ["text", "normalized_text"],
            ("normalized_text_unique_global",),
            s04,
        )

        def s05_bank(rows):
            rows[0]["bank_id"] = "bank_unlisted"

        self._run_rejected_case(
            records,
            "S05",
            "undeclared_bank_id",
            ["bank_id"],
            ("bank_manifest_closed",),
            s05_bank,
        )

        def s05_split(rows):
            rows[0]["bank_split_role"] = "final_eval"

        self._run_rejected_case(
            records,
            "S05",
            "wrong_bank_split_role",
            ["bank_split_role"],
            ("bank_split_role_exact",),
            s05_split,
        )

        def s06(rows):
            source = next(row for row in rows if row["bank_split_role"] == "train")
            destination_bank = next(
                bank for bank in generator.BANKS if bank.split_role == "calibration"
            )
            source["bank_id"] = destination_bank.bank_id
            source["bank_split_role"] = destination_bank.split_role

        self._run_rejected_case(
            records,
            "S06",
            "move_row_between_bank_and_partition",
            ["bank_id", "bank_split_role"],
            (
                "bank_cardinality_and_label_balance_exact",
                "partition_cardinality_and_label_balance_exact",
            ),
            s06,
        )

        def s07_scenario(rows):
            bank_id = generator.BANKS[0].bank_id
            scenario_id = rows[0]["scenario_id"]
            for row in rows:
                if (
                    row["bank_id"] == bank_id
                    and row["scenario_id"] == scenario_id
                    and row["expected_feedback_type"] == "imperative"
                ):
                    row["expected_feedback_type"] = "evaluative"

        self._run_rejected_case(
            records,
            "S07",
            "relabel_one_bank_scenario",
            ["scenario_id", "expected_feedback_type"],
            ("scenario_x_label_support_exact",),
            s07_scenario,
        )

        def s07_action(rows):
            bank_id = generator.BANKS[0].bank_id
            action = rows[0]["semantic_payload"]["action"]
            for row in rows:
                if (
                    row["bank_id"] == bank_id
                    and row["semantic_payload"]["action"] == action
                    and row["expected_feedback_type"] == "descriptive"
                ):
                    row["expected_feedback_type"] = "imperative"

        self._run_rejected_case(
            records,
            "S07",
            "relabel_one_bank_action",
            ["semantic_payload.action", "expected_feedback_type"],
            ("action_x_label_support_exact",),
            s07_action,
        )

        def s08(rows):
            member = next(row for row in rows if row["contrast_triad_id"] == first_triad)
            member["contrast_triad_id"] = second_triad

        self._run_rejected_case(
            records,
            "S08",
            "triads_of_two_and_four",
            ["contrast_triad_id"],
            ("eid_triad_shape_exact",),
            s08,
        )

        coordinate_variants = {
            "bank": ("bank_id", generator.BANKS[1].bank_id),
            "split": ("bank_split_role", "final_eval"),
            "scenario": ("scenario_id", self.rows[-1]["scenario_id"]),
            "grounding": ("grounding_id", self.rows[-1]["grounding_id"]),
        }
        context_key = "scenario_family"
        coordinate_variants["context"] = (context_key, "a different valid context")
        for variant, (field, value) in coordinate_variants.items():
            def s09(rows, field=field, value=value):
                member = next(
                    row for row in rows if row["contrast_triad_id"] == first_triad
                )
                member[field] = value

            self._run_rejected_case(
                records,
                "S09",
                variant,
                [field],
                ("eid_triad_coordinates_identical",),
                s09,
            )

        def s10(rows):
            member = next(row for row in rows if row["contrast_triad_id"] == first_triad)
            replacement = next(
                action
                for action in generator.ALLOWED_ACTIONS
                if action != member["semantic_payload"]["action"]
            )
            member["semantic_payload"]["action"] = replacement

        self._run_rejected_case(
            records,
            "S10",
            "one_triad_member_payload",
            ["semantic_payload"],
            ("eid_triad_payload_identical",),
            s10,
        )

        def s11(rows):
            for row in rows:
                if row["contrast_triad_id"] == first_triad:
                    row["grounding_id"] = "clean-v6:grounding:missing-grid-member"

        self._run_rejected_case(
            records,
            "S11",
            "whole_triad_new_grounding",
            ["grounding_id"],
            ("cross_bank_payload_grid_exact_144x15",),
            s11,
        )

        def s12(rows):
            for row in rows:
                if row["contrast_triad_id"] == first_triad:
                    row["semantic_payload"]["result"] = "a bank-specific altered result"

        self._run_rejected_case(
            records,
            "S12",
            "whole_triad_bank_specific_payload",
            ["semantic_payload"],
            ("cross_bank_payload_identity",),
            s12,
        )

        # L01-L24: precommitted style, leakage, and semantic mutations.
        def l01(rows):
            prefixes = {
                "evaluative": "Honestly",
                "imperative": "Please",
                "descriptive": "Currently",
            }
            for row in rows:
                _set_text(row, f'{prefixes[row["expected_feedback_type"]]} {row["text"]}')

        self._run_rejected_case(
            records,
            "L01",
            "label_specific_first_token",
            ["text:first_token"],
            ("first_token_lookup_at_chance",),
            l01,
        )

        def l02(rows):
            second = {
                "evaluative": "quality",
                "imperative": "action",
                "descriptive": "status",
            }
            for row in rows:
                _set_text(row, f'Regarding {second[row["expected_feedback_type"]]} {row["text"]}')

        self._run_rejected_case(
            records,
            "L02",
            "label_specific_prefix_two",
            ["text:prefix_2"],
            ("prefix2_lookup_at_chance",),
            l02,
        )

        def l03(rows):
            punctuation = {"evaluative": ".", "imperative": "!", "descriptive": "?"}
            for row in rows:
                _set_text(
                    row,
                    row["text"].rstrip().rstrip(".!?")
                    + punctuation[row["expected_feedback_type"]],
                )

        self._run_rejected_case(
            records,
            "L03",
            "label_specific_terminal_punctuation",
            ["text:terminal_punctuation"],
            ("punctuation_lookup_at_chance",),
            l03,
        )

        def l04(rows):
            for row in rows:
                row_text = row["text"]
                transformed = {
                    "evaluative": row_text.upper(),
                    "imperative": row_text.lower(),
                    "descriptive": row_text.title(),
                }[row["expected_feedback_type"]]
                _set_text(row, transformed)

        self._run_rejected_case(
            records,
            "L04",
            "label_specific_case",
            ["text:case"],
            ("initial_case_lookup_at_chance",),
            l04,
        )

        def l05(rows):
            suffix = (
                "while every nearby station remains visible throughout the active "
                "cooperative kitchen round"
            )
            for row in rows:
                if row["expected_feedback_type"] == "descriptive":
                    _set_text(row, _append_before_terminal(row["text"], suffix))

        self._run_rejected_case(
            records,
            "L05",
            "descriptive_only_long_suffix",
            ["text:length"],
            ("length_style_below_precommitted_cap",),
            l05,
        )

        def l06(rows):
            left, right = _matching_cross_partition_pair(rows)
            _set_text(rows[right], rows[left]["text"])

        self._run_rejected_case(
            records,
            "L06",
            "cross_bank_exact",
            ["text", "normalized_text"],
            ("cross_bank_exact_zero",),
            l06,
        )

        def l07(rows):
            left, right = _matching_cross_partition_pair(rows)
            _set_core_text(rows[right], _sequence_only_variant(rows[left]["core_text"]))

        self._run_rejected_case(
            records,
            "L07",
            "cross_bank_sequence_only",
            ["core_text"],
            ("cross_bank_sequence_zero",),
            l07,
        )

        def l08(rows):
            left, right = _matching_cross_partition_pair(rows)
            _set_core_text(rows[right], _jaccard_only_variant(rows[left]["core_text"]))

        self._run_rejected_case(
            records,
            "L08",
            "cross_bank_jaccard_only",
            ["core_text"],
            ("cross_bank_jaccard_zero",),
            l08,
        )

        def l09(rows):
            train_core = {}
            for row in rows:
                key = (
                    row["scenario_id"],
                    row["grounding_id"],
                    row["expected_feedback_type"],
                )
                if row["bank_split_role"] == "train":
                    train_core.setdefault(key, row["core_text"])
            for row in rows:
                if row["bank_split_role"] == "final_eval":
                    key = (
                        row["scenario_id"],
                        row["grounding_id"],
                        row["expected_feedback_type"],
                    )
                    _set_core_text(row, train_core[key])

        self._run_rejected_case(
            records,
            "L09",
            "final_reuses_train_core",
            ["core_text", "bank_split_role"],
            ("cross_partition_core_exact_zero",),
            l09,
        )

        def l10(rows):
            left, right = _matching_cross_partition_pair(rows)
            _set_core_text(rows[right], f'was {rows[left]["core_text"]}')

        self._run_rejected_case(
            records,
            "L10",
            "core_tense_or_passive_only",
            ["core_text"],
            ("cross_partition_core_sequence_zero",),
            l10,
        )

        def l11(rows):
            suffixes = {
                "evaluative": "during the active round",
                "imperative": "within the current task",
                "descriptive": "across the visible kitchen",
            }
            for row in rows:
                _set_core_text(
                    row,
                    _append_before_terminal(
                        row["core_text"], suffixes[row["expected_feedback_type"]]
                    ),
                )

        self._run_rejected_case(
            records,
            "L11",
            "shared_same_label_four_token_suffix",
            ["core_text:masked_4gram"],
            ("cross_partition_masked_shared_4gram_zero",),
            l11,
        )

        def l12(rows):
            prefixes = {
                "evaluative": "after that move",
                "imperative": "for this task",
                "descriptive": "in this kitchen",
            }
            for row in rows:
                _set_core_text(
                    row,
                    f'{prefixes[row["expected_feedback_type"]]} {row["core_text"]}',
                )

        self._run_rejected_case(
            records,
            "L12",
            "predictive_repeated_three_grams",
            ["core_text:masked_3gram"],
            ("ngram_lookup_below_precommitted_cap",),
            l12,
        )

        def l13(rows):
            slots = {"evaluative": 0, "imperative": 1, "descriptive": 2}
            for row in rows:
                row["outer_frame_slot"] = slots[row["expected_feedback_type"]]

        self._run_rejected_case(
            records,
            "L13",
            "fixed_label_specific_frame_slot",
            ["outer_frame_slot"],
            ("outer_frame_slot_label_balanced",),
            l13,
        )

        def l14(rows):
            pickup = _row_for(rows, label="descriptive", scenario_fragment="get_tomato")
            yielded = _row_for(rows, label="descriptive", scenario_fragment="yield_")
            pickup["scenario_id"] = yielded["scenario_id"]
            pickup["scenario_family"] = yielded["scenario_family"]
            pickup["semantic_payload"] = deepcopy(yielded["semantic_payload"])

        self._run_rejected_case(
            records,
            "L14",
            "pickup_descriptive_receives_yield_payload",
            ["scenario_id", "semantic_payload", "expected_feedback_type"],
            ("scenario_x_label_support_exact",),
            l14,
        )

        for hidden in ("\u200b", "\u2060"):
            def l15(rows, hidden=hidden):
                _set_text(
                    rows[0],
                    rows[0]["text"][:2] + hidden + rows[0]["text"][2:],
                )

            self._run_rejected_case(
                records,
                "L15",
                f"U+{ord(hidden):04X}",
                ["text"],
                ("no_hidden_identifier_nonce_or_control_text",),
                l15,
            )

        def l16(rows):
            left, right = _matching_cross_partition_pair(rows)
            unicode_variant = rows[left]["text"].replace(" ", "\u00a0")
            self.assertNotEqual(unicode_variant, rows[left]["text"])
            self.assertEqual(
                generator.normalize_text(unicode_variant), rows[left]["normalized_text"]
            )
            _set_text(rows[right], unicode_variant)

        self._run_rejected_case(
            records,
            "L16",
            "unicode_normalization_exact_collision",
            ["text", "normalized_text"],
            ("normalized_text_unique_global",),
            l16,
        )

        def l17(rows):
            target = rows[0]
            poisoned_source = target["core_text"]
            for row in rows:
                if (
                    row["scenario_id"] == target["scenario_id"]
                    and row["grounding_id"] == target["grounding_id"]
                ):
                    row["semantic_payload"]["source"] = poisoned_source

        self._run_rejected_case(
            records,
            "L17",
            "payload_source_contains_core_text",
            ["semantic_payload.source", "entity_mask"],
            ("entity_mask_source_allowlist",),
            l17,
        )

        def l18(rows):
            _set_text(
                rows[0],
                _append_before_terminal(
                    rows[0]["text"], "bank five final set eval group"
                ),
            )

        self._run_rejected_case(
            records,
            "L18",
            "visible_split_identifiers",
            ["text"],
            ("no_hidden_identifier_nonce_or_control_text",),
            l18,
        )

        # L19 mutates the report schema, but still starts from a complete top-level audit.
        l19_report = split_audit.audit_split(
            deepcopy(self.rows), raise_on_failure=True
        )
        l19_pre = _canonical_json_sha256(l19_report)
        del l19_report["lexical_leakage"]["ngram_lookup_baselines"]["views"][
            "core_text_entity_masked"
        ]["train_to_final_eval"]["accuracy"]
        l19_post = _canonical_json_sha256(l19_report)
        with self.assertRaises(split_audit.AuditFailure) as l19_caught:
            split_audit.validate_report_schema(l19_report)
        self.assertIn("report_schema_complete", l19_caught.exception.failure_codes)
        records.append(
            {
                "mutation_id": "L19",
                "variant": "delete_masked_core_lookup_accuracy",
                "targets": [
                    "lexical_leakage.ngram_lookup_baselines.views."
                    "core_text_entity_masked.train_to_final_eval.accuracy"
                ],
                "target_rows": [],
                "target_row_count": 0,
                "target_rows_sha256": _canonical_json_sha256([]),
                "pre_sha256": l19_pre,
                "post_sha256": l19_post,
                "expected_failure_codes": ["report_schema_complete"],
                "observed_failure_codes": list(l19_caught.exception.failure_codes),
                "passed": True,
            }
        )

        def l20(rows):
            _set_core_text(rows[0], "")

        self._run_rejected_case(
            records,
            "L20",
            "empty_core_text",
            ["core_text"],
            ("core_text_required_nonempty",),
            l20,
        )

        base_triad_rows = {
            row["expected_feedback_type"]: row
            for row in self.rows
            if row["contrast_triad_id"] == first_triad
        }

        def l21(rows):
            target = next(
                row
                for row in rows
                if row["feedback_id"] == base_triad_rows["evaluative"]["feedback_id"]
            )
            _set_text(
                target,
                _combine_sentences(
                    base_triad_rows["evaluative"]["text"],
                    base_triad_rows["imperative"]["text"],
                ),
            )

        self._run_rejected_case(
            records,
            "L21",
            "mixed_evaluative_imperative",
            ["text", "speech_act_contract"],
            ("exactly_one_speech_act",),
            l21,
        )

        def l22(rows):
            target = next(
                row
                for row in rows
                if row["feedback_id"] == base_triad_rows["descriptive"]["feedback_id"]
            )
            _set_text(
                target,
                _combine_sentences(
                    base_triad_rows["descriptive"]["text"],
                    base_triad_rows["evaluative"]["text"],
                ),
            )

        self._run_rejected_case(
            records,
            "L22",
            "mixed_descriptive_evaluative",
            ["text", "speech_act_contract"],
            ("exactly_one_speech_act",),
            l22,
        )

        def l23(rows):
            target = _row_for(rows, label="evaluative")
            _set_text(target, "That result is good.")

        self._run_rejected_case(
            records,
            "L23",
            "evaluative_without_recent_completed_anchor",
            ["text", "speech_act_contract.temporal_frame"],
            ("direct_single_speech_act_matches_target",),
            l23,
        )

        def l24(rows):
            target = _row_for(rows, label="imperative")
            _set_text(target, "You picked up the tomato at the tomato dispenser.")

        self._run_rejected_case(
            records,
            "L24",
            "imperative_replaced_by_past_statement",
            ["text", "speech_act_contract.temporal_frame"],
            ("direct_single_speech_act_matches_target",),
            l24,
        )

        # R13-R24: independent reviewer mutations.
        def r13(rows):
            target = rows[0]
            replacement = next(
                action
                for action in generator.ALLOWED_ACTIONS
                if action != target["semantic_payload"]["action"]
            )
            for row in rows:
                if (
                    row["scenario_id"] == target["scenario_id"]
                    and row["grounding_id"] == target["grounding_id"]
                ):
                    row["semantic_payload"]["action"] = replacement

        self._run_rejected_case(
            records,
            "R13",
            "legal_but_wrong_action_for_frozen_scenario",
            ["semantic_payload.action", "frozen_ontology"],
            ("payload_equals_frozen_ontology",),
            r13,
        )

        def r14(rows):
            triad = {
                row["expected_feedback_type"]: row
                for row in rows
                if row["contrast_triad_id"] == first_triad
            }
            _set_text(triad["evaluative"], triad["imperative"]["text"])
            _set_core_text(triad["evaluative"], triad["imperative"]["core_text"])

        self._run_rejected_case(
            records,
            "R14",
            "evaluative_row_uses_imperative_surface",
            ["text", "core_text", "expected_feedback_type"],
            ("direct_single_speech_act_matches_target",),
            r14,
        )

        def r15(rows):
            triad = {
                row["expected_feedback_type"]: row
                for row in rows
                if row["contrast_triad_id"] == first_triad
            }
            _set_text(
                triad["descriptive"],
                _combine_sentences(
                    triad["descriptive"]["text"], triad["imperative"]["text"]
                ),
            )

        self._run_rejected_case(
            records,
            "R15",
            "descriptive_with_imperative_clause",
            ["text", "speech_act_contract"],
            ("exactly_one_speech_act",),
            r15,
        )

        def r16_reference(rows):
            rows[0]["reference_type"] = "action"

        self._run_rejected_case(
            records,
            "R16",
            "legacy_reference_type",
            ["reference_type"],
            ("current_three_class_schema_only",),
            r16_reference,
        )

        def r16_ontology(rows):
            _set_text(rows[0], _append_before_terminal(rows[0]["text"], "chop lettuce"))

        self._run_rejected_case(
            records,
            "R16",
            "forbidden_old_ontology_terms",
            ["text"],
            ("forbidden_legacy_ontology_zero",),
            r16_ontology,
        )

        identifier_variants = {
            "bank_id": "bank five",
            "nonce": "nonce marker",
            "zero_width": "hidden\u200bmarker",
            "control": "hidden\u0007marker",
        }
        for variant, insertion in identifier_variants.items():
            def r17(rows, insertion=insertion):
                _set_text(rows[0], _append_before_terminal(rows[0]["text"], insertion))

            self._run_rejected_case(
                records,
                "R17",
                variant,
                ["text"],
                ("no_hidden_identifier_nonce_or_control_text",),
                r17,
            )

        def r18(rows):
            left, right = _matching_cross_partition_pair(rows)
            _set_text(rows[right], rows[left]["text"])
            _set_core_text(rows[right], rows[left]["core_text"])

        self._run_rejected_case(
            records,
            "R18",
            "copy_train_surface_into_final",
            ["text", "core_text", "bank_split_role"],
            ("cross_partition_exact_zero", "components_partition_exclusive"),
            r18,
        )

        def r19(rows):
            left, right = _matching_cross_partition_pair(rows)
            _set_core_text(rows[right], _sequence_only_variant(rows[left]["core_text"]))

        self._run_rejected_case(
            records,
            "R19",
            "cross_partition_sequence_only_fixture",
            ["core_text"],
            ("cross_partition_sequence_zero",),
            r19,
        )

        def r20(rows):
            left, right = _matching_cross_partition_pair(rows)
            _set_core_text(rows[right], _jaccard_only_variant(rows[left]["core_text"]))

        self._run_rejected_case(
            records,
            "R20",
            "cross_partition_jaccard_only_fixture",
            ["core_text"],
            ("cross_partition_jaccard_zero",),
            r20,
        )

        def r21(rows):
            left, right = _matching_cross_partition_pair(rows)
            shared = (
                "the same coordinated action phrase remains active throughout this kitchen round"
            )
            self.assertGreaterEqual(len(generator.normalize_text(shared).split()), 8)
            _set_core_text(rows[left], _append_before_terminal(rows[left]["core_text"], shared))
            _set_core_text(rows[right], _append_before_terminal(rows[right]["core_text"], shared))

        self._run_rejected_case(
            records,
            "R21",
            "same_label_shared_eight_plus_token_core",
            ["core_text:masked_ngram", "core_text:lcs"],
            ("cross_partition_label_core_ngram_zero",),
            r21,
        )

        def r22(rows):
            prefixes = {
                "evaluative": "Honestly",
                "imperative": "Please",
                "descriptive": "Currently",
            }
            for row in rows:
                _set_text(row, f'{prefixes[row["expected_feedback_type"]]} {row["text"]}')

        self._run_rejected_case(
            records,
            "R22",
            "all_banks_label_first_token_probe",
            ["text:first_token", "nuisance_style_probe"],
            ("nuisance_style_probe_below_precommitted_cap",),
            r22,
        )

        original_model_input = generator.extract_model_input(self.rows[0]["text"])
        self.assertEqual(original_model_input, self.rows[0]["text"])
        with self.assertRaises(TypeError):
            generator.extract_model_input(self.rows[0])

        def r23(rows):
            rows[0]["classification_label"] = "Descriptive"
            rows[0]["leaked_label"] = rows[0]["expected_feedback_type"]
            self.assertEqual(
                generator.extract_model_input(rows[0]["text"]), original_model_input
            )

        self._run_rejected_case(
            records,
            "R23",
            "non_text_label_metadata_leak",
            ["classification_label", "leaked_label", "model_input"],
            ("model_feature_allowlist_raw_text_only",),
            r23,
        )

        def run_r24_variant(variant: str, mutate) -> None:
            # Every R24 variant starts with the same top-level audit as row mutations.
            report = split_audit.audit_split(
                deepcopy(self.rows), raise_on_failure=True
            )
            with tempfile.TemporaryDirectory(prefix="direct-fg-v6-r24-") as temp:
                corpus_path = Path(temp) / "corpus.json"
                corpus_path.write_text(
                    json.dumps(
                        self.rows,
                        ensure_ascii=False,
                        indent=2,
                        sort_keys=True,
                    )
                    + "\n",
                    encoding="utf-8",
                )
                report["output_artifacts"] = {
                    "corpus": {
                        "path": str(corpus_path.resolve()),
                        "sha256": _file_sha256(corpus_path),
                    },
                    "model_written": False,
                }
                clean = split_audit.validate_artifact_integrity(
                    report, corpus_path=corpus_path
                )
                self.assertFalse(clean["failure_codes"], clean)
                pre_hash = _canonical_json_sha256(
                    {"report": report, "corpus_sha256": _file_sha256(corpus_path)}
                )
                mutate(report, corpus_path)
                post_hash = _canonical_json_sha256(
                    {"report": report, "corpus_sha256": _file_sha256(corpus_path)}
                )
                self.assertNotEqual(pre_hash, post_hash)
                observed_report = split_audit.validate_artifact_integrity(
                    report, corpus_path=corpus_path
                )
                observed = tuple(observed_report["failure_codes"])
                expected = (
                    "no_external_eval_data_reads"
                    if variant == "forged_external_read"
                    else "artifact_and_source_hashes_resolve"
                )
                self.assertIn(expected, observed, f"R24/{variant}: {observed}")
                records.append(
                    {
                        "mutation_id": "R24",
                        "variant": variant,
                        "targets": ["corpus", "script_hashes", "provenance"],
                        "target_rows": [],
                        "target_row_count": 0,
                        "target_rows_sha256": _canonical_json_sha256([]),
                        "pre_sha256": pre_hash,
                        "post_sha256": post_hash,
                        "expected_failure_codes": [expected],
                        "observed_failure_codes": list(observed),
                        "passed": True,
                    }
                )

        def r24_corpus(_report, corpus_path):
            corpus_path.write_bytes(corpus_path.read_bytes() + b" ")

        run_r24_variant("tampered_corpus_byte", r24_corpus)

        def r24_generator(report, _corpus_path):
            report["script_hashes"]["generator"]["sha256"] = "0" * 64

        run_r24_variant("tampered_generator_hash", r24_generator)

        def r24_dependency(report, _corpus_path):
            self.assertTrue(report["script_hashes"]["dependencies"])
            report["script_hashes"]["dependencies"][0]["sha256"] = "0" * 64

        run_r24_variant("tampered_dependency_hash", r24_dependency)

        def r24_external(report, _corpus_path):
            report["generation"]["external_input_files_opened"] = 1
            report["provenance"]["old_human_dev_test_frozen_rows_read"] = 1

        run_r24_variant("forged_external_read", r24_external)

        # C01-C03: positive controls each execute the complete top-level audit.
        self.assertTrue(
            self.report["lexical_leakage"]["raw_overlap"][
                "canonical_entity_only_control_passed"
            ]
        )
        self._run_control_case(
            records,
            "C01",
            ["canonical_entity_overlap", "raw_overlap_allowlist"],
        )

        right_now_counts = Counter(
            row["expected_feedback_type"]
            for row in self.rows
            if re.search(r"\bright now\b", row["text"], re.IGNORECASE)
        )
        self.assertGreater(sum(right_now_counts.values()), 0)
        self.assertEqual(set(right_now_counts), set(generator.LABELS))
        self.assertEqual(len(set(right_now_counts.values())), 1)
        self._run_control_case(
            records,
            "C02",
            ["balanced_shared_bigram:right now"],
        )

        for bank in generator.BANKS:
            slots_by_label = {}
            for label in generator.LABELS:
                slots_by_label[label] = Counter(
                    row["outer_frame_slot"]
                    for row in self.rows
                    if row["bank_id"] == bank.bank_id
                    and row["expected_feedback_type"] == label
                    and _terminal_punctuation(row["text"]) == "<none>"
                )
            first = slots_by_label[generator.LABELS[0]]
            self.assertEqual(sum(first.values()), 24)
            for label in generator.LABELS[1:]:
                self.assertEqual(slots_by_label[label], first)
        self._run_control_case(
            records,
            "C03",
            ["balanced_no_terminal_punctuation_slots"],
        )

        expected_ids = (
            {f"S{index:02d}" for index in range(1, 13)}
            | {f"L{index:02d}" for index in range(1, 25)}
            | {f"R{index:02d}" for index in range(13, 25)}
            | {f"C{index:02d}" for index in range(1, 4)}
        )
        self.assertEqual(len(expected_ids), 51)
        observed_ids = {record["mutation_id"] for record in records}
        self.assertEqual(observed_ids, expected_ids)
        for record in records:
            self.assertTrue(record["targets"])
            self.assertEqual(record["target_row_count"], len(record["target_rows"]))
            self.assertEqual(
                record["target_rows_sha256"],
                _canonical_json_sha256(record["target_rows"]),
            )
            self.assertRegex(record["pre_sha256"], r"^[0-9a-f]{64}$")
            self.assertRegex(record["post_sha256"], r"^[0-9a-f]{64}$")
            self.assertTrue(record["passed"])
            if not record["mutation_id"].startswith("C"):
                self.assertTrue(record["expected_failure_codes"])
                self.assertTrue(record["observed_failure_codes"])
        self.__class__.mutation_records = records
        self.__class__.mutation_matrix_sha256 = _canonical_json_sha256(records)

    def test_cli_artifacts_reload_and_hash_without_writing_a_model(self):
        with tempfile.TemporaryDirectory(prefix="direct-fg-clean-v6-test-") as temp:
            output_dir = Path(temp)
            completed = subprocess.run(
                [
                    sys.executable,
                    str(Path(split_audit.__file__).resolve()),
                    "--output-dir",
                    str(output_dir),
                ],
                cwd=ROOT,
                check=False,
                capture_output=True,
                text=True,
                timeout=600,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            corpus_path = output_dir / "corpus.json"
            report_path = output_dir / "split_audit.json"
            self.assertTrue(corpus_path.is_file())
            self.assertTrue(report_path.is_file())

            corpus = json.loads(corpus_path.read_text(encoding="utf-8"))
            report = json.loads(report_path.read_text(encoding="utf-8"))
            self.assertEqual(corpus, self.rows)
            self.assertTrue(report["quality_gate_passed"])
            self.assertFalse(report["output_artifacts"]["model_written"])
            self.assertEqual(
                report["output_artifacts"]["corpus"]["sha256"],
                _file_sha256(corpus_path),
            )

            report_hash = report["output_artifacts"]["split_audit"]["sha256"]
            report_without_self_hash = deepcopy(report)
            del report_without_self_hash["output_artifacts"]["split_audit"]["sha256"]
            self.assertEqual(
                report["output_artifacts"]["split_audit"]["hash_scope"],
                "canonical_report_without_self_hash",
            )
            self.assertEqual(report_hash, _canonical_json_sha256(report_without_self_hash))
            generator_source = report["script_hashes"]["generator"]
            audit_source = report["script_hashes"]["audit"]
            self.assertEqual(Path(generator_source["path"]), Path(generator.__file__).resolve())
            self.assertEqual(
                generator_source["sha256"], _file_sha256(Path(generator.__file__))
            )
            self.assertEqual(Path(audit_source["path"]), Path(split_audit.__file__).resolve())
            self.assertEqual(
                audit_source["sha256"], _file_sha256(Path(split_audit.__file__))
            )
            self.assertTrue(report["script_hashes"]["dependencies"])
            for dependency in report["script_hashes"]["dependencies"]:
                dependency_path = Path(dependency["path"])
                self.assertTrue(dependency_path.is_file())
                self.assertEqual(dependency["sha256"], _file_sha256(dependency_path))
            integrity = split_audit.validate_artifact_integrity(
                report, corpus_path=corpus_path
            )
            self.assertFalse(integrity["failure_codes"], integrity)

            written = [path for path in output_dir.rglob("*") if path.is_file()]
            self.assertEqual({path.name for path in written}, {"corpus.json", "split_audit.json"})
            self.assertFalse(any(path.suffix.casefold() in MODEL_SUFFIXES for path in written))
            self.assertFalse(report["provenance"]["training_performed"])
            self.assertFalse(report["provenance"]["production_promotion_eligible"])


if __name__ == "__main__":
    unittest.main()
