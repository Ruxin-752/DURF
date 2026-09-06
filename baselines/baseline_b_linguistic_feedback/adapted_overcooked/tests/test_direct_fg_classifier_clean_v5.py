from __future__ import annotations

from collections import Counter, defaultdict
import json
from pathlib import Path
import re
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts import audit_direct_fg_clean_v5_split as split_audit
from scripts import generate_direct_fg_clean_v5 as generator


class CleanV5GenerationAndSplitTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rows = generator.generate_rows()
        cls.generation = generator.audit_rows(cls.rows)
        cls.report = split_audit.audit_split(cls.rows)

    def test_generation_is_balanced_unique_and_raw_only(self):
        self.assertEqual(len(self.rows), 2160)
        self.assertEqual(
            Counter(row["expected_feedback_type"] for row in self.rows),
            Counter({"evaluative": 720, "imperative": 720, "descriptive": 720}),
        )
        self.assertEqual(len({row["normalized_text"] for row in self.rows}), 2160)
        self.assertEqual(self.generation["external_input_files_opened"], 0)
        self.assertEqual(self.generation["old_human_dev_test_frozen_rows_read"], 0)
        self.assertFalse(self.generation["reference_type_used"])
        source = Path(generator.__file__).read_text(encoding="utf-8")
        self.assertNotIn("read_text(", source)
        self.assertNotIn("read_bytes(", source)
        self.assertNotIn("json.load(", source)
        self.assertNotIn("human_feedback_form", source)
        self.assertNotIn("reference_classifier_feedback", source)

    def test_each_bank_has_full_balanced_ontology_support(self):
        self.assertEqual(len(generator.BANKS), 5)
        for bank in generator.BANKS:
            selected = [row for row in self.rows if row["bank_id"] == bank.bank_id]
            self.assertEqual(len(selected), 432)
            self.assertEqual(
                Counter(row["expected_feedback_type"] for row in selected),
                Counter({label: 144 for label in generator.LABELS}),
            )
            self.assertEqual(len({row["scenario_id"] for row in selected}), 24)
            self.assertEqual(
                {row["semantic_payload"]["action"] for row in selected},
                set(generator.ontology.ALLOWED_ACTIONS),
            )
            action_label = {
                (row["semantic_payload"]["action"], row["expected_feedback_type"])
                for row in selected
            }
            self.assertEqual(len(action_label), 30)

    def test_contrast_triads_share_exact_semantic_payload(self):
        by_triad = defaultdict(list)
        for row in self.rows:
            by_triad[row["contrast_triad_id"]].append(row)
        self.assertEqual(len(by_triad), 720)
        for triad_rows in by_triad.values():
            self.assertEqual(len(triad_rows), 3)
            self.assertEqual(
                {row["expected_feedback_type"] for row in triad_rows},
                set(generator.LABELS),
            )
            payloads = {
                json.dumps(row["semantic_payload"], sort_keys=True)
                for row in triad_rows
            }
            self.assertEqual(len(payloads), 1)

        cross_bank_payloads = defaultdict(set)
        for row in self.rows:
            key = (row["scenario_id"], row["grounding_id"])
            cross_bank_payloads[key].add(
                json.dumps(row["semantic_payload"], sort_keys=True)
            )
        self.assertTrue(all(len(payloads) == 1 for payloads in cross_bank_payloads.values()))

    def test_text_contract_has_no_identifiers_nonces_or_invisible_characters(self):
        bank_ids = {bank.bank_id.casefold() for bank in generator.BANKS}
        for row in self.rows:
            self.assertEqual(
                generator.speech_act_matches(row["text"]),
                (row["expected_feedback_type"],),
            )
            self.assertFalse(set(row["normalized_text"].split()) & bank_ids)
            self.assertFalse(
                any(
                    __import__("unicodedata").category(character).startswith("C")
                    for character in row["text"]
                )
            )
            self.assertNotRegex(row["text"], r"\b(?:nonce|bank[_ -]?0?[1-5])\b")

    def test_strict_bank_and_partition_similarity_gate(self):
        self.assertTrue(self.report["quality_gate_passed"])
        self.assertEqual(self.report["status"], "passed_split_audit")
        self.assertEqual(
            self.report["cross_bank_similarity"]["counts"],
            {"exact": 0, "sequence": 0, "jaccard": 0, "either": 0},
        )
        self.assertEqual(
            self.report["cross_partition_similarity"]["counts"],
            {"exact": 0, "sequence": 0, "jaccard": 0, "either": 0},
        )
        self.assertLess(
            self.report["cross_bank_similarity"]["maximum_sequence_ratio"],
            split_audit.SEQUENCE_THRESHOLD,
        )
        self.assertLess(
            self.report["cross_bank_similarity"]["maximum_token_jaccard"],
            split_audit.JACCARD_THRESHOLD,
        )
        graph = self.report["component_graph"]
        self.assertEqual(graph["component_count"], 5)
        self.assertEqual(graph["component_sizes_desc"], [432] * 5)
        self.assertFalse(graph["scenario_grounding_action_edges_used"])
        self.assertFalse(graph["length_prefilter_used"])

    def test_split_scope_metadata_baseline_hashes_and_examples_are_explicit(self):
        self.assertEqual(
            self.generation["metadata_only_empirical_majority_accuracy"], 1.0 / 3.0
        )
        self.assertTrue(self.generation["metadata_only_baseline_is_chance"])
        for details in self.report["partition_contract"].values():
            self.assertTrue(details["class_balanced"])
            self.assertTrue(details["all_24_scenarios_present"])
            self.assertTrue(details["all_10_actions_present"])
            self.assertTrue(details["all_30_action_x_label_cells_present"])
        self.assertEqual(
            self.report["partition_contract"]["train"]["rows"], 1296
        )
        self.assertEqual(
            self.report["partition_contract"]["calibration"]["rows"], 432
        )
        self.assertEqual(
            self.report["partition_contract"]["final_eval"]["rows"], 432
        )
        self.assertTrue(
            self.report["split_protocol"]["scenario_content_intentionally_shared"]
        )
        self.assertEqual(
            self.report["split_protocol"]["benchmark_scope"],
            "surface_heldout_synthetic_only",
        )
        self.assertFalse(self.report["provenance"]["training_performed"])
        self.assertFalse(self.report["provenance"]["production_promotion_eligible"])
        self.assertFalse(self.report["provenance"]["independent_human_semantic_audit"])
        self.assertEqual(len(self.report["examples"]["by_bank"]), 5)
        for pairs_key in ("top_sequence_pair_hashes", "top_jaccard_pair_hashes"):
            pairs = self.report["cross_bank_similarity"][pairs_key]
            self.assertEqual(len(pairs), 5)
            for pair in pairs:
                self.assertRegex(pair["left_text_sha256"], r"^[0-9a-f]{64}$")
                self.assertRegex(pair["right_text_sha256"], r"^[0-9a-f]{64}$")
        self.assertEqual(
            self.report["v4_stress_reference"]["report_sha256"],
            "a4669257f62bb144ce00c9cc416056fa247e990b21fee412001e0c4c91051d49",
        )
        self.assertEqual(self.report["v4_stress_reference"]["imperative_recall"], 0.74)

    def test_similarity_oracle_mutations_cover_exact_seq_only_and_jaccard_only(self):
        exact_text = "the pot is cooking two tomatoes and one onion"
        self.assertEqual(
            split_audit.similarity_flags(exact_text, exact_text),
            (True, True, True),
        )

        seq_left = "the tomato is beside the onion at the counter"
        seq_right = "the tomatoes are beside the onions at the counters"
        seq_flags = split_audit.similarity_flags(seq_left, seq_right)
        self.assertEqual(seq_flags, (False, True, False))
        self.assertGreaterEqual(
            split_audit.sequence_ratio(seq_left, seq_right),
            split_audit.SEQUENCE_THRESHOLD,
        )
        self.assertLess(
            split_audit.token_jaccard(seq_left, seq_right),
            split_audit.JACCARD_THRESHOLD,
        )

        jac_left = "tomato onion dish soup pot counter station chef"
        jac_right = "chef station counter pot soup dish onion tomato"
        jac_flags = split_audit.similarity_flags(jac_left, jac_right)
        self.assertEqual(jac_flags, (False, False, True))
        self.assertLess(
            split_audit.sequence_ratio(jac_left, jac_right),
            split_audit.SEQUENCE_THRESHOLD,
        )
        self.assertGreaterEqual(
            split_audit.token_jaccard(jac_left, jac_right),
            split_audit.JACCARD_THRESHOLD,
        )


if __name__ == "__main__":
    unittest.main()
