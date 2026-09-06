from __future__ import annotations

import ast
from collections import Counter, defaultdict
import json
from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts import generate_direct_fg_clean_v4 as generator
from scripts import train_direct_fg_classifier_clean_v4 as trainer


class CleanV4ContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rows = generator.generate_rows()
        cls.audit = generator.audit_rows(cls.rows)
        cls.partitions, cls.split_audit = trainer.base.split_rows(cls.rows)

    def test_generation_is_2160_balanced_unique_rows(self):
        self.assertEqual(len(self.rows), 2160)
        self.assertEqual(
            Counter(row["expected_feedback_type"] for row in self.rows),
            Counter({"evaluative": 720, "imperative": 720, "descriptive": 720}),
        )
        self.assertEqual(len({row["normalized_text"] for row in self.rows}), 2160)
        self.assertEqual(self.audit["short_contrast_rows"], 720)
        self.assertEqual(
            self.audit["short_contrast_label_counts"],
            {"descriptive": 240, "evaluative": 240, "imperative": 240},
        )
        self.assertEqual(
            self.audit["surface_family_counts"],
            {"descriptive": 72, "evaluative": 72, "imperative": 72},
        )

    def test_each_typed_grounding_has_one_short_three_way_contrast(self):
        grouped = defaultdict(list)
        for row in self.rows:
            if row["generation_provenance"]["v4_short_contrast_extension"]:
                grouped[row["grounding_id"]].append(row)
        self.assertEqual(len(grouped), 240)
        for rows in grouped.values():
            self.assertEqual(len(rows), 3)
            self.assertEqual(
                {row["expected_feedback_type"] for row in rows}, set(generator.LABELS)
            )
            self.assertEqual(len({row["scenario_id"] for row in rows}), 1)

    def test_short_frames_are_single_act_grammatical_and_current_ontology(self):
        short_rows = [
            row
            for row in self.rows
            if row["generation_provenance"]["v4_short_contrast_extension"]
        ]
        for row in short_rows:
            label = row["expected_feedback_type"]
            text = row["text"]
            self.assertEqual(generator.speech_act_matches(text), (label,))
            self.assertEqual(len(__import__("re").findall(r"[.!?]", text)), 1)
            self.assertNotIn("{", text)
            self.assertNotIn("}", text)
            self.assertIsNone(generator.v3.BAD_GERUND_CONSTRUCTION_RE.search(text))
            for compiled in generator.v3.FORBIDDEN_RE:
                self.assertIsNone(compiled.search(text))
        self.assertEqual(self.audit["single_speech_act_failures"], 0)
        self.assertEqual(self.audit["surface_grammar_failures"], 0)
        self.assertEqual(self.audit["forbidden_ontology_hits"], 0)

    def test_v4_reads_no_old_or_human_data_and_uses_no_reference_target(self):
        self.assertEqual(self.audit["external_input_files_opened"], 0)
        self.assertEqual(self.audit["old_or_human_data_rows_read"], 0)
        self.assertFalse(self.audit["reference_type_used"])
        self.assertTrue(self.audit["v3_code_reused_without_v3_artifact_reads"])
        self.assertTrue(all("reference_type" not in row for row in self.rows))
        generator_source = Path(generator.__file__).read_text(encoding="utf-8")
        self.assertNotIn("human_feedback_form", generator_source)
        self.assertNotIn("reference_classifier_feedback", generator_source)
        self.assertNotIn("model.joblib", generator_source)
        self.assertNotIn("read_text(", generator_source)
        self.assertNotIn("read_bytes(", generator_source)
        self.assertNotIn("json.load(", generator_source)
        self.assertNotIn("open(", generator_source)

    def test_strict_or_split_is_balanced_and_pairwise_clean(self):
        self.assertTrue(self.split_audit["similarity_quality_gate_passed"])
        self.assertEqual(
            self.split_audit["cross_partition_similarity_counts"],
            {
                "normalized_exact": 0,
                "sequence_matcher_at_least_0_88": 0,
                "token_jaccard_at_least_0_70": 0,
            },
        )
        self.assertFalse(
            self.split_audit["component_graph"]["length_prefilter_used"]
        )
        self.assertEqual(
            self.split_audit["component_graph"]["edge_rule"],
            "sequence_matcher_gte_0.88_OR_token_jaccard_gte_0.70",
        )
        for overlap in self.split_audit["pairwise_group_overlap"].values():
            self.assertTrue(all(value == 0 for value in overlap.values()))
        for rows in self.partitions.values():
            counts = Counter(row["expected_feedback_type"] for row in rows)
            self.assertEqual(set(counts), set(generator.LABELS))
            self.assertEqual(len(set(counts.values())), 1)

    def test_old_and_new_diagnostics_are_non_gate_and_disjoint(self):
        old = trainer.V3_TRAINING_INFORMED_REGRESSION
        new = trainer.V4_UNSEEN_SHORT_DIAGNOSTIC
        self.assertEqual(Counter(label for _, label in old), Counter({label: 10 for label in generator.LABELS}))
        self.assertEqual(Counter(label for _, label in new), Counter({label: 10 for label in generator.LABELS}))
        old_texts = {generator.v3.normalize_text(text) for text, _ in old}
        new_texts = {generator.v3.normalize_text(text) for text, _ in new}
        train_texts = {row["normalized_text"] for row in self.rows}
        self.assertFalse(old_texts & new_texts)
        self.assertFalse(new_texts & train_texts)
        self.assertEqual(
            trainer.V4_UNSEEN_SHORT_DIAGNOSTIC_VERSION,
            "clean-v4-unseen-short-diagnostic-v1",
        )

    def test_training_path_contains_exactly_one_final_eval_call(self):
        source = Path(trainer.__file__).read_text(encoding="utf-8")
        tree = ast.parse(source)
        function = next(
            node
            for node in tree.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name == "train_candidate_one_final_look"
        )
        calls = [
            node
            for node in ast.walk(function)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "base"
            and node.func.attr == "evaluate_frozen_final"
        ]
        self.assertEqual(len(calls), 1)

    def test_dependency_manifest_binds_all_reused_local_code(self):
        manifest = trainer._dependency_manifest()
        self.assertEqual(
            set(manifest["local_code"]),
            {
                "v4_generator",
                "v4_trainer",
                "reused_v3_typed_generator",
                "reused_v3_training_algorithms",
            },
        )
        for entry in manifest["local_code"].values():
            self.assertEqual(trainer._sha256(Path(entry["path"])), entry["sha256"])


if __name__ == "__main__":
    unittest.main()
