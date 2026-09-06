from __future__ import annotations

import builtins
from collections import Counter
from dataclasses import replace
from difflib import SequenceMatcher
import io
import json
from pathlib import Path
import random
import sys
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts import generate_direct_fg_clean_v3 as generator
from scripts import train_direct_fg_classifier_clean_v3 as trainer


class CleanV3GenerationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rows = generator.generate_rows()
        cls.audit = generator.audit_rows(cls.rows)

    def test_generation_contract_is_balanced_unique_and_direct(self):
        self.assertEqual(len(self.rows), 1440)
        self.assertEqual(
            Counter(row["expected_feedback_type"] for row in self.rows),
            Counter({"evaluative": 480, "imperative": 480, "descriptive": 480}),
        )
        self.assertEqual(len({row["normalized_text"] for row in self.rows}), 1440)
        self.assertEqual(len({row["scenario_id"] for row in self.rows}), 24)
        for label in generator.LABELS:
            self.assertEqual(
                len(
                    {
                        row["surface_family"]
                        for row in self.rows
                        if row["expected_feedback_type"] == label
                    }
                ),
                48,
            )
        for row in self.rows:
            label = row["expected_feedback_type"]
            self.assertEqual(row["classification_label"], generator.CANONICAL_LABELS[label])
            self.assertEqual(row["label_source"], generator.LABEL_SOURCE)
            self.assertNotIn("reference_type", row)
            self.assertEqual(generator.speech_act_matches(row["text"]), (label,))
        self.assertFalse(self.audit["reference_type_used"])
        self.assertEqual(self.audit["external_input_files_opened"], 0)

    def test_generation_is_order_and_byte_deterministic(self):
        regenerated = generator.generate_rows()
        first = json.dumps(self.rows, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        second = json.dumps(regenerated, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        self.assertEqual(first, second)

    def test_typed_scenarios_and_surface_grammar_fail_closed(self):
        invalid = replace(generator.SCENARIOS[0], canonical_action="CHOP_TOMATO")
        with mock.patch.object(generator, "SCENARIOS", (invalid, *generator.SCENARIOS[1:])):
            with self.assertRaisesRegex(ValueError, "disallowed action"):
                generator.validate_scenarios()
        broken = [dict(row) for row in self.rows]
        broken[0]["text"] = broken[0]["text"] + " {unfinished}"
        broken[0]["normalized_text"] = generator.normalize_text(broken[0]["text"])
        with self.assertRaisesRegex(ValueError, "surface grammar failures"):
            generator.audit_rows(broken)
        for row in self.rows:
            self.assertIsNone(generator.BAD_GERUND_CONSTRUCTION_RE.search(row["text"]))
            self.assertEqual(len(__import__("re").findall(r"[.!?]", row["text"])), 1)

    def test_no_legacy_generator_import_or_data_input_path(self):
        sources = {
            "generator": Path(generator.__file__).read_text(encoding="utf-8"),
            "trainer": Path(trainer.__file__).read_text(encoding="utf-8"),
        }
        forbidden = (
            "train_direct_fg_classifier_candidate_v2",
            "run_direct_fg_v3_ablation",
            "generate_feedback_form_hard_train",
            "generate_feedback_form_boundary_train",
            "human_feedback_form",
            "reference_classifier_feedback",
        )
        for name, source in sources.items():
            for token in forbidden:
                self.assertNotIn(token, source, f"{name} imports/references {token}")


class CleanV3SimilarityTests(unittest.TestCase):
    def test_oracle_has_no_hidden_length_or_and_gate(self):
        seq_left = "abcdefghijklmnopqrst"
        seq_right = "abcdefghijklmnopqrsx"
        sequence = SequenceMatcher(None, seq_left, seq_right, autojunk=False).ratio()
        jaccard = trainer._token_jaccard(set(seq_left.split()), set(seq_right.split()))
        self.assertGreaterEqual(sequence, trainer.SEQUENCE_THRESHOLD)
        self.assertLess(jaccard, trainer.JACCARD_THRESHOLD)
        self.assertGreaterEqual(trainer._sequence_ratio(seq_left, seq_right), trainer.SEQUENCE_THRESHOLD)

        jac_left = "alpha bravo charlie delta echo foxtrot"
        jac_right = "foxtrot delta bravo echo alpha charlie"
        sequence = SequenceMatcher(None, jac_left, jac_right, autojunk=False).ratio()
        jaccard = trainer._token_jaccard(set(jac_left.split()), set(jac_right.split()))
        self.assertLess(sequence, trainer.SEQUENCE_THRESHOLD)
        self.assertGreaterEqual(jaccard, trainer.JACCARD_THRESHOLD)

        short = "alpha beta"
        long = "alpha alpha alpha alpha beta"
        length_ratio = len(short) / len(long)
        self.assertLess(length_ratio, trainer.SEQUENCE_THRESHOLD)
        self.assertGreaterEqual(
            trainer._token_jaccard(set(short.split()), set(long.split())),
            trainer.JACCARD_THRESHOLD,
        )

    def test_transitive_or_edges_form_one_component(self):
        texts = (
            "alpha bravo charlie delta",
            "echo delta bravo alpha charlie",
            "echo bravo charlie delta",
        )
        self.assertGreaterEqual(
            trainer._token_jaccard(set(texts[0].split()), set(texts[1].split())),
            trainer.JACCARD_THRESHOLD,
        )
        self.assertGreaterEqual(
            trainer._token_jaccard(set(texts[1].split()), set(texts[2].split())),
            trainer.JACCARD_THRESHOLD,
        )
        self.assertLess(
            trainer._token_jaccard(set(texts[0].split()), set(texts[2].split())),
            trainer.JACCARD_THRESHOLD,
        )
        rows = []
        for index, text in enumerate(texts):
            rows.append(
                {
                    "feedback_id": f"row-{index}",
                    "normalized_text": text,
                    "surface_family": f"family-{index}",
                    "scenario_group": f"scenario-group-{index}",
                    "grounding_id": f"grounding-{index}",
                    "scenario_id": f"scenario-{index}",
                    "expected_feedback_type": generator.LABELS[index],
                }
            )
        _, graph = trainer.build_similarity_components(rows)
        self.assertEqual(graph["audit"]["component_count"], 1)


class CleanV3PipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.forbidden_reads = []
        original_builtin_open = builtins.open
        original_io_open = io.open

        def check_path(file):
            try:
                resolved = Path(file).resolve()
            except (TypeError, OSError):
                return
            normalized = str(resolved).replace("\\", "/").casefold()
            adapted_data = "/adapted_overcooked/data/" in normalized
            if adapted_data:
                cls.forbidden_reads.append(normalized)
                raise AssertionError(f"clean-v3 opened forbidden data input: {resolved}")

        def guarded_builtin_open(file, *args, **kwargs):
            mode = str(args[0] if args else kwargs.get("mode", "r"))
            if "r" in mode:
                check_path(file)
            return original_builtin_open(file, *args, **kwargs)

        def guarded_io_open(file, *args, **kwargs):
            mode = str(args[0] if args else kwargs.get("mode", "r"))
            if "r" in mode:
                check_path(file)
            return original_io_open(file, *args, **kwargs)

        with mock.patch("builtins.open", guarded_builtin_open), mock.patch(
            "io.open", guarded_io_open
        ):
            cls.artifact, cls.report, cls.partitions = trainer.train_candidate_in_memory()

    def test_no_old_human_test_or_membership_data_was_read(self):
        self.assertEqual(self.forbidden_reads, [])
        self.assertEqual(self.report["provenance"]["old_synthetic_rows_read"], 0)
        self.assertEqual(self.report["provenance"]["human_rows_read"], 0)
        self.assertEqual(
            self.report["provenance"]["dev_test_frozen_membership_files_read"], 0
        )

    def test_all_partition_pairs_are_component_and_or_disjoint(self):
        split = self.report["split"]
        self.assertEqual(
            split["cross_partition_similarity_counts"],
            {
                "normalized_exact": 0,
                "sequence_matcher_at_least_0_88": 0,
                "token_jaccard_at_least_0_70": 0,
            },
        )
        self.assertTrue(split["similarity_quality_gate_passed"])
        for overlap in split["pairwise_group_overlap"].values():
            self.assertTrue(all(value == 0 for value in overlap.values()))
        for rows in self.partitions.values():
            counts = Counter(row["expected_feedback_type"] for row in rows)
            self.assertEqual(set(counts), set(generator.LABELS))
            self.assertEqual(len(set(counts.values())), 1)

    def test_split_is_invariant_to_input_order(self):
        rows = generator.generate_rows()
        random.Random(99).shuffle(rows)
        shuffled, _ = trainer.split_rows(rows)
        expected = {
            row["feedback_id"]: split
            for split, values in self.partitions.items()
            for row in values
        }
        actual = {
            row["feedback_id"]: split
            for split, values in shuffled.items()
            for row in values
        }
        self.assertEqual(expected, actual)

    def test_selection_calibration_and_final_roles_are_separate(self):
        selection = self.report["selection"]
        calibration = self.report["calibration"]
        final_eval = self.report["final_synthetic_evaluation"]
        self.assertEqual(selection["selection_scope"], "train_group_cv_only")
        self.assertEqual(selection["calibration_rows_seen"], 0)
        self.assertEqual(selection["final_eval_rows_seen"], 0)
        self.assertTrue(selection["vectorizer_refit_inside_each_fold"])
        self.assertEqual(calibration["fit_split"], "calibration")
        self.assertFalse(calibration["fit_split_used_for_model_selection"])
        self.assertFalse(calibration["final_eval_used"])
        self.assertGreater(calibration["temperature"], 0.0)
        self.assertEqual(final_eval["selection_role"], "none_frozen_final_evaluation_only")
        self.assertFalse(final_eval["model_or_temperature_mutated"])
        self.assertEqual(
            self.report["frozen_model_state_sha256_before_final"],
            self.report["frozen_model_state_sha256_after_final"],
        )

    def test_frozen_evaluator_does_not_call_selection_or_fitting(self):
        final_rows = self.partitions["final_eval"]
        temperature = self.artifact["probability_calibration"]["temperature"]
        with mock.patch.object(
            trainer,
            "select_hyperparameters_train_only",
            side_effect=AssertionError("selection called from final evaluator"),
        ), mock.patch.object(
            trainer,
            "fit_base_model",
            side_effect=AssertionError("base fit called from final evaluator"),
        ), mock.patch.object(
            trainer,
            "fit_temperature_calibration_only",
            side_effect=AssertionError("calibration called from final evaluator"),
        ):
            metrics = trainer.evaluate_frozen_final(
                self.artifact["vectorizer"],
                self.artifact["classifier"],
                temperature,
                final_rows,
            )
        self.assertEqual(metrics["selection_role"], "none_frozen_final_evaluation_only")

    def test_status_is_computed_and_natural_probe_is_not_a_gate(self):
        expected = (
            "passed_research_candidate"
            if all(self.report["status_checks"].values())
            else "failed_research_candidate"
        )
        self.assertEqual(self.report["status"], expected)
        self.assertFalse(self.report["natural_probe"]["included_in_pass_gate"])
        self.assertFalse(self.report["production_promotion_eligible"])
        self.assertTrue(self.report["not_a_real_player_accuracy_claim"])

    def test_isolated_writer_hashes_every_output(self):
        output_dir = ROOT / "outputs" / "clean-v3-writer-contract-no-real-write"
        with mock.patch.object(Path, "mkdir"), mock.patch.object(
            trainer, "_write_json"
        ) as write_json, mock.patch("joblib.dump") as dump, mock.patch.object(
            trainer.os, "replace"
        ), mock.patch.object(
            trainer, "_sha256", side_effect=lambda path: f"sha256:{Path(path).name}"
        ):
            manifest = trainer.write_candidate_outputs(
                output_dir, self.artifact, self.report, self.partitions
            )
        self.assertEqual(write_json.call_count, 3)
        dump.assert_called_once()
        self.assertEqual(set(manifest["outputs"]), {"corpus", "report", "model"})
        self.assertEqual(
            {entry["sha256"] for entry in manifest["outputs"].values()},
            {"sha256:corpus.json", "sha256:evaluation_report.json", "sha256:model.joblib"},
        )
        self.assertTrue(manifest["research_candidate_only"])
        self.assertFalse(manifest["production_promotion_eligible"])


if __name__ == "__main__":
    unittest.main()
