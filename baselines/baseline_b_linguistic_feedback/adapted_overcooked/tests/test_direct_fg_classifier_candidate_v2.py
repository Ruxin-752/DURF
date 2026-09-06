from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.train_direct_fg_classifier_candidate_v2 import (  # noqa: E402
    _near_values,
    build_direct_fg_v2_corpus,
    compute_candidate_status,
    train_direct_fg_v2_candidate,
)


class DirectFgClassifierCandidateV2Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.rows, cls.audit = build_direct_fg_v2_corpus()

    def test_exact_source_counts_and_quality_exclusions(self) -> None:
        self.assertEqual(len(self.rows), 917)
        self.assertEqual(
            self.audit["source_counts"],
            {
                "boundary_direct_explicit": 504,
                "hard_v2_direct_explicit": 288,
                "human_augmentation_reviewed": 84,
                "human_explicit_reviewed": 41,
            },
        )
        self.assertEqual(
            self.audit["input_audits"]["human_train"]["excluded_rows"], 3
        )
        self.assertEqual(
            self.audit["input_audits"]["human_augmentation"]["excluded_rows"], 1
        )
        self.assertEqual(
            self.audit["input_audits"]["boundary"]["excluded_exact_core_overlap"],
            144,
        )

    def test_split_is_component_and_near_duplicate_disjoint(self) -> None:
        self.assertEqual(self.audit["split_rows"], {"dev": 162, "train": 755})
        self.assertEqual(
            self.audit["split_label_counts"]["dev"],
            {"descriptive": 54, "evaluative": 54, "imperative": 54},
        )
        self.assertTrue(
            all(value == 0 for value in self.audit["split_integrity"].values())
        )
        self.assertEqual(
            self.audit["component_split"]["cross_split_near_overlap"], 0
        )
        self.assertGreater(self.audit["component_split"]["near_edge_count"], 0)

    def test_near_metric_combines_sequence_tokens_and_length(self) -> None:
        near, sequence, token_jaccard, length_ratio = _near_values(
            "could you move the clean plate to the counter",
            "can you move the clean plate to the counter",
        )
        self.assertTrue(near)
        self.assertGreaterEqual(sequence, 0.88)
        self.assertGreaterEqual(token_jaccard, 0.70)
        self.assertGreaterEqual(length_ratio, 0.88)
        self.assertFalse(
            _near_values("good move", "please fetch the nearest onion")[0]
        )

    def test_no_mixed_test_or_frozen_input_is_opened(self) -> None:
        self.assertEqual(
            self.audit["data_scope"],
            {
                "deepseek_rows": 0,
                "mixed_test_containers_opened": 0,
                "human_test_files_opened": 0,
                "frozen_files_opened": 0,
            },
        )

    def test_status_is_computed_from_required_gates(self) -> None:
        self.assertEqual(
            compute_candidate_status({"a": True, "b": True}),
            "passed_research_candidate_not_promoted",
        )
        self.assertEqual(
            compute_candidate_status({"a": True, "b": False}),
            "failed_research_candidate_not_promoted",
        )
        self.assertEqual(
            compute_candidate_status({}),
            "failed_research_candidate_not_promoted",
        )

    def test_model_clears_evaluative_recall_gate(self) -> None:
        artifact, report = train_direct_fg_v2_candidate(self.rows)
        metrics = report["direct_fg_near_component_disjoint_dev"]["classification"]
        self.assertGreaterEqual(metrics["accuracy"], 0.87)
        self.assertGreaterEqual(metrics["macro_f1"], 0.87)
        self.assertGreaterEqual(metrics["per_class"]["evaluative"]["recall"], 0.85)
        self.assertFalse(artifact["reference_type_used_as_target"])
        self.assertEqual(report["selection_protocol"]["test_or_frozen_rows_used"], 0)


if __name__ == "__main__":
    unittest.main()
