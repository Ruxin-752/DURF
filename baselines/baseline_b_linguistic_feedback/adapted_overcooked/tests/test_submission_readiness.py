from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.check_submission_readiness import build_readiness_report, render_markdown


class SubmissionReadinessTests(unittest.TestCase):
    def test_checked_in_artifacts_keep_claim_scopes_separate(self) -> None:
        report = build_readiness_report(ROOT)
        self.assertTrue(report["engineering_ready"])
        self.assertEqual(
            report["overall_status"], "engineering_ready_human_evaluation_pending"
        )
        self.assertEqual(report["human_evidence"]["five_class_gold_rows"], 0)
        self.assertFalse(
            report["human_evidence"]["five_class_accuracy_claim_allowed"]
        )
        self.assertGreaterEqual(
            report["headline_metrics"]["five_class_paper_protocol"]["accuracy"],
            0.87,
        )
        self.assertIn(
            "auxiliary",
            report["claim_boundaries"][-1].lower(),
        )

    def test_markdown_does_not_claim_human_accuracy(self) -> None:
        markdown = render_markdown(build_readiness_report(ROOT))
        self.assertIn("尚需真人完成", markdown)
        self.assertIn("不声明真人语言 accuracy", markdown)
        self.assertIn("不是论文 87% 指标", markdown)


if __name__ == "__main__":
    unittest.main()
