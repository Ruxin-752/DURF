from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.prepare_reference_classifier_corpus import (  # noqa: E402
    paper_strict_partition,
    prepare_reference_corpus,
)
from src.evaluation_splits import normalize_text  # noqa: E402


def _row(text: str, label: str, split: str, index: int) -> dict:
    return {
        "feedback_id": f"{split}_{label}_{index}",
        "text": text,
        "reference_type": label,
        "split": split,
        "group_id": f"{split}_group_{index}",
        "source": "llm",
        "phrase_annotations": [
            {"text": text, "start": 0, "end": len(text), "reference_type": label}
        ],
    }


class ReferenceClassifierCorpusTests(unittest.TestCase):
    def _complete_sources(self) -> tuple[list[dict], list[dict], list[dict]]:
        v7: list[dict] = []
        behavior: list[dict] = []
        other: list[dict] = []
        for split_index, split in enumerate(("train", "dev", "test")):
            for label_index, label in enumerate(
                ("trajectory", "feature", "action_spatial")
            ):
                v7.append(
                    _row(
                        f"unique {split} {label} phrase",
                        label,
                        split,
                        split_index * 10 + label_index,
                    )
                )
            behavior.append(
                _row(
                    f"unique {split} behavioral phrase",
                    "action_behavioral",
                    split,
                    100 + split_index,
                )
            )
            other.append(
                _row(
                    f"unique {split} other phrase",
                    "other",
                    split,
                    200 + split_index,
                )
            )
        return v7, behavior, other

    def test_drops_cross_label_and_cross_split_exact_text(self) -> None:
        v7, behavior, other = self._complete_sources()
        v7.extend(
            [
                _row("same conflicting phrase", "trajectory", "train", 300),
                _row("same conflicting phrase", "feature", "train", 301),
                _row("same leaked phrase", "trajectory", "train", 302),
                _row("same leaked phrase", "trajectory", "test", 303),
            ]
        )
        kept, report = prepare_reference_corpus(v7, behavior, other)
        texts = {row["text"] for row in kept}
        self.assertNotIn("same conflicting phrase", texts)
        self.assertNotIn("same leaked phrase", texts)
        self.assertEqual(report["cross_label_conflicting_texts"], 1)
        self.assertEqual(report["cross_split_leaking_texts"], 1)
        self.assertEqual(report["kept_rows"], 15)

    def test_rejects_group_ids_declared_in_multiple_splits(self) -> None:
        v7, behavior, other = self._complete_sources()
        v7[0]["group_id"] = "leaking_group"
        v7[-1]["group_id"] = "leaking_group"
        with self.assertRaisesRegex(ValueError, "group ids cross declared splits"):
            prepare_reference_corpus(v7, behavior, other)

    def test_paper_strict_partition_is_task_and_text_disjoint(self) -> None:
        paper_labels = (
            "trajectory",
            "features",
            "object_spatial",
            "object_behavior",
            "other",
        )
        source = [
            {
                "task_uuid": f"task-{task}",
                "phrase": f"paper task {task} label {label}",
                "reference_type": label,
            }
            for task in range(30)
            for label in paper_labels
        ]
        rows, report = paper_strict_partition(
            source,
            candidate_seeds=64,
            minimum_per_label=1,
        )
        self.assertFalse(report["selection_uses_model_predictions"])
        self.assertEqual(report["group_overlap_count"], 0)
        self.assertEqual(report["normalized_text_overlap_count"], 0)
        groups = {
            split: {
                row["paper_task_uuid"] for row in rows if row["split"] == split
            }
            for split in ("train", "dev", "test")
        }
        texts = {
            split: {
                normalize_text(row["text"])
                for row in rows
                if row["split"] == split
            }
            for split in ("train", "dev", "test")
        }
        for left, right in (("train", "dev"), ("train", "test"), ("dev", "test")):
            self.assertFalse(groups[left] & groups[right])
            self.assertFalse(texts[left] & texts[right])
        self.assertTrue(all(row["source"] == "original_paper_human" for row in rows))

    def test_reserved_external_text_is_removed_before_training(self) -> None:
        v7, behavior, other = self._complete_sources()
        v7.append(_row("sealed human phrase", "trajectory", "train", 500))
        kept, report = prepare_reference_corpus(
            v7,
            behavior,
            other,
            reserved_external_texts={"sealed human phrase"},
        )
        self.assertNotIn("sealed human phrase", {row["text"] for row in kept})
        self.assertEqual(report["reserved_external_text_rows_dropped"], 1)


if __name__ == "__main__":
    unittest.main()
