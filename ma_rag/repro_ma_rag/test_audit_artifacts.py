from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

try:
    from .audit_artifacts import audit_all_datasets, audit_dataset
except ImportError:
    from audit_artifacts import audit_all_datasets, audit_dataset


def valid_row(row_id: int = 0, question: str = "Question?") -> dict:
    return {
        "id": row_id,
        "question": question,
        "option": {"A": "Alpha", "B": "Beta"},
        "answer": "A",
    }


class DatasetAuditTests(unittest.TestCase):
    def write(self, root: Path, name: str, rows: list[dict]) -> Path:
        path = root / name
        path.write_text(json.dumps(rows), encoding="utf-8")
        return path

    def test_valid_dataset(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            result, _ = audit_dataset(
                self.write(Path(temp), "valid.json", [valid_row()])
            )
            self.assertTrue(result["passed"])

    def test_duplicate_id_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            rows = [valid_row(1, "One?"), valid_row(1, "Two?")]
            result, _ = audit_dataset(self.write(Path(temp), "dup.json", rows))
            self.assertFalse(result["passed"])
            self.assertEqual(result["duplicate_ids"], 1)

    def test_answer_outside_options_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            row = valid_row()
            row["answer"] = "C"
            result, _ = audit_dataset(self.write(Path(temp), "bad.json", [row]))
            self.assertFalse(result["passed"])
            self.assertIn("answer not in options", result["errors"][0])

    def test_cross_dataset_overlap_detected(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            self.write(root, "a.json", [valid_row(0, "Same  question?")])
            self.write(root, "b.json", [valid_row(0, " same question? ")])
            result = audit_all_datasets(root)
            self.assertEqual(
                result["cross_dataset_question_overlaps"]["a::b"]["normalized"],
                1,
            )


if __name__ == "__main__":
    unittest.main()
