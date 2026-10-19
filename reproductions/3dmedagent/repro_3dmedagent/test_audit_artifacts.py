import csv
import tempfile
import unittest
from pathlib import Path

try:
    from .audit_artifacts import audit_csv
except ImportError:
    from audit_artifacts import audit_csv


FIELDS = [
    "Question ID",
    "Image ID",
    "question type",
    "question subtype",
    "multiple-choice question",
    "correct option",
    "split",
    "dataset",
]


class CsvAuditTests(unittest.TestCase):
    def audit(self, rows):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "test.csv"
            with path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=FIELDS)
                writer.writeheader()
                writer.writerows(rows)
            return audit_csv(path)

    def row(self, **updates):
        row = {
            "Question ID": "1",
            "Image ID": "scan-1",
            "question type": "recognition",
            "question subtype": "existence",
            "multiple-choice question": "Question? A: Yes B: No",
            "correct option": "A",
            "split": "train",
            "dataset": "source",
        }
        row.update(updates)
        return row

    def test_valid_row(self):
        self.assertTrue(self.audit([self.row()])["passed"])

    def test_duplicate_id_and_question_rejected(self):
        result = self.audit([self.row(), self.row()])
        self.assertFalse(result["passed"])
        self.assertIn("duplicate Question ID", result["errors"])
        self.assertEqual(result["repeated_scan_mcq_pair_rows"], 1)

    def test_same_question_on_different_scans_is_allowed(self):
        result = self.audit(
            [
                self.row(),
                self.row(**{"Question ID": "2", "Image ID": "scan-2"}),
            ]
        )
        self.assertTrue(result["passed"])
        self.assertEqual(result["unique_mcq_texts"], 1)
        self.assertEqual(result["unique_scan_mcq_pairs"], 2)

    def test_missing_answer_rejected(self):
        result = self.audit([self.row(**{"correct option": ""})])
        self.assertFalse(result["passed"])
        self.assertIn("missing correct option", result["errors"])


if __name__ == "__main__":
    unittest.main()
