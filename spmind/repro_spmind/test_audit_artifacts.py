import json
import tempfile
import unittest
from pathlib import Path

from audit_artifacts import audit_benchmark


def record(**updates):
    value = {
        "id": "signal_calibration_001",
        "query": "Process {input_dir} into {output_dir}.",
        "tier": "basic",
        "category": "SIGNAL_CALIBRATION",
        "stages": ["illumination"],
        "num_stages": 1,
        "context_specific": False,
    }
    value.update(updates)
    return value


class BenchmarkAuditTests(unittest.TestCase):
    def audit(self, records):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bench.jsonl"
            path.write_text(
                "\n".join(json.dumps(item) for item in records) + "\n",
                encoding="utf-8",
            )
            return audit_benchmark(path)

    def test_valid_minimal_record_passes(self):
        result = self.audit([record()])
        self.assertTrue(result["passed"])
        self.assertEqual(result["record_count"], 1)

    def test_duplicate_id_and_query_fail(self):
        result = self.audit([record(), record()])
        self.assertFalse(result["passed"])
        self.assertTrue(any("duplicate IDs" in error for error in result["errors"]))
        self.assertTrue(
            any("duplicate normalized queries" in error for error in result["errors"])
        )

    def test_num_stages_mismatch_fails(self):
        result = self.audit([record(num_stages=2)])
        self.assertFalse(result["passed"])
        self.assertTrue(any("len(stages)" in error for error in result["errors"]))

    def test_tier_stage_rule_fails(self):
        result = self.audit(
            [
                record(
                    tier="advanced",
                    stages=["illumination", "registration"],
                    num_stages=2,
                )
            ]
        )
        self.assertFalse(result["passed"])
        self.assertTrue(
            any("tier/stage-count rule" in error for error in result["errors"])
        )


if __name__ == "__main__":
    unittest.main()
