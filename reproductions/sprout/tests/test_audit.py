from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT / "official" / "repo"
sys.path.insert(0, str(ROOT))

from audit_sprout import audit  # noqa: E402


class SproutReleaseAuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.result = audit(
            REPO, ROOT / "official" / "paper.pdf", ROOT / "official" / "source.tar.gz"
        )

    def test_self_reference_executes_on_synthetic_histology(self):
        result = self.result["self_reference"]
        self.assertTrue(result["high_confidence"])
        self.assertGreaterEqual(result["precision"], 0.99)
        self.assertGreaterEqual(result["recall"], 0.60)

    def test_canonical_ot_failure_is_reproduced(self):
        result = self.result["partial_ot"]
        self.assertIn("AttributeError", result["canonical_error"])
        self.assertIn("numItermax", result["canonical_error"])
        self.assertNotIn("numItermax", result["constructor_attributes"])

    def test_runtime_completed_ot_reveals_n_scaling(self):
        result = self.result["partial_ot"]
        self.assertAlmostEqual(result["real_mass"], 2.4, places=5)
        self.assertAlmostEqual(result["slack_mass"], 1.6, places=5)
        for row_sum in result["row_sums"]:
            self.assertAlmostEqual(row_sum, 1.0, places=5)

    def test_convexity_does_not_imply_unique_optimizer(self):
        result = self.result["theory"]
        self.assertTrue(result["distinct"])
        self.assertTrue(result["both_zero_cost_and_zero_marginal_kl"])
        self.assertEqual(result["diagonal_rows"], result["diffuse_rows"])
        self.assertEqual(result["diagonal_columns"], result["diffuse_columns"])

    def test_containment_penalty_prefers_small_instances(self):
        result = self.result["refinement"]
        self.assertAlmostEqual(result["output_scores"][0], 0.8)
        self.assertAlmostEqual(result["output_scores"][1], 0.8)
        self.assertLess(result["output_scores"][2], 0.22)

    def test_empty_nms_input_bug_is_reproduced(self):
        self.assertIn("IndexError", self.result["refinement"]["empty_input_error"])

    def test_perfect_identity_metrics_are_one(self):
        for value in self.result["metrics"]["perfect_identity"].values():
            self.assertAlmostEqual(value, 1.0, places=5)

    def test_release_availability_gaps_are_recorded(self):
        release = self.result["release"]
        self.assertFalse(release["documented_requirements_exists"])
        self.assertTrue(release["released_requirements_exists"])
        self.assertTrue(release["preprocessed_dataset_link_is_empty"])
        self.assertFalse(release["default_backbone_repo_resolved_at_freeze"])
        self.assertEqual(release["empirical_artifacts"], [])


if __name__ == "__main__":
    unittest.main()
