import sys
import unittest
from pathlib import Path


BUNDLE = Path(__file__).resolve().parents[1]
CANDIDATE = BUNDLE.parent
sys.path.insert(0, str(BUNDLE))

from audit_claims import run_audit


class CameGradAuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.result = run_audit(
            CANDIDATE / "official" / "code",
            CANDIDATE / "official" / "source" / "example_paper.tex",
        )
        cls.claims = {claim["id"]: claim for claim in cls.result["claims"]}

    def test_interaction_identity_and_mutation(self):
        evidence = self.claims["C1"]["evidence"]
        self.assertTrue(evidence["paper_anchor_present"])
        self.assertLess(evidence["identity_max_abs_error"], 1e-10)
        self.assertGreater(evidence["sign_mutation_max_abs_error"], 1.0)
        self.assertEqual(evidence["opposing_joint_energy"], 0.0)

    def test_stage1_counterexample_is_feasible_and_not_pareto_descent(self):
        evidence = self.claims["C2"]["evidence"]
        self.assertTrue(evidence["paper_anchor_present"])
        self.assertLessEqual(evidence["trust_region_residual"], 1e-12)
        self.assertAlmostEqual(
            evidence["primal_u_star"], evidence["dual_recovered_u"]
        )
        self.assertLess(evidence["worst_inner_product"], 0.0)

    def test_stage2_covariance_scaling_and_epsilon_shortfall(self):
        evidence = self.claims["C3"]["evidence"]
        self.assertTrue(evidence["paper_anchor_present"])
        self.assertGreater(evidence["relative_target_shortfall"], 0.0)
        self.assertAlmostEqual(
            evidence["observed_trace_ratio"],
            evidence["declared_trace_ratio"],
            places=12,
        )

    def test_adaptive_fusion_can_destroy_geometric_validity(self):
        evidence = self.claims["C4"]["evidence"]
        self.assertTrue(evidence["paper_formula_anchor_present"])
        self.assertTrue(evidence["paper_defines_validity_for_rectified_direction"])
        self.assertTrue(evidence["paper_acknowledges_conflict_resurgence"])
        self.assertEqual(evidence["dual_alpha_star"], 0.0)
        self.assertGreater(evidence["dual_right_derivative_at_zero"], 0.0)
        self.assertTrue(evidence["stage1_is_geometrically_valid"])
        self.assertAlmostEqual(
            evidence["stage2_observed_norm"],
            evidence["stage2_target_norm"],
            places=10,
        )
        self.assertFalse(evidence["final_is_geometrically_valid"])
        self.assertLess(min(evidence["final_inner_products"]), 0.0)

    def test_released_optimizer_is_missing(self):
        evidence = self.claims["C5"]["evidence"]
        self.assertFalse(evidence["author_optimizer_file_present"])
        self.assertTrue(evidence["trainer_imports_missing_optimizer"])
        self.assertTrue(evidence["readme_says_core_withheld"])

    def test_prepared_score(self):
        self.assertEqual(self.result["prepared_score"], 6)


if __name__ == "__main__":
    unittest.main()
