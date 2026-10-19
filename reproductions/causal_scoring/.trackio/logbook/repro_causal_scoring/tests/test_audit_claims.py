import sys
import unittest
from pathlib import Path


BUNDLE = Path(__file__).resolve().parents[1]
CANDIDATE = BUNDLE.parent
sys.path.insert(0, str(BUNDLE))

from audit_claims import run_audit


class CausalScoringAuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.result = run_audit(CANDIDATE / "official" / "source" / "main.tex")
        cls.claims = {claim["id"]: claim for claim in cls.result["claims"]}

    def test_c1_conditional_bound_and_crossfit_scope(self):
        evidence = self.claims["C1"]["evidence"]
        self.assertTrue(evidence["paper_anchors_present"])
        self.assertLessEqual(evidence["conditional_proof_max_bias_residual"], 1e-12)
        self.assertLessEqual(
            evidence["generic_sum_variance_inequality_max_residual"], 1e-12
        )
        self.assertGreater(evidence["weakened_constant_mutation_violations"], 0)
        self.assertGreater(evidence["crossfit_contribution_covariance"], 0)
        self.assertGreater(evidence["iid_formula_residual"], 0)

    def test_c2_curvature_identity_and_mutation(self):
        evidence = self.claims["C2"]["evidence"]
        self.assertTrue(evidence["paper_anchor_present"])
        self.assertEqual(evidence["symbolic_difference"], "0")
        self.assertLess(evidence["max_relative_error"], 1e-12)
        self.assertGreater(evidence["mutation_max_relative_error"], 0.1)

    def test_c3_strict_propriety_and_mutation(self):
        evidence = self.claims["C3"]["evidence"]
        self.assertTrue(evidence["paper_anchor_present"])
        self.assertEqual(evidence["loss1_differential_residual"], "0")
        self.assertEqual(evidence["loss0_differential_residual"], "0")
        self.assertGreater(evidence["minimum_declared_weight_at_half"], 0)
        self.assertLess(evidence["max_argmin_abs_error"], 3e-4)
        self.assertGreater(
            evidence["label_swap_mutation_min_mean_argmin_error"], 0.2
        )

    def test_c4_mapping_and_no_vanish_clause(self):
        evidence = self.claims["C4"]["evidence"]
        self.assertTrue(evidence["paper_anchors_present"])
        self.assertEqual(evidence["integrated_link_derivative_residual"], "0")
        self.assertEqual(evidence["quartic_symbolic_residual"], "0")
        self.assertTrue(evidence["strictly_monotone"])
        self.assertLess(evidence["max_roundtrip_relative_error"], 1e-8)
        self.assertLess(evidence["max_symmetry_error"], 1e-12)
        self.assertGreaterEqual(evidence["minimum_largest_root"], 4.0 - 1e-10)
        self.assertLess(
            evidence["quartic_root_vs_inverse_link_max_abs_error"], 1e-8
        )
        self.assertLess(evidence["quartic_max_relative_residual"], 1e-10)
        self.assertLess(evidence["canonical_gradient_abs_at_z_1e6_y1"], 0.002)
        self.assertGreater(evidence["mismatched_sigmoid_max_abs_gradient"], 1e3)

    def test_c5_kang_schafer_mechanism_and_feature_mutation(self):
        evidence = self.claims["C5"]["evidence"]
        self.assertTrue(evidence["paper_anchor_present"])
        expected = evidence["expected_fit_count_per_method"]
        self.assertTrue(
            all(count == expected for count in evidence["successful_fit_count"].values())
        )
        self.assertGreater(
            evidence["observed_rmse_ratio_log_over_tailored"], 2.0
        )
        self.assertGreater(evidence["misspecification_amplification"], 1.5)
        self.assertGreater(
            evidence["observed_tailored_abs_error_win_rate"], 0.7
        )
        self.assertLess(
            evidence["mean_boundary_rates"]["observed_tailored"],
            evidence["mean_boundary_rates"]["observed_log"],
        )

    def test_current_prepared_score(self):
        self.assertEqual(self.result["prepared_score"], 7)


if __name__ == "__main__":
    unittest.main()
