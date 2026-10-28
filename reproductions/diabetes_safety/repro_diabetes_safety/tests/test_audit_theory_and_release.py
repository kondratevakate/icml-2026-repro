from __future__ import annotations

import sys
import unittest
from pathlib import Path


HERE = Path(__file__).resolve()
REPRO = HERE.parents[1]
ROOT = HERE.parents[2]
sys.path.insert(0, str(REPRO))

from audit_theory_and_release import (  # noqa: E402
    audit_release_source,
    audit_safety_bound,
    finite_penalty_counterexample,
)


class DiabetesSafetyAuditTests(unittest.TestCase):
    def test_conditional_theorem_implication(self) -> None:
        result = audit_safety_bound()
        self.assertEqual(result["verdict"], "VERIFIED_CONDITIONAL_IMPLICATION")
        self.assertEqual(result["checked_implications"], 320)
        self.assertEqual(result["hypoglycemia_checked_implications"], 160)
        self.assertEqual(result["hyperglycemia_checked_implications"], 160)
        self.assertLessEqual(result["max_safety_margin_residual"], 0.0)
        self.assertGreater(result["hypoglycemia_mutation_failures"], 0)
        self.assertGreater(result["hyperglycemia_mutation_failures"], 0)

    def test_finite_penalty_retains_unsafe_probability(self) -> None:
        result = finite_penalty_counterexample()
        self.assertGreater(result["unsafe_action_probability"], 0.0)
        self.assertFalse(result["hard_pruning"])

    def test_released_path_and_cohort_contracts(self) -> None:
        result = audit_release_source(ROOT / "official" / "GlucoAlg")
        self.assertFalse(result["path_contract_matches"])
        self.assertFalse(result["cohort_indices_match"])
        self.assertTrue(result["finite_logit_penalty_anchor"])
        self.assertTrue(result["cohort_literal_comparison_anchor"])


if __name__ == "__main__":
    unittest.main()
