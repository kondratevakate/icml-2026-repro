from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from audit_claims import (  # noqa: E402
    random_delta_counterexample,
    selected_quantile_counterexample,
    set_stability_audit,
    stability_rate_sanity,
)


class AuditClaimsTest(unittest.TestCase):
    def test_set_stability_decomposition(self) -> None:
        checks = set_stability_audit()["checks"]
        self.assertTrue(all(checks.values()))

    def test_random_delta_proof_step_counterexample(self) -> None:
        checks = random_delta_counterexample()["checks"]
        self.assertTrue(checks["paper_realized_delta_step_fails"])
        self.assertTrue(checks["expected_delta_repairs_this_step"])

    def test_stability_rate_sanity_and_mutation(self) -> None:
        checks = stability_rate_sanity()["checks"]
        self.assertTrue(all(checks.values()))

    def test_selected_quantile_exact_counterexample(self) -> None:
        result = selected_quantile_counterexample()
        self.assertAlmostEqual(result["exact_coverage_at_q_L"], 27 / 31)
        self.assertGreater(result["exhaustive_grid_failure_count"], 0)
        self.assertTrue(all(result["checks"].values()))


if __name__ == "__main__":
    unittest.main()
