from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from audit_claims import (  # noqa: E402
    bridge_error_reduction_check,
    no_future_diagnostic,
    policy_rate_reduction_check,
    relevance_diagnostic,
    simulate_official_dgp,
    wrapped_gaussian_counterexample,
)


class ClaimAuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.data = simulate_official_dgp(n=20000, seed=20260729)

    def test_counterexample_has_no_l2_bridge(self) -> None:
        result = wrapped_gaussian_counterexample()
        self.assertTrue(all(result["checks"].values()))

    def test_official_dgp_satisfies_relevance_diagnostic(self) -> None:
        result = relevance_diagnostic(self.data)
        self.assertTrue(all(result["checks"].values()))

    def test_ill_posedness_reduction(self) -> None:
        result = bridge_error_reduction_check(trials=400)
        self.assertEqual(result["violations"], 0)
        self.assertTrue(all(result["checks"].values()))

    def test_official_dgp_has_no_direct_future_dependence(self) -> None:
        result = no_future_diagnostic(self.data)
        self.assertTrue(all(result["checks"].values()))

    def test_policy_rate_reduction(self) -> None:
        result = policy_rate_reduction_check()
        self.assertTrue(all(result["checks"].values()))


if __name__ == "__main__":
    unittest.main()
