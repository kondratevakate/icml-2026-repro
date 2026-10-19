from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
MODULE_PATH = HERE.parent / "audit_claims.py"
SPEC = importlib.util.spec_from_file_location("audit_claims", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
AUDIT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(AUDIT)


class SBayFDNNAuditTests(unittest.TestCase):
    def test_independent_pip_is_half_at_analytic_threshold(self) -> None:
        inclusion, spike, slab, width = 1e-5, 1e-5, 2e-3, 64
        c1 = np.log(inclusion / (1.0 - inclusion))
        c1 += 0.5 * width * np.log(spike / slab)
        c2 = 0.5 / spike - 0.5 / slab
        threshold = -c1 / c2
        observed = AUDIT.independent_pip(
            np.asarray([threshold]), inclusion, spike, slab, width
        )
        self.assertAlmostEqual(float(observed[0]), 0.5, places=14)

    def test_independent_pip_increases_with_column_norm(self) -> None:
        values = AUDIT.independent_pip(
            np.asarray([0.0, 0.004, 0.008]), 1e-5, 1e-5, 2e-3, 64
        )
        self.assertTrue(bool(np.all(np.diff(values) > 0)))

    def test_interval_mapping_merges_touching_supports(self) -> None:
        intervals = AUDIT.independent_intervals(np.asarray([0, 1, 2]), 24, 4)
        self.assertEqual(intervals, [(0.0, 0.15000000000000002)])

    def test_reversed_rate_condition_has_counterexample(self) -> None:
        n = np.asarray([10.0, 100.0, 1000.0])
        complexity = 1.0 / n
        epsilon2 = complexity**2
        self.assertTrue(bool(np.all(epsilon2 <= complexity)))
        self.assertTrue(bool(np.all(np.diff(complexity / epsilon2) > 0)))

    def test_positive_shrinking_bound_breaks_approximation_rate(self) -> None:
        n = np.asarray([100.0, 400.0, 1600.0, 6400.0, 25600.0])
        scale = np.sqrt(n)
        output_upper_bound = 2.0 / n
        error_lower_bound = 1.0 - output_upper_bound
        claimed_rate = scale**-2 + scale**-1
        self.assertTrue(bool(np.all(np.diff(claimed_rate) < 0)))
        self.assertTrue(
            bool(np.all(np.diff(error_lower_bound / claimed_rate) > 0))
        )


if __name__ == "__main__":
    unittest.main()
