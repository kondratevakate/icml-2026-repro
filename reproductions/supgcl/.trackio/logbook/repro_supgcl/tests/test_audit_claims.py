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


class SupGCLAuditTests(unittest.TestCase):
    def test_softmax_rows_sum_to_one(self) -> None:
        values = np.asarray([[1.0, 2.0, -4.0], [9.0, 3.0, 2.0]])
        observed = AUDIT.softmax(values)
        np.testing.assert_allclose(observed.sum(axis=1), 1.0, atol=1e-15)

    def test_kl_is_zero_for_identical_distribution(self) -> None:
        p = np.asarray([0.2, 0.3, 0.5])
        self.assertAlmostEqual(AUDIT.kl(p, p), 0.0, places=15)

    def test_theorem_one_chain_rule(self) -> None:
        for seed in range(20):
            result = AUDIT.theorem_one_probe(seed)
            self.assertLess(result["absolute_error"], 1e-12)

    def test_corollary_distributions_approach_uniform(self) -> None:
        result = AUDIT.corollary_probe(17)
        rows = result["records"]
        errors = [
            max(
                row["max_teacher_distance_to_uniform"],
                row["max_student_distance_to_uniform"],
            )
            for row in rows
        ]
        self.assertTrue(all(b < a for a, b in zip(errors, errors[1:])))

    def test_corollary_loss_reaches_node_limit(self) -> None:
        result = AUDIT.corollary_probe(29)
        final = result["records"][-1]
        self.assertLess(final["distance_to_node_limit"], 1e-4)
        self.assertLess(final["augmentation_kl"], 1e-8)


if __name__ == "__main__":
    unittest.main()
