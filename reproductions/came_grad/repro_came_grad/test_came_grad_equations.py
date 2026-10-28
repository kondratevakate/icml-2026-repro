import math
import unittest

import numpy as np

from came_grad_equations import combine_gradients
from audit_claims import source_anchors


class CameGradEquationTests(unittest.TestCase):
    def test_trust_region_boundary_and_stage2_norm(self):
        gradients = np.array([[2.0, 1.0], [-1.0, 2.0], [0.5, -0.25]])
        weights = np.array([1.0, 2.0, 0.5])
        rho, kappa = 0.4, 1.5
        result = combine_gradients(
            gradients, weights, rho=rho, kappa=kappa, nu=0.2
        )
        self.assertAlmostEqual(
            np.linalg.norm(result.rectified - result.mean),
            rho * np.linalg.norm(result.mean),
            places=10,
        )
        self.assertAlmostEqual(
            np.linalg.norm(result.enhanced),
            kappa * np.linalg.norm(result.joint),
            places=10,
        )

    def test_fusion_limiting_cases(self):
        gradients = np.array([[2.0, 1.0], [-1.0, 2.0]])
        weights = np.array([1.0, 3.0])
        left = combine_gradients(
            gradients, weights, rho=0.2, kappa=1.2, nu=0.0
        )
        right = combine_gradients(
            gradients, weights, rho=0.2, kappa=1.0, nu=1.0
        )
        np.testing.assert_allclose(left.final, left.enhanced)
        np.testing.assert_allclose(right.final, right.joint)

    def test_invalid_hyperparameters_rejected(self):
        gradients = np.eye(2)
        weights = np.ones(2)
        for kwargs in (
            {"rho": -0.1},
            {"rho": 1.0},
            {"kappa": 0.9},
            {"nu": -0.1},
            {"nu": 1.1},
        ):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                combine_gradients(gradients, weights, **kwargs)

    def test_equal_norm_does_not_fix_covariance(self):
        constant = np.repeat([[1.0, 0.0]], 1000, axis=0)
        signs = np.tile([-1.0, 1.0], 500)
        variable = np.column_stack(
            [np.full(1000, math.sqrt(0.5)), signs * math.sqrt(0.5)]
        )
        np.testing.assert_allclose(
            np.linalg.norm(constant, axis=1),
            np.linalg.norm(variable, axis=1),
        )
        self.assertAlmostEqual(
            float(np.trace(np.cov(constant, rowvar=False, bias=True))), 0.0
        )
        self.assertGreater(
            float(np.trace(np.cov(variable, rowvar=False, bias=True))), 0.49
        )

    def test_published_ce_average_arithmetic(self):
        tex = (
            __import__("pathlib").Path(__file__).parents[1]
            / "official"
            / "arxiv-source"
            / "example_paper.tex"
        )
        tables = source_anchors(tex)["published_table_arithmetic"]
        self.assertEqual(tables["mimic_cxr"]["pair_count"], 8)
        self.assertEqual(tables["iu_xray"]["pair_count"], 8)
        self.assertAlmostEqual(
            tables["mimic_cxr"]["mean_absolute_difference"], 0.023125
        )
        self.assertAlmostEqual(
            tables["iu_xray"]["mean_absolute_difference"], 0.019
        )


if __name__ == "__main__":
    unittest.main()
