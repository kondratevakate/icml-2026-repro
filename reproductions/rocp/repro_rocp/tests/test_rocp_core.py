from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rocp_core import DiscreteROCP, robust_closed_form


class DiscreteROCPTests(unittest.TestCase):
    def setUp(self) -> None:
        self.losses = np.asarray(
            [
                [0.0, 3.0, 8.0],
                [2.0, 0.0, 5.0],
                [20.0, 4.0, 0.0],
            ]
        )
        self.predictor = DiscreteROCP(self.losses)

    def test_t_zero_uses_paper_remark_3_2(self) -> None:
        choice = self.predictor.pointwise_choice((0.7, 0.2, 0.1), 0.0)
        self.assertEqual(choice.label_set, frozenset((0, 1, 2)))
        self.assertEqual(choice.theta, np.max(self.losses[:, choice.action]))

    def test_beta_eventually_selects_full_set(self) -> None:
        selected = self.predictor.predict_set((0.7, 0.2, 0.1), beta=1e6)
        self.assertEqual(selected, frozenset((0, 1, 2)))

    def test_algorithm1_includes_candidate_in_n_plus_1_constraint(self) -> None:
        calibration_probs = [(0.75, 0.20, 0.05), (0.15, 0.70, 0.15)]
        calibration_labels = [0, 1]
        test_probs = (0.05, 0.25, 0.70)
        result = self.predictor.algorithm1_prediction_set(
            calibration_probs, calibration_labels, test_probs, alpha=1 / 3
        )
        self.assertTrue(result)
        for candidate in range(3):
            beta = self.predictor.minimal_feasible_beta(
                [*calibration_probs, test_probs],
                [*calibration_labels, candidate],
                alpha=1 / 3,
            )
            self.assertEqual(
                candidate in result,
                candidate in self.predictor.predict_set(test_probs, beta),
            )

    def test_closed_form_asymmetric_counterexample(self) -> None:
        losses = np.asarray([[1.0, 2.0], [1.0, 2.0], [100.0, 2.0]])
        values = robust_closed_form(losses, (0, 1), alpha=0.05)
        np.testing.assert_allclose(values, (5.95, 2.0))
        self.assertEqual(int(np.argmin(values)), 1)


if __name__ == "__main__":
    unittest.main()
