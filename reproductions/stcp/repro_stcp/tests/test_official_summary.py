from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from run_official_synthetic import summarize  # noqa: E402


class OfficialSummaryTest(unittest.TestCase):
    def test_selects_lowest_std_with_acceptable_coverage(self) -> None:
        result = {
            "meta": {
                "method_order": ["GLCP", "SCC"],
                "lbds": [0.0, 0.1, 1.0],
                "alpha": 0.1,
                "n": 30,
                "m": 500,
                "repeats": 10,
                "testN": 100,
            },
            "base": [
                np.array([0.90, 0.90]),
                np.array([2.0, 2.0]),
                np.array([1.0, 0.8]),
                np.array([0.1, 0.1]),
            ],
            "StCP": [
                np.array(
                    [[0.90, 0.91, 0.80], [0.90, 0.91, 0.92]]
                ),
                np.ones((2, 3)),
                np.array([[1.0, 0.6, 0.1], [0.8, 0.7, 0.6]]),
                np.ones((2, 3)),
            ],
            "selected_lambda": np.array([0.1, 1.0]),
            "StCP-sel": [],
        }
        evidence = summarize(result, 1.0, "abc")
        self.assertEqual(
            evidence["methods"]["GLCP"]["best_acceptable"]["lambda"],
            0.1,
        )
        self.assertAlmostEqual(
            evidence["methods"]["GLCP"]["best_acceptable"][
                "relative_std_reduction"
            ],
            0.4,
        )
        self.assertEqual(evidence["claim_verdict"], "DIRECTIONAL ONLY")


if __name__ == "__main__":
    unittest.main()
