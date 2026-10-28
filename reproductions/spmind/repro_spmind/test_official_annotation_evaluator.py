import importlib.util
import math
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


EVALUATOR_PATH = (
    Path(__file__).parents[1]
    / "official"
    / "code"
    / "experiments"
    / "run_annotation_eval.py"
)
SPEC = importlib.util.spec_from_file_location("official_annotation_eval", EVALUATOR_PATH)
EVALUATOR = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(EVALUATOR)


class OfficialAnnotationEvaluatorTests(unittest.TestCase):
    def test_ghk_boundary_and_quarter_distance(self):
        self.assertEqual(EVALUATOR.ghk(1.0, 0.25), 1.0)
        self.assertAlmostEqual(
            EVALUATOR.ghk(0.75, 0.25), math.exp(-0.5), places=12
        )

    def test_cluster_mode(self):
        frame = pd.DataFrame(
            {
                "cluster": [1, 1, 1, 2],
                "Annotation": ["B", "B", "T", "T"],
            }
        )
        self.assertEqual(
            EVALUATOR._cluster_mode_labels(frame, "cluster", "Annotation"),
            {1: "B", 2: "T"},
        )

    def test_inconsistent_prediction_cluster_is_rejected(self):
        frame = pd.DataFrame(
            {"cluster": [1, 1], "Annotation": ["B", "T"]}
        )
        with self.assertRaisesRegex(ValueError, "inconsistent annotations"):
            EVALUATOR._validate_consistency(frame, "cluster", "Annotation")

    def test_equal_length_inputs_are_positionally_aligned(self):
        """Expose the evaluator's order-sensitive equal-length fast path."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            gt = root / "gt.csv"
            pred = root / "pred.csv"
            pd.DataFrame(
                {
                    "cellLabel": [1, 2],
                    "cluster": [1, 2],
                    "Annotation": ["A", "B"],
                }
            ).to_csv(gt, index=False)
            # Same cells and correct labels, but rows are deliberately reversed.
            pd.DataFrame(
                {
                    "cellLabel": [2, 1],
                    "cluster": [2, 1],
                    "Annotation": ["B", "A"],
                }
            ).to_csv(pred, index=False)

            original_describe = EVALUATOR.generate_descriptions
            original_embed = EVALUATOR.embed_labels
            try:
                EVALUATOR.generate_descriptions = (
                    lambda labels, *args, **kwargs: {label: label for label in labels}
                )
                EVALUATOR.embed_labels = lambda labels, *args, **kwargs: {
                    label: np.array([1.0, 0.0])
                    if label == "A"
                    else np.array([0.0, 1.0])
                    for label in labels
                }
                result = EVALUATOR.run_single(
                    str(gt), str(pred), generate_desc=False
                )
            finally:
                EVALUATOR.generate_descriptions = original_describe
                EVALUATOR.embed_labels = original_embed

            # An ID-based join would score 1.0. The released evaluator instead
            # aligns equal-length files by row position and scores near zero.
            self.assertAlmostEqual(
                result["summary"]["ghk_cell_weighted"], math.exp(-8), places=12
            )


if __name__ == "__main__":
    unittest.main()
