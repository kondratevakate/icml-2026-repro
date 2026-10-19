import sys
import unittest
from pathlib import Path


BUNDLE = Path(__file__).resolve().parents[1]
MODULE_ROOT = BUNDLE
PROJECT_ROOT = BUNDLE.parent
sys.path.insert(0, str(MODULE_ROOT))

from audit_claims import run_audit


class MedMambaAuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.evidence = run_audit(
            PROJECT_ROOT / "official" / "code",
            PROJECT_ROOT / "official" / "source",
        )
        cls.claims = {item["id"]: item for item in cls.evidence["claims"]}

    def test_embedding_collapses_channel_axis(self):
        claim = self.claims["C1"]
        self.assertEqual(claim["observed_rank"], 3)
        self.assertEqual(claim["conv_groups"], [1, 1, 1])

    def test_first_difference_is_not_zero(self):
        claim = self.claims["C2"]
        self.assertTrue(claim["observed_first_step_equals_input"])
        self.assertEqual(claim["declared_first_step_max_abs"], 0.0)
        self.assertGreater(claim["operator_max_abs_difference"], 0.0)

    def test_frequency_filter_has_no_frequency_axis(self):
        claim = self.claims["C3"]
        self.assertEqual(claim["real_weight_shape"], [12])
        self.assertFalse(claim["has_frequency_axis"])

    def test_graph_is_input_invariant_and_output_disconnected(self):
        claim = self.claims["C4"]
        self.assertEqual(claim["adjacency_input_change_max_abs"], 0.0)
        self.assertGreater(claim["adjacency_parameter_mutation_max_abs"], 0.0)
        self.assertEqual(claim["output_parameter_mutation_max_abs"], 0.0)
        self.assertTrue(
            all(value is None for value in claim["classification_proxy_graph_gradients"].values())
        )

    def test_spatial_mamba_scans_time_not_channels(self):
        claim = self.claims["C5"]
        self.assertTrue(claim["mamba_input_shapes"])
        self.assertTrue(claim["all_mamba_sequence_axes_equal_time"])
        self.assertNotEqual(claim["input_time_length"], claim["input_channel_count"])

    def test_score_is_ten_without_empirical_claim(self):
        self.assertEqual(self.evidence["prepared_score"], 10)
        self.assertEqual(self.claims["C6"]["score"], 0)


if __name__ == "__main__":
    unittest.main()
