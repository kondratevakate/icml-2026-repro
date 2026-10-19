from __future__ import annotations

import importlib.util
import math
import sys
import unittest
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT / "official" / "repo"
CHECKPOINTS = ROOT / "official" / "checkpoint"

sys.path.insert(0, str(ROOT))
from audit_medcrp_cl import (  # noqa: E402
    EXPECTED_GROUP_SIZES,
    EXPECTED_TASK_TO_MODALITY,
    audit,
    paper_threshold_and_error,
)


def load_official_model_module():
    path = REPO / "scripts" / "model.py"
    spec = importlib.util.spec_from_file_location("official_medcrp_model", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class MedCRPReleaseAuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.result = audit(REPO, CHECKPOINTS)
        cls.model_module = load_official_model_module()

    def test_official_checkpoint_has_reported_partition(self):
        checkpoint = self.result["checkpoint"]
        self.assertEqual(checkpoint["total_tasks"], 16)
        self.assertEqual(checkpoint["task_to_modality"], EXPECTED_TASK_TO_MODALITY)
        self.assertEqual(checkpoint["group_sizes"], EXPECTED_GROUP_SIZES)

    def test_checkpoint_and_paper_alpha_mismatch_is_detected(self):
        checkpoint = self.result["checkpoint"]
        self.assertEqual(checkpoint["checkpoint_alpha"], 2.0)
        self.assertEqual(checkpoint["code_and_paper_alpha"], 5.0)
        self.assertFalse(
            self.result["checks"]["checkpoint_alpha_matches_paper_and_code"]
        )

    def test_checkpoint_similarity_statistics_are_recomputed(self):
        checkpoint = self.result["checkpoint"]
        self.assertAlmostEqual(checkpoint["intra_std"], 0.08084310674119105)
        self.assertAlmostEqual(checkpoint["inter_std"], 0.1212978440647021)
        self.assertEqual(checkpoint["intra_n"], 8)
        self.assertEqual(checkpoint["inter_n"], 54)

    def test_fixed_gaussian_overlap_refutes_zero_error_step(self):
        _, error = paper_threshold_and_error(0.94, 0.05, 0.51, 0.10)
        self.assertGreater(error, 0.04)
        self.assertLess(error, 0.05)
        self.assertGreater(self.result["theory_check"]["fixed_distribution_tail_error"], 0)

    def test_dynamic_lora_isolates_modalities(self):
        torch.manual_seed(7)
        layer = torch.nn.Linear(3, 2, bias=False)
        dynamic = self.model_module.DynamicModalityLoRALinear(
            layer, rank=1, alpha=1, initial_modalities=1, max_modalities=3
        )
        x = torch.tensor([[1.0, 2.0, 3.0]])
        dynamic.set_modality(0)
        with torch.no_grad():
            dynamic.lora_A[0].weight.fill_(1.0)
            dynamic.lora_B[0].weight.fill_(1.0)
        output_zero = dynamic(x).clone()

        dynamic.set_modality(1)
        output_one = dynamic(x).clone()
        self.assertFalse(torch.allclose(output_zero, output_one))

        dynamic.set_modality(0)
        self.assertTrue(torch.allclose(output_zero, dynamic(x)))

    def test_welford_update_matches_sample_statistics(self):
        manager = self.model_module.AdaptiveCRPModalityManager(
            None, None, {"crp_alpha": 5.0}, "cpu"
        )
        stats = {"n": 0, "mean": 0.0, "M2": 0.0}
        for value in (0.2, 0.4, 0.8):
            manager._update_welford(stats, value)
        self.assertAlmostEqual(stats["mean"], 1.4 / 3)
        expected = math.sqrt(sum((x - 1.4 / 3) ** 2 for x in (0.2, 0.4, 0.8)) / 2)
        self.assertAlmostEqual(manager._get_std(stats), expected)

    def test_ewc_release_is_aggregate_state_not_replay_data(self):
        ewc = self.result["ewc_state"]
        self.assertEqual(ewc["modality_task_count"], EXPECTED_GROUP_SIZES)
        self.assertTrue(ewc["contains_only_aggregate_tensors_and_identifiers"])
        self.assertEqual(set(ewc["key_summary"]), {"0", "1", "2", "3", "4"})

    def test_task_specific_ewc_key_alias_does_not_match_later_task(self):
        class Dummy(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.tensor([3.0]))

            def get_trainable_parameters(self, modality_id):
                return [self.weight]

            def get_trainable_param_names(self, modality_id):
                return ["adapter.task_5.0"]

        model = Dummy()
        manager = self.model_module.EWCManager(10.0, 1, {})
        manager.modality_task_count[1] = 2
        manager.modality_fisher[1] = {"adapter.task_1.0": torch.ones(1)}
        manager.modality_params[1] = {"adapter.task_1.0": torch.zeros(1)}
        self.assertEqual(manager.get_ewc_loss(model, 1).item(), 0.0)

    def test_release_has_no_metric_outputs_and_non_pip_requirements(self):
        release = self.result["release"]
        self.assertEqual(release["run_level_metric_files"], [])
        self.assertIn("conda", release["non_pip_requirement_entries"])
        self.assertIn("libmambapy", release["non_pip_requirement_entries"])


if __name__ == "__main__":
    unittest.main()
