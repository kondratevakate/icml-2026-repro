from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from audit_dpsurv import audit  # noqa: E402


class DPsurvReleaseAuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        evidence = ROOT / "evidence" / "audit_results.json"
        if evidence.exists():
            cls.result = json.loads(evidence.read_text(encoding="utf-8"))
        else:
            cls.result = audit(
                ROOT / "official" / "repo",
                ROOT / "official" / "paper.pdf",
                ROOT / "official" / "source.tar.gz",
            )

    def test_five_cohorts_and_folds_are_released(self) -> None:
        self.assertTrue(
            self.result["checks"]["five_cohorts_and_five_folds_present"]
        )

    def test_outer_splits_have_no_case_or_slide_leakage(self) -> None:
        self.assertTrue(
            self.result["checks"]["train_test_splits_are_disjoint"]
        )

    def test_each_case_is_held_out_once(self) -> None:
        self.assertTrue(
            self.result["checks"]["each_case_appears_once_in_test"]
        )

    def test_some_released_rows_lack_dss_endpoint(self) -> None:
        self.assertFalse(
            self.result["checks"]["all_released_rows_have_dss_endpoint"]
        )

    def test_synthetic_forward_backward_executes(self) -> None:
        self.assertTrue(
            self.result["checks"]["synthetic_forward_and_backward_execute"]
        )

    def test_grfn_belief_is_bounded_by_plausibility(self) -> None:
        self.assertTrue(
            self.result["checks"]["belief_is_bounded_by_plausibility"]
        )

    def test_default_lambda_coincides_with_paper(self) -> None:
        self.assertTrue(
            self.result["checks"]["training_matches_paper_at_lambda_half"]
        )

    def test_nondefault_training_lambda_mismatch_is_reproduced(self) -> None:
        self.assertFalse(
            self.result["checks"][
                "training_matches_paper_away_from_lambda_half"
            ]
        )

    def test_nondefault_evaluation_lambda_direction_is_reversed(self) -> None:
        self.assertFalse(
            self.result["checks"][
                "evaluation_matches_paper_away_from_lambda_half"
            ]
        )

    def test_empty_component_initialization_bug_is_reproduced(self) -> None:
        self.assertFalse(
            self.result["checks"][
                "all_low_probability_component_is_supported"
            ]
        )

    def test_empirical_run_artifacts_are_absent(self) -> None:
        self.assertFalse(
            self.result["checks"]["run_level_empirical_artifacts_present"]
        )

    def test_visualization_entrypoint_shadow_bug_is_reproduced(self) -> None:
        self.assertFalse(
            self.result["checks"]["visualization_entrypoint_executes"]
        )
        self.assertEqual(
            self.result["visualization"][
                "get_panther_encoder_definition_count"
            ],
            2,
        )


if __name__ == "__main__":
    unittest.main()
