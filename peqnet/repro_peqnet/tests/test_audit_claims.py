from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from audit_claims import (  # noqa: E402
    build_evidence,
    ltmle_lipschitz_counterexample,
    policy_embedding_check,
)


class PEQNetAuditTests(unittest.TestCase):
    def test_policy_embedding_pipeline(self) -> None:
        result = policy_embedding_check()
        self.assertTrue(result["checks"]["all_values_finite"])
        self.assertTrue(result["checks"]["zero_diagonal"])
        self.assertTrue(result["checks"]["symmetric_dissimilarity"])
        self.assertTrue(result["checks"]["policy_ordering_preserved"])

    def test_mds_does_not_establish_exact_mmd_identity(self) -> None:
        result = policy_embedding_check()
        self.assertTrue(result["checks"]["mds_is_not_exact_identity_to_mmd"])

    def test_bounded_ltmle_update_need_not_be_lipschitz(self) -> None:
        result = ltmle_lipschitz_counterexample()
        self.assertTrue(all(result["checks"].values()))

    def test_full_source_audit(self) -> None:
        source_root = ROOT.parent / "official" / "source"
        result = build_evidence(source_root)
        self.assertEqual(result["claims"]["C4"]["verdict"], "VERIFIED")
        self.assertTrue(all(result["claims"]["C5"]["checks"].values()))
        self.assertEqual(result["prepared_points"], 4)


if __name__ == "__main__":
    unittest.main()
