from __future__ import annotations

import sys
import shutil
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from audit_claims import audit_c1, audit_c4  # noqa: E402


PAPER = ROOT.parent / "paper_source"
REPO = ROOT.parent / "official_pyhealth"


class AuditClaimsTest(unittest.TestCase):
    def test_c1_inventory_passes_at_pinned_release(self) -> None:
        result = audit_c1(PAPER, REPO)
        self.assertEqual(result["verdict"], "VERIFIED")
        self.assertTrue(all(result["checks"].values()))
        self.assertGreaterEqual(result["paper_totals"]["datasets"], 15)
        self.assertGreaterEqual(result["paper_totals"]["tasks"], 20)
        self.assertGreaterEqual(result["paper_totals"]["models"], 25)

    def test_c4_detects_table2_mismatch(self) -> None:
        result = audit_c4(PAPER)
        self.assertEqual(result["verdict"], "FALSIFIED")
        self.assertEqual(
            result["observed_mortality_loc"],
            {"PyHealth 2.0": 34, "PyHealth 1.16": 27, "Pandas": 51},
        )
        self.assertFalse(result["checks"]["table2_has_claimed_pyhealth_20_7"])
        self.assertFalse(result["checks"]["table2_has_claimed_pyhealth_116_24"])

    def test_c1_fails_without_genomics_exports(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            mutated_repo = Path(temp_dir) / "repo"
            shutil.copytree(REPO / "pyhealth", mutated_repo / "pyhealth")
            init_file = mutated_repo / "pyhealth" / "datasets" / "__init__.py"
            text = init_file.read_text(encoding="utf-8")
            for line in (
                "from .clinvar import ClinVarDataset\n",
                "from .cosmic import COSMICDataset\n",
                "from .tcga_prad import TCGAPRADDataset\n",
            ):
                text = text.replace(line, "")
            init_file.write_text(text, encoding="utf-8")

            result = audit_c1(PAPER, mutated_repo)
            self.assertEqual(result["verdict"], "FALSIFIED")
            self.assertFalse(
                result["checks"]["all_modalities_have_exported_witnesses"]
            )

    def test_c4_detector_flips_if_table_matches_challenge(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            mutated_paper = Path(temp_dir) / "paper"
            sections = mutated_paper / "sections"
            sections.mkdir(parents=True)
            for name in ("abstract.tex", "methodology.tex", "results.tex"):
                shutil.copy2(PAPER / "sections" / name, sections / name)

            results_file = sections / "results.tex"
            text = results_file.read_text(encoding="utf-8")
            text = text.replace(
                r"PyHealth 1.16        & 14           & \textbf{27}",
                r"PyHealth 1.16        & 14           & \textbf{24}",
            )
            text = text.replace(
                r"PyHealth 2.0         & \textbf{10} "
                r"\textcolor{green}{$\downarrow$} & 34 ",
                r"PyHealth 2.0         & \textbf{10} "
                r"\textcolor{green}{$\downarrow$} & 7 ",
            )
            results_file.write_text(text, encoding="utf-8")

            result = audit_c4(mutated_paper)
            self.assertEqual(result["verdict"], "VERIFIED")
            self.assertTrue(all(result["checks"].values()))


if __name__ == "__main__":
    unittest.main()
