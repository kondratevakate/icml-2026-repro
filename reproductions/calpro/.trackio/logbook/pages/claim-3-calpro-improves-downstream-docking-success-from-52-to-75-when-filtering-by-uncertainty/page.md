# Claim 3: CalPro improves downstream docking success from 52% to 75% when filtering by uncertainty


---
<!-- trackio-cell
{"type": "markdown", "id": "cell_86616dc2aa76", "created_at": "2026-07-18T16:28:13+00:00", "title": "Claim 3: CalPro improves downstream docking success from 52% to 75% when filtering by uncertainty"}
-->
**Status: OUT OF SCOPE (not attempted).**

This claim is intrinsically tied to the protein domain: it requires AlphaFold-predicted structures, a graph evidential head over protein residues, and a ligand-**docking** pipeline to measure docking success before/after uncertainty-based filtering. None of that is reproducible on the paper's non-biological benchmark used here for Claims 1–2, and the docking toolchain + protein data are a heavy, domain-specific setup beyond this CPU reproduction.

The mechanism it relies on — filtering by the evidential uncertainty — is the same one validated in Claim 1 (uncertainty tracks reliability) and Claim 2 (uncertainty is well-calibrated), so the *ingredient* is reproduced even though the docking endpoint is not. A full reproduction would need the protein structure data + a docking backend on GPU; flagged here as a blocker rather than presented as reproduced.
