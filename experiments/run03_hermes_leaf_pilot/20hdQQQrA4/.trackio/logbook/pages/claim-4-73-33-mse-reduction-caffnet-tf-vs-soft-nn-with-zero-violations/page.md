# Claim 4 — 73.33% MSE reduction (CAffNet-TF vs soft NN) with zero violations

---
<!-- trackio-cell
{"type": "markdown", "id": "cell_ac635c9e16e9", "created_at": "2026-08-02T04:30:00+00:00", "title": "Claim 4 \u2014 73.33% MSE reduction (CAffNet-TF vs soft NN) with zero violations"}
-->
*Source:* Section 4.1, Table 2; setup Appendix D.1.
*Type:* simulation (training). *Script:* `verify_claim4.py` -> `results/claim4.json`.

**Verdict: verified (CPU-scale, reduced epochs).**

Reduced-scale run: 8000 epochs (paper: 50000), 5 seeds, CPU (torch-cpu).
- CAffNet-TF MSE: 0.000993 (paper: 0.0012)
- CAffNet-TF violates: 0 (paper: 0)
- Soft NN MSE: 0.00407 (paper: 0.0045), violates: 26% of samples
- Reproduced MSE reduction: **75.58%** (paper: 73.33%)
- Mutation (removing CAffine layer): violations = 0.199 > 0, confirming layer necessity.

Note: The 73.33% figure in Table 2 was computed from paper numbers 1-0.0012/0.0045 = 73.33%; our CPU-scale run achieves close agreement with more epochs would converge toward the paper value.
