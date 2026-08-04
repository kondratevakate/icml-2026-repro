## Claim 1 — — Collapsed Effective Operator S := A - X C^{-1} X^T is the Schur complement of the graded Laplacian L* (Def 3.3, Sec 3.2).

- **Verdict:** `verified`  (original gate: `verified`)
- **Source:** Definition 3.3, Section 3.2 (arXiv:2606.23517)
- **Seed:** 260723  (master 260622 + 101)
- **Why:** Schur-complement identity matches definition/elimination/block factorisation to <1e-8; graded L* and S are PSD.
- **Mutation test:** Use S_mut = A + X C^{-1} X^T (wrong sign).  [mutation breaks]  Wrong sign -> not the Schur complement; 0<=S<=A (Claim 2) fails.
- **Key numerics:**
  - `formula_vs_elimination_residual`: 0.0
  - `formula_vs_block_factorisation_residual`: 1.5987211554602254e-14
  - `formula_vs_definition_residual`: 0.0
  - `vertex_energy_equivalence_max_err`: 3.979039320256561e-13
  - `graded_Lstar_PSD`: True
  - `graded_S_PSD`: True
  - `graded_schur_residual`: 1.014299755297543e-12

---
<!-- trackio-cell
{"type": "markdown", "id": "cell_<built-in function hash>", "created_at": "2026-01-01T00:00:00+00:00", "title": "Claim 1 \u2014 \u2014 Collapsed Effective Operator S := A - X C^{-1} X^T is the Schur complement of the graded Laplacian L* (Def 3.3, Sec 3.2)."}\n-->
