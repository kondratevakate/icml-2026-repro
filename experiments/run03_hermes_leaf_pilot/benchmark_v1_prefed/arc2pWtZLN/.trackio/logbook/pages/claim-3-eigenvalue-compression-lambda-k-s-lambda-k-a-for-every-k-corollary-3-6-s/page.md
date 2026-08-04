## Claim 3 — — Eigenvalue compression lambda_k(S) <= lambda_k(A) for every k (Corollary 3.6, Sec 3.2).

- **Verdict:** `verified`  (original gate: `verified`)
- **Source:** Corollary 3.6, Section 3.2 (arXiv:2606.23517)
- **Seed:** 260925  (master 260622 + 303)
- **Why:** lambda_k(S) <= lambda_k(A) with 0 violations across 40 random instances.
- **Mutation test:** Use S_mut = A + X C^{-1} X^T (wrong sign, breaking S <= A).  [mutation breaks]  Wrong sign -> S_mut >= A, so lambda_k(S_mut) >= lambda_k(A): compression reverses.
- **Key numerics:**
  - `n_random_instances`: 40
  - `n_eigenvalue_violations`: 0
  - `min_gap`: 1.6867422749125136
  - `max_gap`: 8.015847160539826
  - `graded_all_compressed`: True

---
<!-- trackio-cell
{"type": "markdown", "id": "cell_<built-in function hash>", "created_at": "2026-01-01T00:00:00+00:00", "title": "Claim 3 \u2014 \u2014 Eigenvalue compression lambda_k(S) <= lambda_k(A) for every k (Corollary 3.6, Sec 3.2)."}\n-->
