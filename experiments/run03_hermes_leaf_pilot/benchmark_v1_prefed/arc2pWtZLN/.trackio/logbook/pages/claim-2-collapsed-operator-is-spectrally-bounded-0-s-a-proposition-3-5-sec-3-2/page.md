## Claim 2 — — Collapsed operator is spectrally bounded 0 <= S <= A (Proposition 3.5, Sec 3.2).

- **Verdict:** `verified`  (original gate: `verified`)
- **Source:** Proposition 3.5, Section 3.2 (arXiv:2606.23517)
- **Seed:** 260824  (master 260622 + 202)
- **Why:** 0<=S<=A holds on all random instances (0 failures); graded S_eps PSD.
- **Mutation test:** Break Thm 3.2 PSD assumption: gamma_0 = 1.5*beta_1*sigma_min^+(B_1).  [mutation breaks]  Once L* is indefinite the Schur complement is not guaranteed PSD; 0<=S<=A can fail.
- **Key numerics:**
  - `n_random_instances`: 40
  - `n_failures_of_0leSleA`: 0
  - `min_eig_S`: 0.1662622595990091
  - `min_eig_A_minus_S`: 0.0005476355760261252
  - `graded_Lstar_PSD`: True
  - `graded_S_eps_PSD`: True

---
<!-- trackio-cell
{"type": "markdown", "id": "cell_<built-in function hash>", "created_at": "2026-01-01T00:00:00+00:00", "title": "Claim 2 \u2014 \u2014 Collapsed operator is spectrally bounded 0 <= S <= A (Proposition 3.5, Sec 3.2)."}\n-->
