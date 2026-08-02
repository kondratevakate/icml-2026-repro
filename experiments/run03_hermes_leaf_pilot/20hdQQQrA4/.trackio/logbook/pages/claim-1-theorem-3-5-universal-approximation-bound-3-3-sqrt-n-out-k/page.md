# Claim 1 — Theorem 3.5 (universal approximation, bound (3 + 3 sqrt(n_out)) K)

---
<!-- trackio-cell
{"type": "markdown", "id": "cell_8b731c14aee2", "created_at": "2026-08-02T04:30:00+00:00", "title": "Claim 1 \u2014 Theorem 3.5 (universal approximation, bound (3 + 3 sqrt(n_out)) K)"}
-->
*Source:* Theorem 3.5 (Sec. 3.2); proof Appendix C, Eqs. (23), (28), (31).
*Type:* theory, numerically checkable. *Script:* `verify_claim1.py` -> `results/claim1.json`.

**Verdict: verified.**

Method: exact numpy implementation of Eqs. (8), (9), (12); instances constructed to satisfy the
proof's hypotheses (`f_t` feasible, `||f_theta − f_t||_p < K`, `||w_phi||_p < 2K`, K = 0.01).
Exhaustive grid: n_out in {1,2,3,4} x m = n_out + {0,1,3} x {0,1,2} forced linearly dependent
rows x p in {1, 1.5, 2, 3} x 25 seeds = **3200 instances** (never a single seed); half of them are
built with `f_t` on an active constraint so the projection branch is actually exercised
(**807 Case-2 instances** where `f_theta` is infeasible).

Numbers (`results/claim1.json`, `faithful`):
- violations of `||P* − f_t||_p < (3 + 3 sqrt(n_out)) K`: **0 / 3200**
- worst observed ratio `||P* − f_t||_p / eps`: **0.265** (the bound is not tight — it holds with a ~3.8x margin)
- violations of the intermediate steps: `||P* − f_theta||_p < (2+3 sqrt n)K`: 0; the Case-2 step
  `min_gamma ||f_t − P_gamma||_p < (1+3 sqrt n)K`: 0/807; Eq. (31) matrix-norm bounds
  `||A_g^+ A_g||_p, ||I − A_g^+ A_g||_p <= sqrt(n_out)`: 0 violations; Lemma 3.3 (a feasible
  candidate always exists): 0 failures.

Mutation tests (both must and do break the bound):
- **M1** — violate premise (28) by drawing `||w_phi||_p = 50 K`: **617 / 3200** violations, worst ratio **6.65**.
- **M2** — replace `argmin` by `argmax` in Eq. (12): **728 / 3200** violations, worst ratio **357**.

So the bound is not an artefact of the instance generator: it depends on both the null-space-norm
premise and on the minimum-distance selection rule.
