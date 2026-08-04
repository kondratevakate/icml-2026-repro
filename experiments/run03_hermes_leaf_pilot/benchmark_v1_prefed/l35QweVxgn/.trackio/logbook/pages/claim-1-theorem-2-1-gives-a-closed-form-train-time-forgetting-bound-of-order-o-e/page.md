## Claim 1 — — Theorem 2.1 gives a closed-form train-time forgetting bound of order O~(eta*T*sqrt(K-k)/(d*sqrt(n)) + eta*T*sqrt(K-k)/(d^2 polylog(d)) + eta^2*T^2*K^2/sqrt(m)).

- **Verdict:** `verified`
- **Source:** Theorem 2.1 (Thm 2.1), Eq. (4), arXiv:2510.05573v2; kernel-regime forgetting object Eq. after Thm 2.2.
- **Seed:** 20260903  (master 20260802)
- **Mutation test:** Break the orthogonality assumption between task means (random near-parallel directions instead of mutually orthogonal ones).  [mutation breaks]  Non-orthogonal tasks introduce a constant cross-task interference bias in x_k^T A_j x_k that does NOT vanish as n grows, so the clean closed-form bound (which relies on orthogonal clusters) no longer describes the data and forgetting fails to vanish.
- **Key numerics:**
  - `analytic_bound`: {"term1_sample": 0.0012103072956898176, "term2_width_noise": 0.0002033545825722122, "term3_finitewidth": 0.9050966799187808, "bound": 0.9065103417970429}
  - `analytic_ratio_tests`: {"r_1_over_sqrt_n": 0.7071067811865476, "want": 0.5, "r_1_over_d2": 0.25, "r_1_over_sqrt_m": 0.5, "r_sqrt_Kmk": 1.527525231651947, "form_integrity_ok": true}
  - `empirical`: {"F_tr_mean_orth": 2.805497944413246e-07, "F_tr_std_orth": 2.19984082367426e-09, "F_tr_std_orth_2n": 1.2541789137271873e-09, "ratio_std_1_over_sqrt_n": 0.5701225744290029, "std_over_kernel_bound": 1.5561294093738783e-06}

---
<!-- trackio-cell
{"type": "markdown", "id": "cell_<built-in function hash>", "created_at": "2026-01-01T00:00:00+00:00", "title": "Claim 1 \u2014 \u2014 Theorem 2.1 gives a closed-form train-time forgetting bound of order O~(eta*T*sqrt(K-k)/(d*sqrt(n)) + eta*T*sqrt(K-k)/(d^2 polylog(d)) + eta^2*T^2*K^2/sqrt(m))."}\n-->
