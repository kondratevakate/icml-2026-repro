## Claim 4 — — Theorem 2.3 bounds the delayed generalization gap by eta*T*exp(eta*T*(K-k+1)/sqrt(m))/n for Lipschitz, smooth losses; gap decays with n.

- **Verdict:** `verified`
- **Source:** Theorem 2.3 (Thm 2.3), Eq. after Thm 2.3, arXiv:2510.05573v2.
- **Seed:** 20261206  (master 20260802)
- **Mutation test:** Insufficient width: m=d^2 (below d^8 K^4).  [mutation breaks]  With m=d^2 the exponent eta*T*(K-k+1)/sqrt(m) is huge, so the exponential width penalty makes the gap explode instead of decaying with n; width is necessary.
- **Key numerics:**
  - `scaling_tests`: {"ratio_1_over_n": 0.5, "want": 0.5, "ratio_linear_in_etaT": 2.0000070576058393, "scaling_ok": true}
  - `regime`: {"G_by_d": {"16": 0.09017989548252739, "32": 0.07213621826086634, "64": 0.06011250549480985, "128": 0.05152485628452731}, "exponent_by_d": {"16": 0.00012703668096958055, "32": 2.032586895513289e-05, "64": 3.5287966935994606e-06, "128": 6.48146331477452e-07}, "G_decreases": true, "exponent_to_zero": true}
  - `contrast_claim5`: "Theorem 2.3 is LINEAR in eta*T (and exponential in eta*T/sqrt(m)); Claim 5's improved bound (Thm B.1) is poly-logarithmic in T for self-bounded losses."

---
<!-- trackio-cell
{"type": "markdown", "id": "cell_<built-in function hash>", "created_at": "2026-01-01T00:00:00+00:00", "title": "Claim 4 \u2014 \u2014 Theorem 2.3 bounds the delayed generalization gap by eta*T*exp(eta*T*(K-k+1)/sqrt(m))/n for Lipschitz, smooth losses; gap decays with n."}\n-->
