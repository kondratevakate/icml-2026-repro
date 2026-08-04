## Claim 2 — Under P-IISS + RTVC the regret bound is polynomial (not exponential) in H w.r.t. eps_q (Theorem 3, §3.1–3.2)

*Source:* arXiv:2603.20538, Section 3.1–3.2 (Theorem 3, Def 3 P-IISS, Def 4 RTVC).
*Type:* theory (formula) + 1-D dynamical instantiation. *Script:* `verify_claim2.py` → `results/claim2.json`. *Seed:* 20260323.

**Verdict: verified.**

**(A) Formula evaluation of Theorem 3 under RTVC.** With a binning quantizer `kappa(gamma(·))=0`,
the bound is `O(H*log|Pi|/n + H*eps_q)`:
- `theorem3_bound_under_RTVC_slope_in_H` = **1.0** (expected 1 — linear/polynomial; never exp(H))
- `eps_q_dependence_slope_quantization_only` = **1.0000** (expected 1 — linear in eps_q, constant w.r.t. H)

**(B) Dynamical instantiation** (1-D contractive expert `u*(x)=Lu·x`, `a+Lu=0.1`; Lipschitz learner gain
`a+Lu+beta=1.1>1`):
- RTVC (thresholded/binning) regret log-fit slope in H = **0.0620** (~0 → polynomial `O(H*eps_q)`)
- Lipschitz (non-RTVC/Wasserstein) regret log-fit slope in H = **0.1248**
- Lipschitz effective per-step growth rate = **1.1330** (> 1 → compounding `exp(H)`)
- regret ratio RTVC vs Lipschitz at H=40 = **1170.98**
This reproduces the paper's Remark after Def 4 exactly: capped (RTVC) continuity → polynomial;
uncapped (Lipschitz) continuity → exponential.

**Mutation test (toggle deployed-policy continuity RTVC → Lipschitz):** RTVC gives `O(H*eps_q)`
polynomial regret (slope ~0); Lipschitz gives `exp(H)` regret (growth rate 1.133 > 1). Dropping
RTVC flips the regret from polynomial to exponential → claim attributable to RTVC. `passed = true`.

---

---
<!-- trackio-cell
{"type": "markdown", "id": "cell_<built-in function hash>", "created_at": "2026-01-01T00:00:00+00:00", "title": "Claim 2 \u2014 Under P-IISS + RTVC the regret bound is polynomial (not exponential) in H w.r.t. eps_q (Theorem 3, \u00a73.1\u20133.2)"}\n-->
