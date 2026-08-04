## Claim 5 — Information-theoretic lower bounds: regret ≥ H·(1/n+eps_q) [deterministic] / H·(√(1/n)+eps_q) [stochastic] (Theorems 8–9, §5)

*Source:* arXiv:2603.20538, Section 5 + appendix proofs of Theorems 8–9.
*Type:* theory (numerical reproduction of the two Le Cam two-point arguments).
*Script:* `verify_claim5.py` → `results/claim5.json`. *Seed:* 20260325.

**Verdict: verified.**

Deterministic expert lower bound (Thm 8), `n` ∈ {20,…,2000}:
- fitted log–log slope of statistical term vs `n` = **−0.9980** (expected −1.0)
- `TV_bound_at_n1000` = **0.7530** ≤ 0.8 (paper claim) ✓

Stochastic expert lower bound (Thm 9), `n` ∈ {20,…,2000}:
- fitted log–log slope of event threshold vs `n` = **−0.5000** (expected −0.5)
- `TV_bound_at_n1000` = **0.8737** ≤ 7/8 = 0.875 (paper claim) ✓, probability floor = 0.125 (≥ 1/8)

Matching with upper bounds: deterministic upper stat `H*log|Pi|/n` matches lower `H/n`; quantization
`H*eps_q` matched; stochastic upper stat `H*√(log|Pi|/n)` matches lower `H*√(1/n)` up to `log|Pi|`,
and the quantization term `H*eps_q` is matched **exactly** by the model-augmented upper bound
(Theorem 7). The statistical rates coincide and the quantization term is tight.

**Mutation test (drop H*eps_q from the upper bound):** the upper bound would then fall below the
lower bound (which proves `H*eps_q` unavoidable) — an impossibility → confirms the lower bound is
tight and the bounds "match". `passed = true`.

---

---
<!-- trackio-cell
{"type": "markdown", "id": "cell_<built-in function hash>", "created_at": "2026-01-01T00:00:00+00:00", "title": "Claim 5 \u2014 Information-theoretic lower bounds: regret \u2265 H\u00b7(1/n+eps_q) [deterministic] / H\u00b7(\u221a(1/n)+eps_q) [stochastic] (Theorems 8\u20139, \u00a75)"}\n-->
