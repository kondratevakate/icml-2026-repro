## Claim 3 — Non-smooth quantizers can incur H·Ω(1) regret despite O(eps_q) in-distribution error (Theorem 6, §4.1)

*Source:* arXiv:2603.20538, Section 4.1 + appendix proof of Theorem 6 (deterministic-dynamic part).
*Type:* theory (paper's exact trap-loop construction). *Script:* `verify_claim3.py` → `results/claim3.json`. *Seed:* 20260326.

**Verdict: verified.**

Parameters satisfy the stated condition `k/2 > B/(1−λ)`: `A=0.2, B=0.25, k=1, d=0.6, λ=0.45`,
`eps_q=0.05, H=300, N=4000`. The paper's trap-loop (capture set `I_P` ⇒ oscillate between
`I_T1` and `I_T2` forever) is reproduced; two predictions distinguished:

**(P1) In-distribution one-step error** (expert generates states):
- non-smooth = **0.2188**, theory `O(eps_q)` = **0.2200** (matches), binning = 0.0495.

**(P2) Deployment regret per step** (quantized policy generates states):
- non-smooth = **0.5305** (Ω(1) constant, independent of eps_q), binning = **0.0342** (O(eps_q))
- vs eps_q {0.02,0.05,0.10,0.20} → {0.638, 0.530, 0.487, 0.704} (≈ constant → Ω(1))
- total regret estimate = **159.14** = H·0.530.

**Mutation test (replace non-smooth learning-based quantizer by binning smooth quantizer):**
deployment regret/step drops from Ω(1) **0.5305** → O(eps_q) **0.0342**, in-distribution error also
O(eps_q). The H·Ω(1) failure disappears → attributable to NON-SMOOTHNESS. `passed = true`.

---

---
<!-- trackio-cell
{"type": "markdown", "id": "cell_<built-in function hash>", "created_at": "2026-01-01T00:00:00+00:00", "title": "Claim 3 \u2014 Non-smooth quantizers can incur H\u00b7\u03a9(1) regret despite O(eps_q) in-distribution error (Theorem 6, \u00a74.1)"}\n-->
