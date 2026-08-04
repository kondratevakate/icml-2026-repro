# Claim 2 — Table 1: PT-VCP reduces length 22.894 → 22.614 at α=0.10 (p=0.96), coverage preserved

---
<!-- trackio-cell
{"type": "markdown", "id": "cell_68c911f0a09e", "created_at": "2026-08-02T04:30:00+00:00", "title": "Claim 2 \u2014 Table 1: PT-VCP reduces length 22.894 \u2192 22.614 at \u03b1=0.10 (p=0.96), coverage preserved"}
-->
**Source:** Table 1, row α=0.10 / p=0.96; setting App. D.1.1. **Script:** `verify_claim2.py` → `results/claim2.json`.
**Verdict: `verified` (direction and magnitude reproduced) — with one documented specification discrepancy.**

Justifying numbers (20 seeds, sweep of 16 configurations, all with the paper's α=0.10, p=0.96):
- **All 16/16 configurations** give PT shorter than VCP with coverage 0.898–0.903 (nominal 0.90; paper reports
  0.906/0.909).
- Closest configuration (mixture components at ±10, n=1000 per fold): **VCP 22.837 ± 0.058 vs paper 22.894 ± 0.138**
  (Δ = 0.057, inside the paper's own std-err) and **PT-VCP 22.489 ± 0.063 vs paper 22.614 ± 0.254** (Δ = 0.125,
  inside the paper's std-err). Reduction reproduced: −0.35 here vs −0.28 in the paper.
- **Discrepancy:** read literally, App. D.1.1 says ε ~ N(μ,1)/N(−μ,1) with **μ = 20**, i.e. component means ±20.
  That reading gives VCP 43.520 → PT 42.547 — roughly **2× the published scale**. The published 22.894/22.614 are
  recovered only if "μ = 20" denotes the *separation* between the two mixture components (components at ±10).
  β and the fold sizes are not stated in the paper at all; the result is insensitive to β (identical lengths for
  β=(1,1) and (1,2)) and mildly sensitive to n (22.605–23.078 for n=500…5000).

**Mutation:** covered by Claim 1's mutations, which use the same generator (α′ removed ⇒ coverage breaks;
well-specified noise ⇒ PT becomes longer). No additional mutation run for this row.

---
