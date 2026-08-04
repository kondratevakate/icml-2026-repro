## Claim 5 — Small-variance contribution O(ε√ln(1/ε)) (Lemma 2.1, §2.1)

**Verdict: verified.**

For variables with Σ_ii ≤ ε² and Σ Σ_ii ≤ 1, the bound E max(0, max_i Y_i) = O(ε√ln(1/ε)) was checked across ε from 0.5 down to 0.02, plus correlated (ρ=±0.9, 0) cases. The implied constant C = E_max0 / (ε√ln(1/ε)) stays bounded and shows the log factor is genuinely needed.

Quoted evidence (`results/claim5.json`):
- Independent: `"C_max": 1.767400767674749`, `"C_min": 1.2560807032580235`, `"bounded_constant": true`, `"log_factor_needed": true`
- ε=0.5: `"E_max0": 0.5228778907418355`, `"scale_eps_sqrt_ln": 0.41627730557884884`, `"C": 1.2560807032580235`
- ε=0.02: `"E_max0": 0.06991425512673703`, `"scale_eps_sqrt_ln": 0.039557669321779544`, `"C": 1.767400767674749` — C grows only mildly as ε→0, confirming the O(·) scaling.
- Correlated: `"correlated_bounded": true` (C ∈ [1.40, 1.65] across ρ=±0.9, 0).

**Mutation test** (violate the per-coordinate cap; one variable takes the full budget): `"property_breaks": true` — C then diverges as ε→0 (ε=0.1: C=2.63; ε=0.05: C=4.61; ε=0.02: C=10.09; ε=0.01: C=18.59), showing the Σ_ii ≤ ε² cap is what bounds the contribution.

---

---
<!-- trackio-cell
{"type": "markdown", "id": "cell_<built-in function hash>", "created_at": "2026-01-01T00:00:00+00:00", "title": "Claim 5 \u2014 Small-variance contribution O(\u03b5\u221aln(1/\u03b5)) (Lemma 2.1, \u00a72.1)"}\n-->
