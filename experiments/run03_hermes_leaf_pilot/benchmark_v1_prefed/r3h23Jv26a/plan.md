# plan.md

Start: 2026-08-02T00:56Z+04. Budget: soft 4h, hard 8h, ≤2h/claim.

| Claim | Type | CPU-feasible? | Approach |
|---|---|---|---|
| 1 — Alg.1 + Thm 6: valid marginal coverage + shorter length | theory + small simulation | YES | (a) sympy identity p(1−α')=1−α; (b) exact/analytic coverage of PT over exhaustive (α,p) grid; (c) finite-sample Monte-Carlo VCP+PT over 20 seeds; (d) MUTATION: drop the α'-adjustment (use α inside the branch) ⇒ coverage must fall to ≈p(1−α) < 1−α; MUTATION 2: Gaussian-score DGP (Example 3) ⇒ length must INCREASE. |
| 2 — Table 1: 22.894 → 22.614 at α=0.10, p=0.96 | simulation | YES | Re-implement App. D.1.1 DGP exactly as specified. β and split sizes are NOT given in the paper ⇒ sweep a small grid, 20 seeds; report both the directional result (PT < VCP) and whether the absolute numbers 22.894/22.614 are recoverable. |
| 3 — Table 2: 9/10 real datasets shorter at 90% coverage | data + NN training | PARTIAL | Download the freely-downloadable UCI subsets (concrete, bike, bio/CASP); train the D.2.1 MLP on CPU; run VCP vs PT-VCP, 5 seeds. MEPS19–21 require AHRQ data-use registration + the CQR repo's preprocessing; blog/facebook/star are large or Dataverse-gated. ⇒ partial evidence only; the "9 of 10" statement itself will be `inconclusive` unless all 10 are run. NO toy substitutes. |
| 4 — Def.1 + Prop.2: IS(C_PT) = p(1−p)(E L)² > 0 | theory + simulation | YES | Symbolic derivation of the two-point-mixture variance in sympy; Monte-Carlo estimate of IS over exhaustive p grid vs closed form; MUTATION: p=1 (no PT) ⇒ IS = 0; MUTATION 2: deterministic VCP ⇒ IS = 0. |
| 5 — Prop.1: PT is a special case of localized CP; IS flags it | theory + simulation | YES | Implement localized CP with normalized score S/σ̂ where σ̂ ∈ {ε→0⁺, 1} w.p. {1−p, p}; show interval-by-interval equivalence to PT as ε→0; show IS flags it (>0) while deterministic localized CP with a fixed σ̂ has IS = 0. MUTATION: fixed σ̂ ≡ 1 ⇒ localized CP = VCP, IS = 0, no length gain. |

Order: 1, 4, 5 (pure CPU/theory) → 2 (simulation, number-matching) → 3 (data, capped at 2h).
