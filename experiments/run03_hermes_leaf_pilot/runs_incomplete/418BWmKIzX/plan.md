# plan.md — reproduction plan (418BWmKIzX, theory paper, CPU only)

All 6 anchored claims are THEOREMS (bounds / lower bounds / low-degree hardness). No datasets
or GPU. Verdicts: verified (scaling/construction reproduced + mutation), toy (scaling only,
constant factors not asserted), or inconclusive (cannot falsify a conjecture).

| # | Claim | Type | CPU feasible? | Approach |
|---|---|---|---|---|
| 1 | Thm 1.1 err ≤ η+Õ(Δ^{1/3}/γ) | algorithmic sim | YES | Implement Alg 1+2 exactly; simulate drifting halfspace + Massart noise on sphere; measure true excess error (analytic via angle formula) vs Δ (exp 1/3) and γ (exp −1); multiple seeds. Mut: (M1) Δ=0 → excess→0; (M2) no epoching (W=T) → excess violates bound (drift accumulates). |
| 6 | Thm 3.2 err=Õ(√Δ γ^{-3/2}) | algorithmic sim | YES | Realizable variant (no noise), gradient g=(sign(w·x)−y)x, W=(γΔ)^{-1/2}. Measure exp vs Δ (1/2) and γ (−3/2). Mut: HL94 epoch W=Δ^{-1/2} → γ-exp degrades to −2; add label noise → error jumps. |
| 3 | Thm 2.1 ERM excess Õ(√(dΔ)) | sim + analytic | YES | Run DriftedERM (Alg 3) over tractable VC-d class (halfspaces on {±1}^d, ERM by enum). Fit excess vs (dΔ) → 1/2. Numerically confirm √(dΔ) < (dΔ)^{1/3} for dΔ<1 (the "better than adversarial" statement). Mut: adversarial (not Massart) label noise → excess rises to ~(dΔ)^{1/3}. |
| 4 | Thm 2.2 Ω(√(dΔ)) lower bound | hard-instance sim | YES (constructive) | Build the B.2 hard family; run natural learner (ERM over d-halfspace class) and measure forced excess error; confirm Ω(√(dΔ)) and match upper bound (optimality Θ(√(dΔ))). Mut: (a) Δ→0 → bound vanishes; (b) violate (1-2η)^3>dΔ → forced excess drops below √(dΔ). |
| 5 | Thm 4.1 low-degree degree<O(γ^{-c/4}) | sim (concrete) | YES | Build Def 4.3 trajectory-testing instance; fit actual degree-k polynomial distinguisher over H0/H1 samples on the d-hypercube; compute advantage=|μ_H1−μ_H0|/σ_H0 vs k; show it stays <1 for k<γ^{-c/4} and crosses 1 beyond. Mut: increase signal Δ → threshold degree drops; null signal → no distinguisher at any k. |
| 2 | Thm 1.2 Ω(Δ^{1/3}) gap under LD conjecture | analytic + sim | YES (conditional) | Corollary of Claim 5. Show info-optimal excess ~Δ^{1/2} (from Claim 3/4 family) vs low-degree (computational) excess ~Δ^{1/3} (from Claim 5). Gap = computation, not statistics. Verdict conditional on LD conjecture. Mut: remove LD restriction (unrestricted learner) → reach Δ^{1/2}. |

Priority: 5 (concrete, cheap) → 1,6 (algo sim) → 3,4 (info-theoretic) → 2 (ties to 5).
Every verified claim gets ≥5 seeds (never single seed) and a mutation test.
Static check of exponent relation √(x)<x^{1/3} for x∈(0,1) via sympy (x<1 ⇒ x^{1/2}<x^{1/3}).

Deliverables: verify_claim<N>.py, results/claim<N>.json, logbook.md, check_reproducibility.py, this plan.md, notes_paper.md.
Time budget: soft 4h. Track elapsed in logbook.md.
