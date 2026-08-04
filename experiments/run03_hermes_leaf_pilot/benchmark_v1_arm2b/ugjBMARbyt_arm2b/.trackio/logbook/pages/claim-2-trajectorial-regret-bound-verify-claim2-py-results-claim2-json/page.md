## Claim 2 — Trajectorial regret bound (`verify_claim2.py` → `results/claim2.json`)

**Anchor:** v2 Theorem 4.1.

**Setup.** A genuine finite BOT instance: µ = ν = uniform on m = 4 points, so Π(µ,ν) is the
Birkhoff polytope whose extreme points are the 4! = 24 permutation matrices (a linear
objective is minimised at an extreme point; the Kantorovich value is the assignment optimum).
Plans are embedded as basis coefficients of dπ/dϱ (d = 16); the embedding identity
⟨f*|a_π⟩ = Σ c*_ij π_ij holds to `1.1e−16`. EntUCB/OFUL optimism with the Eq. (7) width,
λ = 1, σ = 0.2, δ = 0.05, 60 seeds, T = 400.

**Numbers.** mean realised regret `51.29`, mean certified Thm-4.1 bound `836.04`
(ratio `0.061`); the bound held in **60/60** runs (required ≥ 0.95). Uniform-in-time
coverage of f* by C_t(δ): **1.00**. Sublinearity across T ∈ {100, 400, 1600, 6400}:
per-step regret falls from `0.16` to `0.049`; log-log slopes — regret `0.757`,
certified bound `0.762` (both inflated above 1/2 on this finite range by the log factors).

**Mutation.** Replacing the σ-sub-Gaussian noise by heavy-tailed Cauchy noise (same declared σ)
destroys the assumption behind the theorem: confidence-set coverage collapses from `1.00` to
`0.00` and mean regret rises to `78.5`.

**Verdict: verified** (finite-dimensional discrete instantiation). Caveat: the entropic
approximation term of Thm 4.1 is absent here because all Birkhoff plans already lie in
Π_H(µ,ν); that term is tested separately in Claim 3. The certified bound is loose by ~16×
on this instance, so the test confirms validity, not tightness.

---
<!-- trackio-cell
{"type": "markdown", "id": "cell_<built-in function hash>", "created_at": "2026-01-01T00:00:00+00:00", "title": "Claim 2 \u2014 Trajectorial regret bound (`verify_claim2.py` \u2192 `results/claim2.json`)"}\n-->
