## Claim 4 — Õ(√(NT)) parametric rate (`verify_claim4.py` → `results/claim4.json`)

**Anchor:** v2 Proposition 5.5.

**Setup.** Basis-truncation EntUCB (finite-dimensional OFUL with the Eq. (7) width) on
N ∈ {2,4,8,16,32}, T ∈ {250…4000}, 60 unit-norm actions, 5 repetitions per cell (125 runs).

**Numbers.** The Prop.-5.5 bound held in **125/125** runs. Stripping the logarithmic factor,
the leading term scales as exactly √N and √T: log-log slopes `0.50000` in N and `0.50000` in T,
with bound/leading-term spread `1.064` over the whole grid. Realised regret is sublinear in T
(slope `0.589` at N = 16) and increases with N (slope `1.053` at T = 2000 — steeper than the
√N of the bound on this finite range, which is consistent with an upper bound).

**Mutation.** (A) Learner truncating to 4 of 16 coefficients with tail-heavy signal: regret
inflates (`69 → 541`, slope `0.750`). (B) Breaking the linear-feedback structure (feedback
independent of the played action) yields **linear** regret: `125.5, 247.3, 487.4, 961.1, 1925.1`,
slope `0.984`.

**Verdict: verified.** The certified Õ(√(NT)) rate is reproduced exactly at the formula level
and holds empirically on every run; mutation B shows the test is not vacuous.

---
<!-- trackio-cell
{"type": "markdown", "id": "cell_<built-in function hash>", "created_at": "2026-01-01T00:00:00+00:00", "title": "Claim 4 \u2014 \u00d5(\u221a(NT)) parametric rate (`verify_claim4.py` \u2192 `results/claim4.json`)"}\n-->
