## Claim 1 — Theorem 1 (Section 2), finite-sample uniform validity

**Statement tested.** `Ĉ(X_{n+1}) = ∪_k Ĉ^{(k)}(X_{n+1})`, and for a test point from **any**
mixture `P = Σ_k π_k P^{(k)}`, `P(Y_{n+1} ∈ Ĉ(X_{n+1})) ≥ 1 − α`, for arbitrary conformity scores.

**Script.** `verify_claim1.py` → `results/claim1.json` (runtime 170.8 s).
Discrete world, K = 3 sources, |X| = 4, |Y| = 20, n_cal = 50 per source, α = 0.1,
8 independent worlds × 4000 draws each, p-values `(1 + #{S_i ≥ s})/(n+1)`.

**Numbers.**
- worst-per-source coverage: **min 0.9767**, mean 0.9861 over 8 worlds; **0/8 worlds below 1 − α = 0.9**.
- Exhaustive over the mixture simplex: coverage under a mixture is the π-weighted average of the
  per-source coverages, so the minimum is attained at a vertex; checked directly on **968** weight
  vectors (11×11 grid × 8 worlds) → **worst mixture coverage 0.9767**.
- Union identity `Ĉ = ∪_k Ĉ^{(k)}`: **0 mismatches** out of 32 000 test points.
- Mean set size 19.20 / 20 labels → the aggregated set is *valid but very conservative*, exactly the
  "Baseline-agg is conservative" behaviour the paper motivates its method with.

**Mutation tests.** (all: same worlds/seeds, one mechanism changed)
| mutation | worst-source coverage (min / mean) | predicted | observed |
|---|---|---|---|
| drop the finite-sample `+1` correction | 0.9662 / 0.9801 | strictly lower coverage | ✔ lower everywhere, still above 0.9 at this conservativeness level |
| mean-p instead of max-p | 0.9520 / 0.9641 | loses the guarantee's slack | ✔ drops by 3.4 points, smaller sets (18.15) |
| **single-source calibration** (use only source 0's p-value) | **0.5514 / 0.6460** | severe under-coverage on the other sources | ✔ **8/8 worlds below 0.9** — the max over sources is the mechanism, not the score quality |

**Verdict: verified.** Coverage never fell below 1 − α anywhere in the design; deleting the max-over-sources
step destroys validity by ~35 coverage points.

---

---
<!-- trackio-cell
{"type": "markdown", "id": "cell_<built-in function hash>", "created_at": "2026-01-01T00:00:00+00:00", "title": "Claim 1 \u2014 Theorem 1 (Section 2), finite-sample uniform validity"}\n-->
