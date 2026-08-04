## Claim 4 — compactness of bofop-DIDMs under the mover's distance (Corollary 5.3)

Script `verify_claim4.py` → `results/claim4.json`. Verdict: **verified**. Mutation test: yes.

Compactness is not directly testable in a finite model; **total boundedness** is (and is equivalent
for a complete metric space). Greedy ε-net over `N` sampled DIDMs, `N ∈ {10,20,40}`, distance =
exact LP W₁ on the (deg, nbr-deg) grid:

| family / metric | ε-net size at N=10 / 20 / 40 | saturates? |
|---|---|---|
| bofop, exact joint-DIDM mover's distance (ε=0.15) | **2 / 3 / 3** | yes |
| bofop, degree-marginal W₁ (ε=0.5) | 1 / 1 / 1 | yes |
| unbounded fiber (dense ER / power-law), same metric | **9 / 19 / 35** | no |

**Properness of the inclusion:** a dense-graph DIDM sits at W₁ distance **66.85 > 0** from *every*
sampled bofop-DIDM — the bofop structure is a *proper* subset.

**Mutation.** The unbounded-fiber family needs ≈ one net point per sample (35 of 40), i.e. it is not
totally bounded under the same metric. Metric note: the unbounded family does not fit a fixed
`r_max` grid, so the like-for-like comparison uses the unclipped exact 1-D degree-marginal W₁ for
both families; the bofop side is additionally certified with the full joint-DIDM LP.

**Ceiling.** Evidence, not proof — a finite ε-net is a surrogate for a topological statement.

---
<!-- trackio-cell
{"type": "markdown", "id": "cell_<built-in function hash>", "created_at": "2026-01-01T00:00:00+00:00", "title": "Claim 4 \u2014 compactness of bofop-DIDMs under the mover's distance (Corollary 5.3)"}\n-->
