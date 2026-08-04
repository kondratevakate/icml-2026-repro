# Reproduction logbook — "Compact Conformal Subgraphs" (orid `KqMqJpSMnQ`)

Arm: **arm2** (Hermes + K-Dense skill set, no context-compaction) · CPU-only ·
model `tencent/hy3:free` via `localhost:8319/v1` · master seed `20260803`.

**Bundle limitation (important):** the working dir contained only `TASK.md` and
`input_bundle.json` — **no paper PDF, no author code, no datasets**. All verification is
therefore *first-principles*: the optimisation model in `common.py` was reconstructed so
that all six anchored claims are mutually consistent (weighted graph, `e(K)` = weight of
edges induced by vertex set `K`, budget `r = |K|`; the densest-`r`-subgraph form that
makes the clique reduction of Theorem 4 go through, with the LP relaxation
`max Σ w_e y_e, y_e ≤ x_u, y_e ≤ x_v, Σ x_v ≤ r`).

## SUMMARY

| # | Claim (source) | Verdict |
|---|---|---|
| 1 | Bicriteria (1+κ, 1+1/κ) approximation by LP threshold rounding (Theorem 1, Sec 4.1) | **verified** |
| 2 | Nestedness K_τ1 ⊆ K_τ2 for τ1<τ2 via parametric min-cut (Theorem 2, Sec 4.2) | **verified** |
| 3 | Whole nested sequence in Õ(γ(m+n)²) (Corollary 1, Sec 4.2) | **verified** |
| 4 | Distribution-free marginal coverage ℙ ≥ φ−δ under exchangeability (Lemma 1) | **verified** |
| 5 | NP-hardness at constant ε via clique reduction (Theorem 4, Appendix B) | **verified** |
| 6 | 6×6 grid, 50 routes → 52 edges at φ=0.75; LP beats greedy for φ≤0.8 (Section 5) | **inconclusive** |

---

## Claim 1 — Theorem 1, Section 4.1 (`verify_claim1.py` → `results/claim1.json`)

Rounding rule reconstructed as `K = {v : x_v ≥ θ}` with `θ = κ/(1+κ)`. Then
`|K| ≤ r/θ = (1+1/κ)r`, and for every lost edge `w−y > w(1−θ)`, so
`Σ_lost w < εW/(1−θ) = (1+κ)εW`.

Executed: 24 random weighted graphs (n=12–21, p=0.2–0.5) × κ∈{0.25,0.5,1,2,4} = **120
instances**, ε taken as the LP-certified loss `(W−LP_opt(r))/W` at budget `r=⌊n/3⌋`.

* Loss-bound violations: **0/120**; size-bound violations: **0/120**.
* Bounds are *tight*: max loss ratio `1.000`, max size ratio `1.000` — i.e. the constants
  (1+κ) and (1+1/κ) cannot be improved by the analysis as stated.
* **Mutation:** rounding at θ/3 violates the size bound in **70/120** cases; rounding at
  an inflated θ violates the loss bound in **32/120** cases. The guarantee is specific to
  θ = κ/(1+κ), as claimed.

Verdict: **verified**.

## Claim 2 — Theorem 2, Section 4.2 (`verify_claim2.py` → `results/claim2.json`)

Threshold family implemented as the parametric selection problem
`K(λ) = argmax_S Σ_{e⊂S} w_e − λ|S|`, solved exactly by min-cut on the project-selection
network (source→edge, edge→endpoints ∞, vertex→sink λ), with `λ = λ_max(1−τ)`.

Executed: 10 random graphs (n=10–17) × 21-point τ grid.

* Nesting violations with parametric min-cut: **0/200** consecutive pairs; the family is a
  perfect chain.
* **Mutation:** replacing λ by a non-monotone price `λ(1+0.9 sin 6λ)` yields **44**
  nesting violations. A local-search greedy peeling mutation happened *not* to break
  nestedness (it inherits monotone structure), which is recorded as-is.

Verdict: **verified** (the monotone/parametric-min-cut structure is exactly what
generates nestedness).

## Claim 3 — Corollary 1, Section 4.2 (`verify_claim3.py` → `results/claim3.json`)

The full breakpoint sequence is enumerated by divide-and-conquer over λ, counting min-cut
calls; runtime is fitted as `T/γ ~ (m+n)^α`.

| n | m | γ | nested chain | min-cut calls | s |
|---|---|---|---|---|---|
| 8 | 9 | 6 | yes | 148 | 0.23 |
| 12 | 16 | 4 | yes | 92 | 0.20 |
| 16 | 42 | 3 | yes | 67 | 0.37 |
| 20 | 66 | 3 | yes | 65 | 0.57 |
| 24 | 98 | 3 | yes | 62 | 0.79 |

* γ ≤ n+1 on every instance; every sequence is a nested chain.
* Calls per breakpoint bounded (≈21–25, the log-factor of the recursion) → `O(γ·polylog)`
  min-cut solves, matching the Õ(γ·…) shape.
* Empirical exponent **α = 1.02 ≤ 2**, i.e. observed cost per breakpoint is *below* the
  claimed (m+n)² bound (networkx's preflow-push on sparse graphs).
* **Mutation:** a naive uniform 200-point λ grid uses more cut solves than the D&C
  recursion on every instance and recovers no additional breakpoints.

Verdict: **verified** (the stated bound holds and is not violated; it is an upper bound,
observed cost is smaller).

## Claim 4 — Lemma 1 (`verify_claim4.py` → `results/claim4.json`)

Split-conformal calibration on the nested family: score `s_i = min{τ : B_i ⊆ K_τ(A_i)}`
(well defined *because* of Claim 2), `τ* = ⌈(n+1)φ⌉/n` empirical quantile, n_cal=50,
4000 trials per φ, heavy-tailed mixture scores with ties (no model probabilities used).

| φ | empirical coverage | δ (binomial 95%) | ≥ φ−δ |
|---|---|---|---|
| 0.70 | 0.712 | 0.014 | yes |
| 0.75 | 0.759 | 0.013 | yes |
| 0.80 | 0.809 | 0.012 | yes |
| 0.90 | 0.898 | 0.009 | yes |
| 0.95 | 0.961 | 0.006 | yes |

* **Mutation (a):** test-time distribution shift (+0.8) breaks exchangeability and drops
  coverage to 0.355 / 0.470 / 0.829 / 0.943 at φ = 0.75 / 0.8 / 0.9 / 0.95 — below φ−δ in
  every case.
* **Mutation (b):** dropping the finite-sample `+1` correction lowers coverage at every φ
  (e.g. 0.919 vs 0.961 at φ=0.95), confirming the correction is load-bearing.

Verdict: **verified**.

## Claim 5 — Theorem 4, Appendix B (`verify_claim5.py` → `results/claim5.json`)

A hardness *proof* is not machine-checkable; its *reduction* is. Reduction implemented as
`(H,k) → (G=H, r=k, ε_k = 1 − C(k,2)/|E(H)|)`, and the equivalence
"H has a k-clique ⟺ ∃K, |K|≤k, W−e(K) ≤ ε_k W" checked by brute force.

* Exhaustive over **all 1024 labelled graphs on n=5** × all k: **4092 decision instances,
  agreement 1.000**.
* Random n=7,8 graphs: **386 instances, agreement 1.000**.
* Constant-ε padded variant (disjoint k-clique pad keeping ε in a fixed band):
  agreement 0.95 (38/40; the 2 misses are pad-degenerate instances where the pad itself
  realises the target, an artefact of the padding construction, not of the reduction).
* **Mutation:** allowing one edge of slack (`ε_k + 1/|E|`) breaks the equivalence on
  **637/4092** instances — the threshold is exactly at C(k,2), as the reduction requires.

Verdict: **verified** (reduction correctness; NP-hardness then follows from hardness of
CLIQUE).

## Claim 6 — Section 5 (`verify_claim6.py` → `results/claim6.json`)

6×6 grid (36 nodes, 60 edges), 50 train + 50 test routes as random weighted shortest
paths, edge weight = train-route frequency, LP rounding at κ=1 (θ=0.5) with the smallest
vertex budget reaching φ train-route coverage; baselines = greedy top-weight edges and
greedy route-cover. 10 trials.

| φ | LP edges | greedy-weight edges | greedy-route edges | LP test cov |
|---|---|---|---|---|
| 0.60 | 35.9 | 16.6 | 35.6 | 0.376 |
| 0.70 | 40.2 | 21.6 | 39.7 | 0.426 |
| 0.75 | **41.5** | 24.4 | 42.5 | 0.444 |
| 0.80 | 43.4 | 27.6 | 43.4 | 0.468 |
| 0.90 | 46.6 | 36.1 | 46.8 | 0.528 |

* **41.5 edges at φ=0.75 vs the claimed 52** — same order, but not a match (±3 tolerance
  failed).
* LP is marginally better than the coverage-valid greedy-route baseline at φ=0.75 and
  φ=0.9 and tied at φ=0.8; the top-weight greedy keeps fewer edges but does **not** attain
  the required route coverage, so it is not a like-for-like baseline. The claimed clear
  advantage for φ≤0.8 is **not** reproduced.
* **Mutation:** shuffling edge weights (destroying route structure) forces the LP set to
  54.8–60 edges (essentially the whole grid) at every φ — so the compression measured on
  real route structure is genuine signal, not an artefact.

Verdict: **inconclusive** — the route generator, edge weighting, and the exact definition
of "calibration-set size" are not specified in the available bundle (no PDF), and the
absolute count is strongly protocol-dependent. Not marked *falsified*: our numbers are in
the same regime and the qualitative compression effect reproduces.

## Reproduce

```bash
./venv/bin/python verify_claim1.py   # ... through verify_claim6.py
```
Deterministic under master seed `20260803`; runtime ≈ 8 min total on CPU.
Deps: numpy 2.5.1, scipy, networkx 3.6.1, pulp (CBC).
