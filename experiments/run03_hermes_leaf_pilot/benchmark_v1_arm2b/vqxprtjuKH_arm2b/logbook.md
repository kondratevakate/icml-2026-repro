# Reproduction Logbook — Allocating Variance to Maximize Expectation

**Paper:** Allocating Variance to Maximize Expectation (OpenReview `vqxprtjuKH`, arXiv:2502.18463v1)
**Run dir:** `experiments/run03_hermes_leaf_pilot/vqxprtjuKH_arm2b`
**Arm:** arm2 · **Agent:** Hermes + K-Dense skills · **Model:** `tencent/hy3:free` (endpoint `http://localhost:8319/v1`)
**Hardware:** CPU only (WSL2) · **Seed:** 20260803 · **Compaction:** off

Each claim was reproduced from first principles with numpy/scipy (no GPU) via `verify_claim<N>.py`,
results stored in `results/claim<N>.json`. The per-claim verdicts below are taken from those JSON files
(`"verdict"` field) and supported with quoted evidence. Mutation tests were run for every verified claim.
No verify script was re-executed to produce this logbook.

## Summary

| # | Claim (source) | Verdict |
|---|---|---|
| 1 | Theorem 1.1, §1.2 — PTAS for independent VarAlloc (E max_i X_i ≥ OPT − ε) | **verified** |
| 2 | Theorem 1.2, §1.2 — PTAS for correlated VarAlloc (same additive ε) | **inconclusive** |
| 3 | Theorem 1.3, §1.2 — GraphVarAlloc m>1, Ω(1/log n)·OPT | **verified** |
| 4 | Theorem 1.6, §1.3 — only Θ(1/p) variables get variance Ω(p) | **inconclusive** |
| 5 | Lemma 2.1, §2.1 — small-variance O(ε√ln(1/ε)) bound | **verified** |
| 6 | Figures 1–2, §1.3 — ER-graph MC illustrations | **inconclusive** |

**Tally:** 3 verified · 0 falsified · 3 inconclusive.

---

## Claim 1 — PTAS for independent VarAlloc (Theorem 1.1, §1.2)

**Verdict: verified.**

The grid PTAS was evaluated on 15 instances (n=4,5; zero-mean, ramp, one-big, random means; eps ∈ {0.5, 0.3, 0.2}).
In every instance the algorithm's expected maximum met the additive ε guarantee relative to a fine-grid reference optimum.

Quoted evidence (`results/claim1.json`):
- `"worst_gap": 0.003382898343035734`, `"all_within_eps": true`
- Example, `random_n5`, eps=0.5: `"alg": 1.4271891972587707`, `"opt_ref": 1.4305720956018064`, `"gap": 0.003382898343035734`, `"ok": true`
- Example, `one_big_n4`, eps=0.2: `"alg": 2.0100889514991795`, `"opt_ref": 2.010108986410822`, `"gap": 2.0034911642596853e-05`, `"ok": true`. As ε shrinks the grid step (`grid_step` 0.25 → 0.09 → 0.04) the gap shrinks monotonically — consistent with a PTAS.

**Mutation test** (coarsen PTAS grid to step 0.5, ≫ ε²): `"property_breaks": true` — gaps blow up (e.g. `zero_means_n4` gap `0.11634760813466993`, `random_n5` gap `0.12712684657043472`), i.e. the claimed property collapses once the PTAS structure is destroyed. This confirms the result is the algorithm's, not an artefact.

---

## Claim 2 — PTAS for correlated VarAlloc (Theorem 1.2, §1.2)

**Verdict: inconclusive.**

The correlated PTAS (n=3,4) met the additive ε guarantee against a 120 000-sample Monte-Carlo reference for OPT, but the claim could not be *certified* because its mutation test did not cleanly break the property.

Quoted evidence (`results/claim2.json`):
- `"worst_gap": 0.0045302300672041`, `"all_within_eps": true`
- Example, n=4, eps=0.3: `"alg": 0.5945257115536037`, `"opt_ref": 0.5990559416208078`, `"gap": 0.0045302300672041`, `"ok_strict": true`
- `"mc_se": 0.00097…` — the reference itself is MC-estimated, so the ~4.5e-3 gap is well within noise but the comparison is approximate.

**Mutation test** (halve trace(Sigma) budget): `"property_breaks": false`. The halved-budget allocation still landed within the ε tolerance (e.g. n=4, eps=0.2: `"alg": 0.4235965186301658`, `"opt_ref": 0.5990559416208078`, `"gap": 0.175459422990642` — gap 0.175 < eps 0.2, so `"ok": true`). Because perturbing the setup did **not** falsify the guarantee, the test cannot distinguish the stated PTAS from a weaker result; combined with the MC-approximated reference, the claim remains **inconclusive** rather than verified.

---

## Claim 3 — GraphVarAlloc with m>1, Ω(1/log n)·OPT (Theorem 1.3, §1.2)

**Verdict: verified.**

12 GraphVarAlloc instances (n=6,8,10; m=12,16,20; random / disjoint-pairs / star / nested topologies) were solved and compared to a reference optimum. The achieved objective exceeded the `1/ln n` lower bound in every case — far better than the theorem's guarantee, as expected.

Quoted evidence (`results/claim3.json`):
- `"min_ratio": 0.5056736150091268`, `"max_ratio": 0.9999999999999999`, `"all_above_1_over_ln_n": true`
- Tightest case, `star_n8`: `"alg": 1.4189399367007576`, `"opt_ref": 2.8060391022678677`, `"ratio": 0.5056736150091268`, `"bound_1_over_ln_n": 0.48089834696298783` (ratio 0.506 > 0.481 bound).
- Random instances comfortably exceed the bound (e.g. `random_n10_q0.8`: `"ratio": 0.9974553477077462`).

**Mutation test** (worst single-variable allocation, lowest coverage): `"property_breaks": true`, `"min_mutated_ratio": 0.0` — the non-trivial allocation is required; a degenerate allocation scores 0 and falls below the bound.

---

## Claim 4 — Θ(1/p) variables receive variance Ω(p) (Theorem 1.6, §1.3)

**Verdict: inconclusive.**

The theorem is asymptotic (n,m→∞) and was tested at finite n=8,12,16 across p ∈ {0.125,…,1.0}. The optimal allocations are qualitatively consistent with concentration — few high-variance variables at large p, mass spreading across more variables as p shrinks — but the precise asymptotic *signature* (count k of Ω(p)-variance variables with k·p in a constant band, and k non-increasing in p) is **not** uniformly reproduced at these sizes.

Quoted evidence (`results/claim4.json`):
- `"all_signatures_ok": false`
- n=8, k_c25 summary: `"k_times_p": [0.667, 1.0, 1.5, 1.833, 1.5, 0.0]`, `"in_constant_band": false`, `"non_increasing_in_p": true` — k·p wanders outside a constant band.
- n=12, k_c25 summary: `"in_constant_band": true` but `"non_increasing_in_p": false` — the monotonicity signature fails here.
- Qualitative concentration is visible, e.g. n=8, p=1.0, rep=0: allocation `[0.247, 0.247, 0.247, 0.247, 5.6e-3, 5.6e-3, ~0, ~0]` with `k_c25=0, k_c50=0` (only ~4 vars carry meaningful variance), versus n=8, p=0.125, rep=0: `k_c25=5` of 8 variables above threshold.

**Mutation test** (uniform allocation instead of optimal): `"property_breaks": true` — uniform spreads variance across all n variables (k_c50 = n for small p), destroying the concentration. The direction of the theorem is supported, but because the exact constant-band asymptotic is not recovered at finite n, the verdict is **inconclusive**.

---

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

## Claim 6 — ER-graph Monte-Carlo illustrations (Figures 1–2, §1.3)

**Verdict: inconclusive.**

n=8 Erdős–Rényi graphs (p = 1/8 … 8/8, 3 graphs per p, 16 000 MC samples) were simulated for independent, positively-, and negatively-correlated Gaussians. The qualitative picture of the paper's figures is reproduced (per-set OPT is increasing in p; the optimal allocation concentrates mass onto few variables), but the **strict** concavity of per-set OPT in p and monotonic increase of concentration are not robustly confirmed.

Quoted evidence (`results/claim6.json`):
- `"all_concave": false`, `"all_concentration_increasing": false`
- Independent: `"increasing": true` (per_set_opt 0.219 → 0.524) but `"concave": false`; `"second_differences": [0.0389, -0.0655, -0.0029, 0.0009, -0.0042, -0.0087]` — includes positive (convex) segments.
- Independent `"concentration_increasing": false` (top-2 mass `[0.886, 0.744, 0.710, 0.684, 0.676, 0.606, 0.624, 0.495]` falls, not rises); negative setting does show `"concentration_increasing": true`.
- `"top2_mass"` profiles confirm heavy concentration onto 1–2 variables (e.g. positive p=0.125: `[0.754, 0.158, 0.050, …]`, top-2 = 0.912).

**Mutation test** (replace optimal allocation by uniform): `"property_breaks": true` — uniform allocation gives top-2 mass exactly 0.25 for every p and destroys the concentration pattern. The concentration effect is genuine, but the paper's figures are explicitly illustrative; at n=8 the exact concavity claim is not reproduced, so the verdict is **inconclusive**.

---

## Notes & limitations
- Claims 2 and 4 depend on Monte-Carlo references and/or asymptotic regimes; at CPU scale and finite n they could not be *certified* (mutation test non-discriminating or finite-size signature failure), hence **inconclusive** rather than falsified.
- No claim was falsified. The three **verified** claims (1, 3, 5) each passed a discriminating mutation test.
- All numbers above are quoted verbatim from `results/claim<N>.json`; no script was re-run.
