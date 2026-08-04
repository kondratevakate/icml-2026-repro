# Reproduction logbook — tRsnpaRO0m

**Paper:** *A Graphop Analysis of Graph Neural Networks on Sparse Graphs: Generalization and Universal Approximation*
**OpenReview:** https://openreview.net/forum?id=tRsnpaRO0m · arXiv 2602.08785 · Area: Deep Learning
**Agent:** arm1 (autonomous) · **Hardware:** CPU only (WSL2), numpy 2.5.1 / scipy 1.18.0 / sympy 1.14.0
**Global seed:** `20260802` (pinned in every script; per-claim derived seeds recorded in each JSON)
**Artifacts:** `verify_claim<N>.py`, `results/claim<N>.json`, venv at `.venv/`
**Reproduce:** `python3 -m venv .venv && .venv/bin/pip install numpy scipy sympy && for i in 1 2 3 4 5 6; do .venv/bin/python verify_claim$i.py; done`

## Nature of this paper and what "reproduce" can mean

This is a **pure theory paper**: all six anchored claims are definitions, a theorem, a
corollary, and two derived results. There are no released artifacts, no datasets and no
reported numbers to match. A CPU reproduction therefore cannot re-derive the proofs; what
it *can* do — and what was done — is build a faithful **finite-dimensional model** of each
statement and test whether the asserted property actually holds there, plus a **mutation
test** that breaks the hypothesis and must break the conclusion. Where this is evidence
rather than proof, the JSON `note` field says so explicitly.

## Summary

| # | Source | Verdict | Key evidence | Mutation (must break) |
|---|--------|---------|--------------|-----------------------|
| 1 | Definition 3.1 (graphops) | **verified** | self-adjointness err 2e-18 (dense) / 1e-16 (sparse); zero positivity violations; finite L∞→L¹ norm for both | antisymmetric kernel → self-adj err 1.4e-2 ✓; signed kernel → positivity violation 0.28 ✓ |
| 2 | Definition 3.1 (bofops) | **verified** | ess sup ν_x(Ω) = 4 constant over n=100…800 (log-log slope −2e-16); dense ER slope 0.98 | +hub of degree √n → slope 0.43, bound destroyed ✓ |
| 3 | Theorem 4.1 | **verified** | empirical Lipschitz ratio max 5.16 ≤ C_theory 22.76; bounded across signal scales 1→64 | cubic (non-Lipschitz) activation → ratio 1.3e-9 → 5.3e+38 across the same scales ✓ |
| 4 | Corollary 5.3 | **verified** | greedy ε-net (ε=0.35, exact LP mover's distance) saturates at 5 for N=20→40; dense extreme DIDM at distance 0.124 > 0 from every bofop DIDM | drop degree bound → net size 10/20/40, never saturates ✓ |
| 5 | Section 6.1 | **verified** | sup-norm test error on a DIDM-continuous target falls 0.0317 → 0.0115 with MPNN width 4→256 (1.2% of target range) | DIDM-discontinuous target → sup error plateaus at ≈0.52 ✓ |
| 6 | Section 6.2 | **verified** (with caveat) | gap \|test−train\| MSE 2.75e-3 → 2.22e-4 over m=50→800, log-log slope **−0.96** (beats the −1/2 the covering argument predicts) | heavy-tailed degrees → gap 2–11× larger at every m ✓ (but *rate* not degraded — see caveat) |

**6/6 verified**, each with a passing mutation test. No claim was falsified; none required GPU.

## Per-claim detail

### Claim 1 — graphops are self-adjoint, positivity-preserving, and unify dense + sparse
Ω = [0,1] discretised into n=400 uniform atoms; (Af)(x) = ∫W(x,y)f(y)dμ(y) → (1/n)·Wf.
Tested a dense graphon kernel `0.5(1+cos 2π(x−y))` and a sparse ~3-regular graph mapped
to the same space via W = n·A. Over 200 random probe pairs: ⟨Af,g⟩−⟨f,Ag⟩ ≤ 1.1e-16 and
no negative output for non-negative input, for **both**. That is the "single class of limit
objects" content of Def. 3.1, and it holds. Mutations target one property each and each
breaks only its own property (the signed-kernel mutation stays self-adjoint at 3e-18 —
a good sign the tests are independent).

### Claim 2 — bofops via ess sup_x ν_x(Ω) < ∞
Under the n-interval embedding the fiber mass ν_x(Ω) is exactly the vertex degree, so
the definition becomes a max-degree bound and is directly measurable. Bounded-degree
family: 4, 4, 4, 4 across n = 100/200/400/800 (growth exponent −2e-16). Dense ER p=0.5:
163 → 1242, exponent 0.98 (≈ linear in n) — outside the class. The √n-hub mutation gives
exponent 0.43, confirming a *single* hub suffices to leave the bofop class. The definition
discriminates exactly as claimed.

### Claim 3 — Theorem 4.1, MPNN Lipschitz w.r.t. the action metric d_M
d_M = ‖f₁−f₂‖₂ + ‖A₁−A₂‖_{2→2}, with the operator-norm term computed **exactly by SVD**
(true sup over the unit ball, not a sampled approximation). MPNN: 3 layers of
h ← tanh(0.6h + 0.5·Ah) on max-degree-4 bofops, n=120, 300 random nearby pairs.
Empirical max ratio 5.16 against the layerwise constant (a+b·r)^D + b·Σ(a+b·r)^k = 22.76 —
the inequality holds with room, i.e. the theorem's form is right and the crude constant is
loose (expected; the paper's C′_{D,r} is not stated numerically, so only the structural
inequality is checkable).
*Honest note:* the tanh ratio **decreases** with signal scale (4.21 → 0.56) rather than
staying flat, because tanh saturates. My first pass had a "scale-invariance" criterion that
this correctly failed; that criterion was wrong — Lipschitz continuity requires the ratio be
*bounded*, not constant — so it was replaced with a boundedness criterion. The cubic-activation
mutation blows up by 47 orders of magnitude over the same sweep, so the test is discriminative.

### Claim 4 — Corollary 5.3, compactness of bofop-DIDMs under the mover's distance
DIDM = pushforward of the vertex measure under v ↦ (deg v, neighbour-degree distribution),
represented as a joint histogram on the compact grid [0,r]², r=6. Mover's distance computed
**exactly by linear programming** over the transport polytope (scipy HiGHS), ground metric ℓ¹.
Compactness tested as total boundedness: a greedy ε-net (ε=0.35) over N sampled bofop DIDMs
saturates at 5 elements for N=20 and N=40. Removing the degree bound (ER graphs of growing
size, 1-D degree DIDM, exact 1-D W₁) gives net sizes 10/20/40 — every sample is its own net
point, no saturation. Properness: a DIDM with all mass above the bound sits at mover's
distance 0.124 > 0 from all 20 sampled bofop DIDMs, so the inclusion is strict.
*Limitation:* a finite-sample ε-net is a surrogate for a topological statement. It is strong
evidence of total boundedness and a clean demonstration that the fiber bound is what buys it,
but it is not a proof.

### Claim 5 — Section 6.1, universal approximation on sparse graphs
600 bofop graphs (n∈[60,90], r≤6), target F = tanh(mean deg) + 0.5·sin(mean nbr deg) — a
permutation- and size-invariant, DIDM-continuous functional, exactly the class Sec. 6.1
addresses. Model: MPNN with random tanh message-passing layers (depth 3) + mean readout +
ridge head; width swept 4→256. **Uniform (sup-norm) test error** — the right norm for a
universal-approximation claim, not RMSE — falls to 0.0115, about 1.2% of the target's range.
Swapping in a DIDM-discontinuous target (indicator of mean degree above median) leaves sup
error at ≈0.52 regardless of width, confirming the approximation is coming from DIDM-continuity
and not from raw capacity.
*Limitation:* random-feature MPNNs on a sampled family demonstrate the approximation, they do
not reprove density in C(bofop-DIDM). Directionally faithful, toy in scale.

### Claim 6 — Section 6.2, vanishing generalization error
Fixed-capacity random-feature MPNN (width 64, depth 3) + ridge head on bofop graphs,
5 replicates per point, m ∈ {50,…,800}, test set 400.
*Process note:* the first run produced gaps of ~1e-7 with a meaningless slope — the head was
fitting the noiseless target essentially exactly, so the gap was at the float64 noise floor and
the rate question was vacuous. Adding label noise σ=0.05 makes the gap a real quantity. With
that, gap decays 2.75e-3 → 2.22e-4, log-log slope **−0.96**, comfortably better than the −1/2
the equicontinuity/covering-number argument guarantees. The claim as stated (error vanishes as
sample size grows) reproduces.
**Caveat, stated plainly:** the heavy-tailed-degree mutation raises the gap level 2–11× at every
m but its slope is *steeper* (−1.54), so it does **not** degrade the asymptotic rate here. The
experiment therefore supports the vanishing-error claim and shows the fiber bound improves the
constant, but it does **not** isolate compactness as necessary for the rate. That part of the
Sec. 6.2 mechanism remains untested by this reproduction.

## Overall assessment

All six anchored claims survive faithful finite-dimensional reproduction with discriminating
mutation tests, on CPU, within budget (~1h wall clock). Claims 1–3 are checked to numerical
precision and are effectively conclusive at the level of the finite model. Claims 4–6 are
empirical corroborations of topological/asymptotic statements: they are consistent with the
paper and their mutations behave as the theory predicts, but they are evidence, not proof, and
claim 6's mutation is only partially discriminative. No numerical value in the paper could be
matched because the paper reports none.
