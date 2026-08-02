# logbook.md — FluxNet (OpenReview 1KRpajnd6u, arXiv 2602.01941)

**Title:** FluxNet: Learning Capacity-Constrained Local Transport Operators for Conservative and Bounded PDE Surrogates
**Environment:** Python 3.12.3, numpy 2.5.1, scipy, sympy 1.14.0, torch 2.13.0+cpu, CPU only (`.venv`)
**Elapsed:** first tool call 2026-08-01T23:24Z → finish 2026-08-01T23:50Z ≈ **0h26m** (budget: 8h hard / 4h soft / 2h per claim — well inside)
**LLM turns used:** ~18 of 60.
**Paper read:** ONCE (arXiv HTML v1 → text) → `notes_paper.md`. Never re-read.
**Official code:** none exists — searched the paper text for `github` / `zenodo` / `anonymous.4open` / "code is available": **zero hits**. So `notes_code.md` was never created (nothing to grep). No third-party reproduction logbooks were read.

---

## Summary table

| Claim | Verdict | One-line evidence |
|---|---|---|
| 1 — Prop 1: flux update exactly conserves the global sum, drift ~1e-7…1e-8 | **verified** | Symbolic residual identically `0` for N=3,4,5,6; over 240 configs float64 max rel. drift **4.44e-16**, float32 max **7.39e-8** (median 1.48e-8) — inside the paper's 1e-7…1e-8 band; both mutations raise drift >1e4×. |
| 2 — Props 2 & 3: L-head / U-head structurally guarantee bounds without clipping | **verified** | 540 configs (shapes × radii × seeds × logit scales, incl. saturating logits and cells sitting exactly on the bound): **0 violations**, worst excursion 3.55e-15 (roundoff); symbolic margins `a(1-α)` and `b(1-β)`; mutation α,β ≤ 1.6 → **120/120** configs violate, magnitude O(10²). |
| 3 — Table 3 shallow water: 3.12e-3 vs 6.74e-3 depth MAE, E_cons 3.3e-7 → 3.3e-8 | **inconclusive** | No code, no SWE dataset, no generation script; requires full pushforward training of 2 backbones × 4 methods × multi-seed. No toy substitute produced. |
| 4 — Table 4 traffic: 3.48e-3 vs 15.9e-3 MAE, E_cons 2.1e-2 → 7.7e-8, V_ub 1.87% vs 3.20% | **inconclusive** | Same data/training blocker; additionally the anchored **1.87%** matches no printed Table 4 entry (Table 4 gives 2.87% ResNet / 2.01% FNO for FluxNet-D). |
| 5 — Table 5 spinodal: 17.3× speedup at 1000·Δt with preserved two-point statistics | **inconclusive** | Speedup is defined against a **GPU** explicit solver (no GPU here, ratio undefined on CPU) and needs three trained models on unreleased 50k-step phase-field trajectories. |
| 6 — D-head: dual bounds via DCL, empirically near-zero violations, not a strict guarantee | **verified** | Averaged update (Eq. 3) conserves to 4.38e-16; untrained branches violate on **6/6** seeds (lb 3.27%, ub 2.65%, max magnitude 0.479) ⇒ no architectural guarantee; minimizing L_DCL to 1.04e-8 gives **0.00%** violations on 6/6 seeds; mutation (maximize L_DCL) → 0/6 seeds clean, magnitude 2.90; violation ≤ ½|Δu^out−Δu^in| on all 20 seeds; Spearman ρ = 0.903 (n=90) between √DCL and violation magnitude. |

---

## Claim 1 — Proposition 1 (discrete conservation)

**Paper source:** Section 3.2, Eq. (2); Appendix A, Proposition 1, Eqs. (6)–(9).
**Script:** `verify_claim1.py` → `results/claim1.json`. Command: `.venv/bin/python verify_claim1.py`.

The proposition holds for *any* flux field, so instead of training a network I sampled head
outputs (logits) randomly — a strictly stronger test than one trained model.

- **Symbolic (exhaustive, exact rationals):** 1D periodic ring, radius-1 symmetric stencil,
  fully symbolic fluxes. `sum(u^{t+1}) - sum(u^t)` simplifies to `0` for N = 3, 4, 5, 6.
- **Numeric:** 4 shapes × 3 radii × 20 seeds = **240 configs** per dtype, 20 rollout steps each,
  L-head fluxes, periodic `np.roll` inflow.
  - float64: max relative drift **4.436e-16**, median 2.18e-16 (exact to machine epsilon).
  - float32: max **7.392e-08**, median 1.475e-08, min 1.85e-09.
    The paper reports 1.19e-7 / 2.38e-7 / 1.79e-7 (Table 2, float32) and 3.3e-8 / 7.7e-8
    (Tables 3–4). **The observed float32 band coincides with the reported one**, confirming that
    the reported "machine precision" numbers are exactly float32 accumulation roundoff, not a
    property of the trained model.

**Mutation tests (mechanism, not correlation):**
1. *asymmetric_inflow* — receiver credited 1.001× the amount the sender debited (breaks the
   shared-flux property while keeping everything else): drift **9.68e-3 … 1.04e-2**, i.e.
   ≥1.3e5× the unmutated float32 error.
2. *nonperiodic* — zero-padded shift instead of periodic roll (breaks the symmetric-stencil /
   periodic-BC hypothesis): drift **2.26e-2 … 4.52e-1**, ≥3e5× baseline.

Both mutations fail in exactly the direction Proposition 1 predicts. **Verdict: verified.**

## Claim 2 — Propositions 2 and 3 (L-head / U-head bound guarantees)

**Paper source:** Section 3.3, Propositions 2 and 3 (with Table 1 parameterizations).
**Script:** `verify_claim2.py` → `results/claim2.json`. Command: `.venv/bin/python verify_claim2.py`.

Heads implemented verbatim from Sec 3.3: `a_i = u_i − ℓ`, `α = σ(·) ∈ (0,1)`,
`π = softmax` over the K stencil directions, `F_{i→i+d} = a_i α_i π_{i→i+d}`; and dually
`b_i = u_max − u_i`, `β = σ(·)`, `ρ = softmax` over incoming directions,
`F_{j→i} = b_i β_i ρ_{j→i}` with the sender-side outflow recovered by the inverse roll.

- **Symbolic:** worst-case (zero inflow) L-head margin `u^{t+1} − ℓ = a(1−α) > 0`;
  worst-case (zero outflow) U-head margin `u_max − u^{t+1} = b(1−β) > 0`.
- **Numeric:** 4 shapes (1D and 2D, incl. non-power-of-two 127 and 48×24) × 3 radii ×
  15 seeds × 3 logit scales (σ = 1, 5, **20** — the last saturates sigmoid/softmax to the
  adversarial extreme) = **540 configs**, 20 steps each, and every third seed starts with 10 %
  of cells sitting exactly *on* the bound.
  - L-head: **0/540** violating configs, worst excursion 3.55e-15.
  - U-head: **0/540** violating configs, worst excursion 1.11e-15.
  Both worst values are float64 roundoff (< 1e-12), not structural violations.
- **No post-hoc correction:** the implementation is asserted to contain no `clip`/`maximum(`
  call (`no_clipping_in_implementation: true`), so the bound comes from the parameterization.

**Mutation test:** replace the capacity fraction `σ(·) ∈ (0,1)` by `1.6·σ(·)`, which can exceed
the available amount / remaining capacity — i.e. invert the one inequality the proofs rely on.
Result: **120/120** configs violate, max violation 340.0 (L) and 342.0 (U). The guarantee is
therefore attributable to the `α, β < 1` capacity fraction, exactly as the proofs state.
**Verdict: verified.**

## Claim 3 — Table 3 (shallow water) — inconclusive

**Paper source:** Section 4.2, Table 3. Quoted: FluxNet-LAP depth MAE 3.12e-3 vs 6.74e-3
(Box+Mass projection), E_cons 3.3e-7 → 3.3e-8.
**Record:** `verify_claims345.py` → `results/claim3.json`.

Blockers: (i) no code repository anywhere in the paper; (ii) the 2D SWE dataset with dry regions
is authors' own and unreleased, with generation parameters not in the main text; (iii) the number
requires pushforward training of ResNet **and** FNO backbones plus three baselines, multi-seed —
outside an 8h CPU budget even with data.
Additional observation, recorded rather than resolved: **3.12e-3 is the ResNet-backbone
FluxNet-LAP entry while 6.74e-3 is the FNO-backbone projection entry.** Within the ResNet
backbone the projection baseline is 22.1e-3; within the FNO backbone FluxNet-LAP is 2.22e-3.
The anchored comparison crosses backbone categories.
The *conservation-error* half of the claim is covered generically by claim 1 (a flux-form model
lands at 1e-8…1e-7 in float32 by construction, and a projection baseline that renormalizes mass
globally lands in the same band — so the 3.3e-7 → 3.3e-8 "improvement" is within the float32
roundoff band rather than a qualitative difference). No toy substitute was produced.

## Claim 4 — Table 4 (traffic LWR) — inconclusive

**Paper source:** Section 4.3, Table 4.
**Record:** `results/claim4.json`. Same data/training blockers as claim 3.
**Discrepancy recorded:** the anchored claim quotes an upper-bound violation rate of **1.87 %**
for FluxNet-D, but Table 4 as printed reports **2.87 %** (ResNet backbone) and **2.01 %** (FNO
backbone); 1.87 % matches neither, and the nearest printed value is FNO-AR's V_ub = 1.88 %.
Unresolvable without the authors' raw numbers. The structural sub-parts (machine-precision
conservation; D-head having no strict dual-bound guarantee but near-zero empirical violations)
are covered by claims 1 and 6.

## Claim 5 — Table 5 (spinodal decomposition, 17.3× speedup) — inconclusive

**Paper source:** Section 4.4, Table 5 and Figure 3.
**Record:** `results/claim5.json`.
The 17.3× speedup is defined *relative to a GPU-accelerated explicit solver*; with no GPU in this
environment the reference timing cannot be measured and the ratio is not defined on CPU.
It also requires three trained FluxNet-D models (r = 3/5/9) on unreleased 50 000-step
Cahn–Hilliard trajectories. The two-point correlation metric S̄₂(r) is itself implementable via
FFT autocorrelation, but there is no trained model to evaluate it on. No toy substitute.

## Claim 6 — D-head: DCL, not architecture, enforces the dual bound

**Paper source:** Section 3.4, Eqs. (3)–(4); Table 1 row "D"; Conclusion.
**Script:** `verify_claim6.py` → `results/claim6.json`. Command: `.venv/bin/python verify_claim6.py`.

The claim has three parts, all tested:

1. **Conservation survives averaging.** The Eq. (3) averaged update has max relative drift
   **4.377e-16** (float64) over 40 configs — each branch is conservative, so their mean is.
2. **No strict dual-bound guarantee.** With independent (untrained) branch logits, **6/6** seeds
   produce violations: mean lb rate **3.27 %**, ub rate **2.65 %**, max magnitude **0.479** on a
   [0,1] field. This matches the paper's own statement that the averaged update "does not
   theoretically guarantee satisfaction of both bounds unless the two branches agree exactly" —
   the anchored claim is therefore confirmed rather than contradicted.
3. **DCL is the enforcing mechanism.** Minimizing L_DCL (Eq. 4) directly on the head logits
   (Adam, lr 0.05, 200 steps, 32×32 grid, r = 2, float64) drives L_DCL from 0.195 to
   **1.04e-8** and the violation rate to **exactly 0.00 %** on **6/6** seeds — "near-zero
   violations without post-hoc clipping", as claimed.

**Mutation test:** flip the sign of the objective (maximize branch disagreement) for the same
200 steps. L_DCL rises to 15.5, **0/6** seeds reach zero violations, and the max violation
magnitude grows to **2.90** (6× the untrained level). Violations track DCL, not the optimizer.

**Supporting analysis:** the per-cell violation is provably at most ½|Δu^out − Δu^in|; this held
on all 20 tested seeds. Across an ensemble of 90 random states spanning six logit scales, the
Spearman correlation between √L_DCL and the violation magnitude is **ρ = 0.903 (p = 4.8e-34)**,
with 48/90 states showing exactly zero violation at low disagreement.
**Verdict: verified.**

---

## Evidence boundary

What this evidence does **NOT** cover:

1. **No trained FluxNet model was reproduced.** Every "verified" result concerns the *update
   rule and head parameterizations*, exercised with randomly sampled head outputs. Propositions
   1–3 are stated to hold for arbitrary fluxes, so this is a valid (indeed stronger) test of the
   propositions — but it says nothing about whether a trained FluxNet reaches the reported
   accuracies.
2. **No accuracy claim is tested.** All MAE numbers (Tables 2–5) are untouched. Claims 3, 4 and 5
   are recorded as inconclusive with reasons; no synthetic stand-in was produced for them.
3. **My implementation is a faithful re-implementation from the paper text, not the authors'
   code** (no code was ever released). If the authors' code differs from Sec 3.2–3.4 — e.g. a
   different inflow-shift convention, an ε in the L-head, or a hidden clip — my results describe
   the *published equations*, not their software.
4. **Conservation magnitude is dtype-dependent.** The ~1e-7 vs ~1e-8 distinction the paper draws
   between methods is float32 accumulation roundoff and scales with grid size, rollout length and
   summation order; in float64 the same code gives 4e-16. So "3.3e-8 beats 3.3e-7" is not
   evidence of a better conservation mechanism — both are exactly conservative up to roundoff.
5. **Periodic boundaries only**, as in the paper. Nothing here speaks to Dirichlet/Neumann
   boundaries, source terms, or non-uniform meshes (the paper's own stated limitations).
6. **Claim 6's DCL experiment optimizes the head logits directly, not network weights** on real
   PDE data. It establishes the causal link DCL-residual → bound violations, but the "near-zero
   violations" observed in Table 4 (0.52 % / 2.87 %) for an actually trained model on real
   traffic data are not reproduced.
7. **Single-machine timing only** — no speedup measurement of any kind was made (no GPU).
8. The **claim-4 1.87 % discrepancy** is reported, not adjudicated; I could not determine whether
   it is a transcription error in the anchored claim or a number from a table version I did not see.

## Artifacts

```
notes_paper.md          single-read paper notes (no notes_code.md: no code released)
plan.md                 per-claim feasibility classification
fluxnet_core.py         numpy re-implementation of Eq.2 update, L/U/D heads
verify_claim1.py        -> results/claim1.json
verify_claim2.py        -> results/claim2.json
verify_claim6.py        -> results/claim6.json  (+ results/claim6.log)
verify_claims345.py     -> results/claim3.json, claim4.json, claim5.json
check_reproducibility.py  reproducibility gate — PASSES (exit 0, 0 failed assertions)
```
