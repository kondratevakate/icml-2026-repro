# plan.md — reproduction plan for `Vv4XRZDMM0`

Paper: **Multi-Distribution Robust Conformal Prediction** (Yang & Jin), arXiv 2601.02998,
OpenReview `Vv4XRZDMM0`. Official code: `code_mdcp/` (read-only; **read, not executed** — it
depends on a WILDS repo bootstrap and an `eval_out` tree we do not have).

## Contamination control

The methodology skill `paper-claim-reproduction` ships a reference file
`references/mdcp-conformal-icml2026.md` — a worked example **for this exact paper**. It was
deliberately **not opened**. Only the skill body (generic methodology) was read. The paper HTML
(arXiv v1) and `code_mdcp/` are the only paper-specific sources used. No HuggingFace
`icml2026-repro` Space was visited.

## Dependency cap

`.venv` provides numpy 2.5.1 / scipy 1.18.0 / sympy 1.14.0 / torch 2.13.0+cpu / scikit-learn 1.9.0.
8 CPU cores. No GPU, no network dataset of WILDS scale. This caps claim 6 (see below).

## Claim routing

| # | Claim (abbrev.) | Source located | Route | Verdict ceiling |
|---|---|---|---|---|
| 1 | Thm 1: finite-sample uniform validity of max-p under any mixture | Thm 1, Sec. 2; proof App. B.1 | simulation, oracle densities, exhaustive over the mixture simplex + seeds | `verified` |
| 2 | Thm 2: optimum is superlevel set of `h_{λ*}`, compl. slackness ⇒ exact 1−α for ≥1 source | Thm 2, Sec. 3.1; proof App. B.3 | **convex program**: instantiate the conditional program as an LP, machine-check every clause via HiGHS duals | `verified` |
| 3 | Thm 3: max-p with learned scores → oracle-optimal set up to boundary | Thm 3, Sec. 3.2; proof App. B.4 | simulation n-sweep of `ρ(Ĉ △ {h*≥1})`; premise-violation mutation | `verified` |
| 4 | 34.39% smaller sets than max-p baseline, classification Linear | Sec. 5.2 text + Fig. 2 (`34.39` occurs 1× in paper text) | **full pipeline**: paper's DGP verbatim, GBM sources + spline λ ERM (9), 100 reps | `verified` |
| 5 | 22.44% average reduction, regression Linear | Sec. 5.3 text + Fig. 5 (`22.44` occurs 1×) | full pipeline, regression DGP verbatim | `verified` |
| 6 | FMoW, 6 regions, 249 countries, near-tight worst-case coverage | Sec. 6.1, Fig. 8 (`249` occurs 1×) | **REFUSE** | `inconclusive` |

### Why claim 6 is refused, not toyed

FMoW-WILDS 2016 slice is >1M satellite images (~100 GB download) and the paper's pipeline
(`code_mdcp/wilds/fmow/`) fine-tunes a DenseNet-121 embedding model — GPU work, far outside a
75-minute CPU budget. Per the hard rules, a synthetic stand-in would be a fake toy. Emit a refusal
artifact with `what_would_be_needed`. The *mechanism* claim it shares with claims 1/4 (max-p gives
worst-case coverage; learned h shrinks sets) is tested there; the FMoW *magnitudes* are not.

## Geometry choices (Thm 2 / Thm 3)

Thm 2's uniqueness clause needs `μ({y : h*(x,y)=1}) = 0`. Under counting measure (classification,
C=6) that set has positive mass and the LP optimum is fractional, so "C* is a superlevel set" is
only well-posed up to the boundary. Claim 2 is therefore run in **both** geometries: the
regression/Lebesgue geometry (fine y-grid, non-degeneracy holds) as primary, classification/counting
as secondary with the boundary clause left unconstrained. Claim 3 is run in the Lebesgue geometry
only — the symmetric-difference statement is not sharply falsifiable under counting measure.

## Mutation tests (stated before running)

- C1: aggregation ladder `max → mean → min`. Predicted: max ≥ 1−α everywhere; mean dips below;
  min collapses toward α. Non-vacuity guard: component coverage ≈ 0.9, set size < |Y|.
- C2: replace λ* by a wrong λ′ and take `{h_λ′ > 1}`. Predicted: never as good as the optimum
  (`frac_as_good_as_optimum == 0.0`) — infeasible or strictly larger.
- C3: violate the theorem's **premise** — feed non-converging scores (uniform λ; single-source λ)
  through the identical estimator and n-sweep. Predicted: error **plateaus** instead of decaying.
- C4/C5: replace the learned λ̂(x) by uniform λ ≡ 1/K in the same pipeline. Predicted: the
  efficiency gain over Baseline-agg largely collapses.

## Budget order

C2 (cheap, decisive) → C1 → C4 (heavy, launched in background) → C3 → C5 → C6 refusal artifact.
