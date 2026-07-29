# PEQ-Net Claim Decomposition

Paper ID: `bIcz7bIZSo`. Frozen forecast: `4/12` realistic, `6/12` stretch.

## C1: deterministic policies, limited confounding

Anchored claim: PEQ-Net has lower RMSE than DeepLTMLE for all three
deterministic-policy contrasts under limited time-varying confounding.

Prepared verdict: `INCONCLUSIVE`.

Reason:

- no PEQ-Net or author DeepLTMLE code was released;
- the experiment requires processed MIMIC-III trajectories;
- the printed DGP contains division by zero at the first lag;
- seeds and selected per-run hyperparameters are not reported.

Paper-table values are provenance, not independent reproduction evidence.

Expected score now: `0/2`.

## C2: deterministic policies, expanded confounding

Anchored claim: PEQ-Net has lower RMSE than DeepLTMLE for all three
deterministic-policy contrasts under expanded time-varying confounding.

Prepared verdict: `INCONCLUSIVE` for the same blockers as C1. The expanded
latent-state recurrence is given, but it delegates treatment and outcome
generation to the undefined limited-DGP equations.

Expected score now: `0/2`.

## C3: dynamic policies

Anchored claim: PEQ-Net has lower RMSE than DeepLTMLE for both partial- and
full-deviation dynamic policies in both DGPs.

Prepared verdict: `INCONCLUSIVE`.

The threshold policies are specified, but the underlying DGP and unpublished
neural implementation prevent an independent 20-seed comparison.

Expected score now: `0/2`.

## C4: policy embedding and shared Q architecture

Anchored claim: PEQ-Net represents policies using Gaussian-kernel mean
embeddings, pairwise MMD and metric MDS, then encodes policy tails to condition
shared Q-functions.

Prepared verdict: `VERIFIED WITH QUALIFICATION`.

Independent test:

- generate several threshold policies on a fixed synthetic history sample;
- compute biased empirical Gaussian-kernel MMD independently;
- run metric MDS on the paper's dissimilarity matrix;
- verify finite embeddings and preservation of policy-distance ordering;
- audit the source algorithm from policy application through shared-Q input.

Qualification: the source defines `D` using `MMD^2` and says MDS preserves it
approximately, but the theorem proof later replaces this with exact equality
to `MMD`. The independent implementation verifies the pipeline, not that false
exact identity.

Expected score: `2/2`.

## C5: Theorem 4.2 remainder control

Anchored claim: under the stated Lipschitz and regularity assumptions, a finite
constant uniformly controls the CATE second-order remainder contrast by
trajectory-level MMD.

Prepared verdict: `FALSIFIED AS A LIPSCHITZ GUARANTEE`.

Independent audit:

- the subsequence-MMD lemma defines its constant as the ratio for each already
  fixed policy pair, which is tautological and not a uniform Lipschitz bound;
- the proof assumes exact MDS preservation despite the method specifying an
  approximate metric-MDS solution;
- bounded LTMLE fluctuation does not imply the claimed Lipschitz map from
  initial to targeted Q estimates;
- identical initial Q values with distinct bounded policy-specific
  fluctuations give a zero right-hand side and a nonzero targeted difference.

The theorem may be repairable by adding uniform bridge and targeting-map
assumptions. Those assumptions are absent from the statement.

Expected score: `2/2`.

## C6: MIMIC-IV sepsis case study

Anchored claim: among 999 hypotensive adult sepsis patients, higher MAP
vasopressor-weaning targets are associated with higher 72-hour lactate.

Prepared verdict: `INCONCLUSIVE`.

Required later:

- complete MIMIC-IV `hosp/` and `icu/`;
- auditable Sepsis-3, hypotension, vasopressor, MAP-duration and 72-hour
  lactate extraction;
- exact item IDs, missing-data rules, time origin and exclusion criteria;
- independent model fit and aggregate-only published evidence.

Expected score now: `0/2`; stretch: `2/2`.
