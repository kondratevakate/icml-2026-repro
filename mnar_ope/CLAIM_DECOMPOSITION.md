# MNAR OPE Claim Decomposition

Paper ID: `vpSFJoxyDz`. Frozen forecast: `6/10` realistic, `8/10` stretch.

## C1: bridge existence and policy identification

Anchored claim: Assumptions 2.1, 2.2, and 3.2-3.4 imply existence of bridge
functions and identify the full-data policy value without modeling the MNAR
mechanism.

Prepared verdict: `FALSIFIED AS WRITTEN` if a complete, relevant, positive,
one-stage MDP has no square-integrable bridge.

Independent test:

- Use a uniform reward on the circle and a wrapped-Gaussian next-state channel.
- All Fourier multipliers are nonzero, so both completeness operators are
  injective.
- The inverse coefficients needed to reproduce the bounded sawtooth reward are
  not square summable. Therefore no L2 bridge exists.
- Confirm that this is exactly the Picard condition added in the paper appendix.

Scope guard: this does not falsify the paper's qualified theorem, which
explicitly includes additional regularity conditions.

Expected score: `2/2`.

## C2: relevance condition

Anchored claim: Assumption 3.2 requires
`S_(t+1) not independent of R_t | S_t, A_t, O_t=1`.

Prepared verdict: `VERIFIED`.

Independent test:

- Parse the exact condition from the pinned arXiv source.
- Confirm that the official simulator makes reward depend directly on the next
  state.
- On an independently generated observed subset, residualize reward and next
  state against current state/action and test the remaining association.

Expected score: `2/2`.

## C3: bridge estimation error bound

Anchored claim: Theorem 5.8 bounds bridge RMSE by the ill-posedness coefficient,
critical radius, and bridge RKHS norm.

Prepared verdict: `VERIFIED` only if:

- the exact theorem statement is found in the pinned source;
- the projected-error-to-L2 reduction follows from the stated definition of
  the ill-posedness coefficient;
- a finite-dimensional independent check satisfies the inequality for random
  errors and operators;
- all additional assumptions are reported rather than omitted.

Expected score: `0-2/2`.

## C4: policy-value error rate

Anchored claim: Theorem 5.9 gives the stated `T^2`, `tau_max`, logarithmic, and
nonparametric sample-size factors.

Prepared verdict: `VERIFIED` only after auditing the full appendix proof from
the one-step error decomposition through the final union bound and critical
radius substitution.

The audit must explicitly report two editorial defects rather than silently
normalizing them: a missing summation sign in an intermediate display and an
implicit `zeta/T` probability-budget adjustment. The subsequent display and
the final `log(T/zeta)` factor restore the intended bookkeeping.

Expected score: `2/2` if every proof obligation passes.

## C5: no-future-dependence assumption

Anchored claim: Assumption 2.1 states that current missingness is independent
of future states and rewards conditional on current state, action, and reward.

Prepared verdict: `VERIFIED`.

Independent test:

- Parse the exact conditional-independence statement from the source.
- Inspect the official simulator's missingness probability and random draw.
- Confirm that future-state terms add no direct predictors after conditioning
  on the exact variables used by the Bernoulli mechanism.

Expected score: `2/2`.
