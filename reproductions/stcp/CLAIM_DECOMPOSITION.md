# StCP Claim Decomposition

Paper: *Stable Localized Conformal Prediction via Transduction*.
OpenReview: `lSMTccAN61`. Frozen forecast: `8/12` realistic, `10/12` stretch.

## C1: set-stability criterion

Anchored claim: set stability is the variance, over construction data, of the
conditional expected prediction-set size.

Full route:

- derive the law-of-total-variance decomposition used in Section 3;
- independently simulate repeated calibration datasets;
- verify that averaging over test covariates removes test-point noise;
- mutation test: use raw per-test set-size variance and show it adds the
  within-calibration component that the proposed criterion excludes.

Expected score: `2/2`.

## C2: StCP marginal-coverage characterization

Anchored claim: Theorem 4.2 bounds deviation from nominal marginal coverage by
the better of a target-alignment error route and a source-estimator error
route, plus finite-sample quantile error.

Full route:

- audit every assumption and quantile step in the theorem/proof;
- independently evaluate both branches of the minimum over a finite parameter
  grid;
- mutation test source misspecification and lambda extremes;
- do not simplify an asymptotic rate into an exact distribution-free claim.

Expected score: `2/2`.

## C3: stability-rate improvement

Anchored claim: Theorem 4.6 gives standard CP stability `O(n^-1)` and StCP
stability controlled by unlabeled sample size and regularization, with faster
decay when `m` and lambda are sufficiently large.

Full route:

- audit Lemma 4.4 and Theorem 4.6 from quantile variance to set stability;
- verify the two curvature cases and all conditioning;
- independently estimate slopes over repeated calibration samples;
- mutation test by resampling unlabeled data independently for each test
  point, which should destroy the intended construction-level stabilization.

Expected score: `2/2`.

## C4: data-driven lambda coverage interval

Anchored claim: Theorem 4.7 gives the finite-sample marginal interval
`[1-alpha-alpha_tol, 1-alpha+alpha_tol+1/(n+1)]` for the selected quantile.

Full route:

- implement the selection rule independently from calibration scores and a
  candidate quantile grid;
- exhaustively enumerate small exchangeable score orbits;
- verify both interval endpoints;
- mutation tests: remove lambda zero from the grid and select outside the
  feasible quantile interval.

Expected score: `2/2`.

## C5: medical stability gains

Anchored claim: SLCP improves stability on DermaMNIST and TissueMNIST while
maintaining acceptable marginal coverage and comparable set size.

Full route:

- run released medical data/checkpoints with fresh deterministic repeats;
- publish only aggregate metrics;
- compare against the corresponding base method under the paper's own
  acceptable-coverage rule;
- mutation test by shuffling source labels or replacing source CDF estimates
  with constants.

Expected score now: `0/2`; stretch: `2/2`.

## C6: low-label synthetic gains

Anchored claim: in the LogAbs setting with `n=30,m=500`, StCP reduces the Std
metric from `1.12` to `0.77` for GLCP and from `0.98` to `0.82` for CQR while
coverage remains near 0.9.

Full route:

- execute the released LogAbs DGP under fresh seeds;
- reproduce 50 repeated calibration datasets or a predeclared minimum with
  confidence intervals;
- verify the direction and approximate magnitude for both models;
- mutation test lambda zero and source/target conditional-shift amplification.

Expected score: `2/2`.
