# Tailored scoring rules claim decomposition

Frozen realistic forecast: `8/12`; stretch: `10/12`.

## C1: IPW downstream MSE bound

Paper claim: under identifiability, bounded conditional second moments, and
cross-fitting, plug-in IPW MSE is bounded by constants plus the expectation of
the declared bias and variance divergences.

Audit: reconstruct the finite-sample bias/variance decomposition from the
appendix, verify every inequality symbolically, and test exact finite
distributions. Mutate one inverse-propensity factor and require violations.
Expected: `2/2` if the proof ledger closes.

## C2: task curvature

Paper claim: the second derivative of the declared task divergence at `q=p`
equals
`2/p^2 + 2/(1-p)^2 + 2/p^3 + 2/(1-p)^3`.

Audit: derive the Hessian with SymPy, simplify the difference to zero, test a
dense boundary-aware grid, and mutate one exponent. Expected: `2/2`.

## C3: closed-form strictly proper scoring rule

Paper claim: integrating the task curvature yields the displayed partial
losses, and their conditional risk is uniquely minimized at the true
probability.

Audit: differentiate both partial losses, verify the proper-loss differential
identities, prove positivity of the weight, and exhaustively check conditional
risk minima on a dense grid. Expected: `2/2`.

## C4: quartic canonical probability mapping

Paper claim: the inverse canonical link reduces to the displayed quartic in
`u=1/[p(1-p)]`; selecting its largest real root yields a monotone probability
mapping and simplifies the logit gradient to `p-y`.

Audit: verify the integrated link, quartic residual, inverse round-trip,
monotonicity, symmetry, and analytic gradient over a wide logit grid. Compare
against incorrect-root and sigmoid mutations. Expected: `2/2`.

## C5: Kang-Schafer weak-overlap mechanism

Paper claim: the tailored objective is especially useful under propensity
misspecification and weak overlap.

Audit: independently implement the fully specified Kang-Schafer DGP and a
minimal cross-fitted linear propensity comparison using frozen seeds. Measure
ATE error and boundary behavior; include a log-loss and link-mismatch
mutation. Expected: `0-2/2`, depending on stability across the full seed panel.

## C6: broad benchmark superiority

Paper claim: the tailored objective consistently improves IPW, Hajek, and AIPW
across IHDP, Jobs, Kang-Schafer, and 32 ACIC 2017 DGPs.

Required route: released implementation or a full independent reconstruction
of datasets, splits, hyperparameter grids, models, estimators, and all runs.
Expected now: `0/2`.
