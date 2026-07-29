# CAME-Grad claim decomposition

Frozen realistic forecast: `6/10`; stretch: `7/10`.

## C1: destructive interference and magnitude collapse

Paper claim: for conflicting task gradients, the negative interaction term
reduces the squared norm of their linear combination and can cause energy
collapse.

Audit: verify the exact two-task norm identity over random vectors and an
opposing-gradient construction; mutate the interaction sign and verify that
the identity fails. Expected: `2/2`.

## C2: Stage 1 strict local Pareto descent

Paper claim: the min-max trust-region formulation forces a consensus direction
with non-negative inner product against every task gradient, establishing a
safe lower bound and eliminating negative transfer.

Audit: solve the declared one-dimensional primal and dual exactly for
incompatible gradients and test every inner product. Expected: `2/2` if a
feasible counterexample yields a negative worst-task improvement.

## C3: Stage 2 target magnitude and scalar covariance scaling

Paper claim: magnitude enhancement strictly enforces
`||u_en|| = kappa ||g_joint||`, and scaling by `kappa` increases gradient-noise
covariance trace by `kappa^2`.

Audit: check the exact normalization including epsilon and independently
verify the conditional scalar identity `Cov(kappa X) = kappa^2 Cov(X)`.
Expected: `1/2`; this does not reproduce the stochastic covariance of the
complete CAME-Grad mechanism.

## C4: geometric-validity scope after adaptive fusion

Paper narrative: the algorithm ensures geometric validity, while the formal
glossary defines that condition for the rectified Stage 1 direction and the
appendix acknowledges conflict resurgence at high `nu`.

Audit: construct task gradients for which Stage 1 is Pareto-valid but a legal
task weighting and `nu` make the final fused direction conflict with one task.
The fixed construction must solve the declared Stage 1 dual, execute Stage 2
normalization, and then execute Stage 3 fusion; a hand-selected consensus
direction is insufficient. Expected: `1/2` as a scope qualification, not a
falsification of the paper's formal Stage 1 definition.

## C5: clinical efficacy across eight RRG methods

Paper claim: average clinical efficacy improves by 2.3% on MIMIC-CXR and 1.9%
on IU X-Ray across eight report-generation methods.

Full route: author optimizer, weights or fresh training, dataset access,
CheXbert evaluation, all eight baselines, and declared hyperparameters.
Expected now: `0/2`.
