# Claim 2: SDE mechanism and convergence


---
<!-- trackio-cell
{"type": "markdown", "id": "cell_1a92fafc8b7b", "created_at": "2026-07-29T17:12:57+00:00", "title": "Claim 2: SDE mechanism and convergence"}
-->
**NOT ESTABLISHED BY THE PRINTED EQUATIONS — 0/2.**

Two update distributions were constructed with exactly equal norm one. The
constant distribution has covariance trace
`0.0`,
while the direction-varying distribution has trace
`0.5000000000000001`.
Thus enforcing update magnitude does not determine gradient-noise covariance
`Sigma` or guarantee restored SDE diffusion. A trust-region bound also does
not alone prove global convergence or movement toward flatter minima.
