# Claim 2: The metrics can separate the effects of over- and under-coverage


---
<!-- trackio-cell
{"type": "markdown", "id": "cell_be086ec180b2", "created_at": "2026-07-19T08:18:56+00:00", "title": "Claim 2: The metrics can separate the effects of over- and under-coverage"}
-->
**Setup.** Theorem 3.1 lets the convex `f` be decomposed as `f = f_+ + f_-` with `f_+(p) = f(max{p, 1-a})` penalising only over-coverage and `f_-(p) = f(min{p, 1-a})` penalising only under-coverage. We estimate each part separately with the same cross-fitted ERT procedure, on the standard-CP (conditionally invalid) sets, n = 30,000.

| component | estimated | theoretical |
| --- | --- | --- |
| over-coverage part | 0.0445 | 0.0460 |
| under-coverage part | 0.0464 | 0.0488 |

**Verdict — Claim 2 reproduced.** Both parts are clearly non-zero and match the analytic values closely. This is exactly the expected structure for a constant-width conformal set on heteroscedastic data: it **over-covers where the noise is small** (small `sigma(X_1)`) and **under-covers where the noise is large**. A single scalar miscoverage number would hide this; the decomposition identifies the direction of the failure, which is the diagnostic value the paper claims.

**Why this matters for medical use.** Under-coverage is the clinically dangerous direction (intervals too narrow for exactly the hard cases), while over-coverage is merely uninformative. Being able to separate the two — rather than reporting one aggregate gap — is directly adoptable for auditing clinical risk models.
