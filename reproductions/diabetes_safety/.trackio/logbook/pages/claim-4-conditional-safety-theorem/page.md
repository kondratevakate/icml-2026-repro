# Claim 4: conditional safety theorem


---
<!-- trackio-cell
{"type": "markdown", "id": "cell_c9a500452f13", "created_at": "2026-07-29T20:21:46+00:00", "title": "Claim 4: conditional safety theorem"}
-->
**Verdict - VERIFIED AS A CONDITIONAL THEOREM (2/2).**

The epsilon-margin implication passes
`320` exhaustive boundary combinations with
maximum safety residual `0.0`.
Weakening the reliability event produces
`40` failures.

This verifies the stated implication only. It does not prove that the learned
predictor satisfies one-sided reliability in deployment.


---
<!-- trackio-cell
{"type": "code", "id": "cell_c5b0f92ffb8c", "created_at": "2026-07-29T20:21:46+00:00", "title": "C4 machine-readable evidence", "language": "json"}
-->
````output
{
  "checked_implications": 320,
  "hyperglycemia_checked_implications": 160,
  "hyperglycemia_mutation_failures": 20,
  "hypoglycemia_checked_implications": 160,
  "hypoglycemia_mutation_failures": 20,
  "max_safety_margin_residual": 0.0,
  "verdict": "VERIFIED_CONDITIONAL_IMPLICATION",
  "weakened_reliability_mutation_failures": 40
}
````
