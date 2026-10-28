# Claim 2: zero-padded difference


---
<!-- trackio-cell
{"type": "markdown", "id": "cell_713728d72be2", "created_at": "2026-07-29T16:40:28+00:00", "title": "Claim 2: zero-padded difference"}
-->
**Verdict - FALSIFIED IN RELEASED IMPLEMENTATION.**

The paper defines `[0; Z[2:T]-Z[1:T-1]]`. On a deterministic nonzero input,
the released operator's first step equals the input and reaches
`72.0`; the declared operator is exactly
zero. A direct corrected mutation recovers the paper operator.


---
<!-- trackio-cell
{"type": "code", "id": "cell_a55dedeb57e3", "created_at": "2026-07-29T16:40:28+00:00", "title": "C2 machine-readable evidence", "language": "python"}
-->
````output
{
  "declared_first_step_max_abs": 0.0,
  "id": "C2",
  "observed_first_step_equals_input": true,
  "observed_first_step_max_abs": 72.0,
  "operator_max_abs_difference": 72.0,
  "paper_anchor_present": true,
  "reason": "Left zero-padding followed by x - padded[:-1] makes delta[0]=x[0], not zero.",
  "score": 2,
  "verdict": "FALSIFIED_IN_RELEASED_IMPLEMENTATION"
}
````
