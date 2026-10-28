# Claim 1: loss-valley theorem


---
<!-- trackio-cell
{"type": "markdown", "id": "cell_9eb0465a3f1e", "created_at": "2026-07-30T07:41:34+00:00", "title": "Claim 1: loss-valley theorem"}
-->
**UNSUPPORTED - 0/2.** Non-uniqueness alone does not imply a
connected or non-isolated solution set. `N(u)=u²`, `f=1` is an explicit
counterexample. The appendix proof uses additional nonlinear regularity,
kernel, connectedness, and representability assumptions.


---
<!-- trackio-cell
{"type": "code", "id": "cell_85f6b607b870", "created_at": "2026-07-30T07:41:34+00:00", "title": "Claim 1: loss-valley theorem evidence", "language": "python"}
-->
````output
{
  "operator": "N(u) = u^2",
  "rhs": 1.0,
  "solutions": [
    -1.0,
    1.0
  ],
  "residuals": [
    0.0,
    0.0
  ],
  "non_unique": true,
  "minimum_pairwise_distance": 2.0,
  "solutions_are_isolated": true,
  "solution_set_connected": false,
  "headline_implication_valid": false,
  "missing_nonlinear_assumptions": [
    "non-trivial kernel of DN[u*]",
    "constant-rank or regular-value condition",
    "connected parameter domain for a continuous solution family",
    "representability of that family by the neural network"
  ]
}
````
