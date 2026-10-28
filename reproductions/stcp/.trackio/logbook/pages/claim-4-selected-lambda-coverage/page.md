# Claim 4: selected-lambda coverage


---
<!-- trackio-cell
{"type": "markdown", "id": "cell_b847a43dc6cc", "created_at": "2026-07-29T16:17:42+00:00", "title": "Claim 4: selected-lambda coverage"}
-->
**Verdict - FALSIFIED AS WRITTEN.**

For iid continuous exchangeable scores, coverage of the `k`-th calibration
order statistic is exactly `k/(n+1)`. The proof uses
`k=ceil(n p)` and incorrectly asserts `k/(n+1) >= p`.

At the paper's `n=30`, `alpha=0.1`, and
`alpha_tol=0.02`, the lower endpoint has `p=0.88`,
`k=27`, and exact coverage `27/31 =
0.870968`, below the claimed 0.88. The audit found
8960 failures over its finite `(n,p)`
grid. Replacing `ceil(n p)` by `ceil((n+1)p)` repairs this lower-endpoint
arithmetic in the tested configuration.


---
<!-- trackio-cell
{"type": "code", "id": "cell_6572f4ce226c", "created_at": "2026-07-29T16:17:42+00:00", "title": "C4 machine-readable evidence", "language": "python"}
-->
````output
{
  "evidence": {
    "alpha": 0.1,
    "alpha_tol": 0.02,
    "candidate_grid_construction": "lambda=0 is feasible; a larger feasible candidate returns q_L, so the max-feasible rule selects q_L",
    "checks": {
      "failure_is_not_tie_dependent": true,
      "n_plus_1_mutation_repairs_lower_endpoint": true,
      "paper_arithmetic_k_over_n_plus_1_ge_p_is_false": true,
      "theorem_lower_endpoint_is_violated": true
    },
    "claimed_lower_bound": 0.88,
    "claimed_upper_bound": 0.952258064516129,
    "corrected_exact_coverage": 0.9032258064516129,
    "corrected_lower_index_ceil_n_plus_1_p": 28,
    "exact_coverage_at_q_L": 0.8709677419354839,
    "exact_coverage_at_q_U": 0.9032258064516129,
    "exchangeable_score_model": "iid continuous scores, e.g. Uniform[0, 1]",
    "exhaustive_grid_failure_count": 8960,
    "first_five_grid_failures": [
      {
        "coverage": 0.3333333333333333,
        "k": 1,
        "n": 2,
        "p": 0.335
      },
      {
        "coverage": 0.3333333333333333,
        "k": 1,
        "n": 2,
        "p": 0.34
      },
      {
        "coverage": 0.3333333333333333,
        "k": 1,
        "n": 2,
        "p": 0.345
      },
      {
        "coverage": 0.3333333333333333,
        "k": 1,
        "n": 2,
        "p": 0.35
      },
      {
        "coverage": 0.3333333333333333,
        "k": 1,
        "n": 2,
        "p": 0.355
      }
    ],
    "lower_bound_shortfall": 0.00903225806451613,
    "lower_probability": 0.88,
    "n": 30,
    "paper_lower_index_ceil_n_p": 27,
    "paper_upper_index_ceil_n_p": 28,
    "upper_probability": 0.92
  },
  "qualification": "For continuous exchangeable scores, the q_L threshold has exact coverage ceil(n p)/(n+1), which can be below p. At n=30, alpha=0.1, alpha_tol=0.02, coverage is 27/31=0.87097 < 0.88.",
  "source_error_detected": true,
  "verdict": "FALSIFIED AS WRITTEN"
}
````
