# Claim 2: Theorem 5.4 approximation bound


---
<!-- trackio-cell
{"type": "markdown", "id": "cell_30a2b711081a", "created_at": "2026-07-30T07:12:50+00:00", "title": "Claim 2: Theorem 5.4 approximation bound"}
-->
**Verdict - FALSIFIED AS STATED (2/2).**

Theorem 5.4 quantifies over a network class with parameter bound `E_n`, but its
three stated assumptions impose no lower bound on `E_n`.

Set the strictly positive `E_n=1/n`, width one, `X(t)=1`, `beta(t)=1`, and
`g(u)=u`. The design, coefficient, and link assumptions hold. Every admissible
network output is `O(1/n)`, so its uniform error tends to one. Meanwhile the
claimed two-term rate tends to zero; on the audit grid, error/rate grows from
`8.91` to
`319.00` and diverges symbolically. Adding a sufficiently large
parameter-bound condition can repair this counterexample, but that condition
is absent from the theorem as stated.


---
<!-- trackio-cell
{"type": "code", "id": "cell_0058dddaf4fe", "created_at": "2026-07-30T07:12:50+00:00", "title": "C2 machine-readable evidence", "language": "python"}
-->
````output
{
  "counterexample": {
    "X_t": 1,
    "all_network_outputs_max_abs": [
      0.02,
      0.005,
      0.00125,
      0.0003125,
      7.8125e-05,
      1.953125e-05
    ],
    "beta_t": 1,
    "claimed_rate_without_constant": [
      0.11,
      0.052500000000000005,
      0.025625000000000002,
      0.01265625,
      0.0062890625,
      0.003134765625
    ],
    "error_lower_bound_over_rate": [
      8.909090909090908,
      18.95238095238095,
      38.97560975609756,
      78.98765432098764,
      158.99378881987576,
      318.9968847352025
    ],
    "g_u": "u",
    "hidden_width": 1,
    "parameter_bound_E_n": [
      0.01,
      0.0025,
      0.000625,
      0.00015625,
      3.90625e-05,
      9.765625e-06
    ],
    "sample_size_n": [
      100,
      400,
      1600,
      6400,
      25600,
      102400
    ],
    "scales_J_equals_H": [
      10,
      20,
      40,
      80,
      160,
      320
    ],
    "true_mean": 1,
    "uniform_error_lower_bound": [
      0.98,
      0.995,
      0.99875,
      0.9996875,
      0.999921875,
      0.99998046875
    ]
  },
  "exponent_combinations_checked": 60,
  "finding": "Theorem 5.4 does not lower-bound E_n. With positive E_n=1/n and width one, every admissible network output is O(1/n), while a constant nonzero truth satisfies the three stated assumptions.",
  "id": "C2",
  "max_exponent_composition_error": 1.3877787807814457e-17,
  "monotone_bound_failures": 0,
  "paper_anchors": {
    "bounded_network_class": true,
    "only_assumptions_1_to_3": true,
    "proof_decomposition": true,
    "relu_approximation": true,
    "spline_projection": true,
    "theorem_54": true
  },
  "score": 2,
  "verdict": "FALSIFIED_AS_STATED"
}
````
