# Claim 3: stability-rate theorem


---
<!-- trackio-cell
{"type": "markdown", "id": "cell_ea45b6f785f1", "created_at": "2026-07-29T16:17:42+00:00", "title": "Claim 3: stability-rate theorem"}
-->
**Verdict - NOT ESTABLISHED UNDER STATED ASSUMPTIONS.**

An independent Gaussian surrogate reproduced the stated
`m^-1 + [n(1+lambda)^2]^-1` algebra with maximum relative error
`0.0025` over 200,000 repetitions.
Ignoring lambda removed the variance gain, as expected.

The theorem proof itself explicitly invokes local second-order smoothness not
listed in its assumptions. It also promotes an `O_p` fixed-point rate to a
second-moment rate without a tail or uniform-integrability argument.
Therefore the rate is supported as algebra but not established under the
theorem's stated assumptions.


---
<!-- trackio-cell
{"type": "code", "id": "cell_72a53c41f176", "created_at": "2026-07-29T16:17:42+00:00", "title": "C3 machine-readable evidence", "language": "python"}
-->
````output
{
  "evidence": {
    "checks": {
      "mutation_removes_lambda_gain": true,
      "rate_algebra_matches_surrogate": true,
      "variance_decreases_with_lambda": true
    },
    "empirical_variance": [
      0.03524488575638624,
      0.02947601723295627,
      0.01677853546214258,
      0.01031547067830152,
      0.004083511456640899,
      0.0022811525249232536
    ],
    "lambda": [
      0.0,
      0.1,
      0.5,
      1.0,
      3.0,
      10.0
    ],
    "lambda_ignored_mutation_variance": [
      0.03524488575638624,
      0.03524488575638624,
      0.03524488575638624,
      0.03524488575638624,
      0.03524488575638624,
      0.03524488575638624
    ],
    "m": 500,
    "max_relative_error": 0.0025032333098233784,
    "n": 30,
    "rate_formula": [
      0.035333333333333335,
      0.02954820936639118,
      0.016814814814814817,
      0.010333333333333333,
      0.004083333333333333,
      0.002275482093663912
    ],
    "repeats": 200000,
    "seed": 20260730
  },
  "qualification": "The rate algebra is reproducible in a transparent surrogate, but the paper explicitly invokes unstated local second-order smoothness and moves from O_p to a second-moment rate without a tail or uniform-integrability argument.",
  "source_assumption_gap_detected": true,
  "verdict": "NOT ESTABLISHED UNDER STATED ASSUMPTIONS"
}
````
