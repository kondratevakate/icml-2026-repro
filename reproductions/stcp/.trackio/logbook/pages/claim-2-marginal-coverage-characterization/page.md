# Claim 2: marginal-coverage characterization


---
<!-- trackio-cell
{"type": "markdown", "id": "cell_e1a73935625b", "created_at": "2026-07-29T16:17:42+00:00", "title": "Claim 2: marginal-coverage characterization"}
-->
**Verdict - NOT ESTABLISHED AS WRITTEN.**

Theorem 4.2 defines `delta_S` from a random source-based estimator, but its
proof bounds an unconditional expectation by the realized `delta_S`. For a
Uniform[0,2] target score and a source quantile equal to 1 or 2 with equal
probability, the unconditional coverage deviation is
`0.25`. On the positive-probability
realization where the source quantile is 1, `delta_S=0`, so the proof's right
side for this step is zero.

Replacing realized `delta_S` with `E[delta_S]` repairs this specific step.
This is a concrete proof counterexample, not a claim that every corrected
version of the theorem is false.


---
<!-- trackio-cell
{"type": "code", "id": "cell_f0c31b100c33", "created_at": "2026-07-29T16:17:42+00:00", "title": "C2 machine-readable evidence", "language": "python"}
-->
````output
{
  "evidence": {
    "checks": {
      "delta_is_zero_on_positive_probability_event": true,
      "density_is_bounded_above_and_below": true,
      "expected_delta_repairs_this_step": true,
      "paper_realized_delta_step_fails": true
    },
    "paper_proof_rhs_on_zero_delta_realization": 0.0,
    "random_source_quantile_probabilities": [
      0.5,
      0.5
    ],
    "random_source_quantile_values": [
      1.0,
      2.0
    ],
    "realized_delta_values": [
      0.0,
      1.0
    ],
    "target_probability": 0.5,
    "true_score_distribution": "Uniform[0, 2]",
    "unconditional_coverage_deviation": 0.25,
    "valid_bound_using_expected_delta": 0.25
  },
  "qualification": "The proof bounds an unconditional expectation by the realized random delta_S. The numerical counterexample shows that step fails; replacing delta_S by E[delta_S] repairs this specific step.",
  "source_mismatch_detected": true,
  "verdict": "NOT ESTABLISHED AS WRITTEN"
}
````
