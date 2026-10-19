# Claim 1: set-stability criterion


---
<!-- trackio-cell
{"type": "markdown", "id": "cell_d19b12fbae67", "created_at": "2026-07-29T16:17:42+00:00", "title": "Claim 1: set-stability criterion"}
-->
**Verdict - VERIFIED.**

The independent experiment generated 4000 calibration datasets
and 500 test covariates per construction.
The variance of conditional mean set size was
`0.547449`. Raw per-test variance was
`1.260903` because it additionally
contains `0.713454` of within-
construction test-point noise.

The law-of-total-variance residual was
`4.441e-16`. This verifies the paper's criterion
and falsifies the mutation that substitutes raw set-size variance.


---
<!-- trackio-cell
{"type": "code", "id": "cell_73d41bed04e3", "created_at": "2026-07-29T16:17:42+00:00", "title": "C1 machine-readable evidence", "language": "python"}
-->
````output
{
  "evidence": {
    "calibration_size": 30,
    "checks": {
      "law_of_total_variance_holds": true,
      "mutation_adds_within_construction_noise": true,
      "raw_variance_is_larger": true
    },
    "conformal_probability": 0.93,
    "law_total_variance_error": 4.440892098500626e-16,
    "mean_within_construction_variance": 0.7134538198404176,
    "raw_per_test_variance_mutation": 1.2609030965721464,
    "repeats": 4000,
    "seed": 20260729,
    "stability_variance": 0.5474492767317293,
    "test_points_per_construction": 500
  },
  "qualification": "The proposed stability criterion is the between-construction component; raw set-size variance adds within-construction test-covariate noise.",
  "verdict": "VERIFIED"
}
````
