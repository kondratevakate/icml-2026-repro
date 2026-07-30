# Claim 1: spike-and-slab region selection


---
<!-- trackio-cell
{"type": "markdown", "id": "cell_8571915371e0", "created_at": "2026-07-30T07:12:50+00:00", "title": "Claim 1: spike-and-slab region selection"}
-->
**Verdict - VERIFIED (2/2).**

An independent group-normal Bayes derivation agrees with the released plug-in
PIP at all `1201` tested column norms. Maximum absolute
error is `7.22e-15`, and the analytic selection
threshold gives PIP `0.500000000000007`.

An independent clamped-knot support implementation also matches the released
basis-to-region mapping exactly. This verifies the method mechanism, not the
paper's broad empirical superiority claim.


---
<!-- trackio-cell
{"type": "code", "id": "cell_f82ce3ce0f83", "created_at": "2026-07-30T07:12:50+00:00", "title": "C1 machine-readable evidence", "language": "python"}
-->
````output
{
  "analytic_norm2_threshold": 0.0036393783154664815,
  "id": "C1",
  "interval_mapping_max_error": 0.0,
  "mapped_intervals": [
    [
      0.0,
      0.15000000000000002
    ],
    [
      0.65,
      1.0
    ]
  ],
  "max_pip_absolute_error": 7.216449660063518e-15,
  "pip_at_analytic_threshold": 0.5000000000000071,
  "pip_grid_points": 1201,
  "score": 2,
  "selected_basis_probe": [
    0,
    1,
    2,
    17,
    18,
    19
  ],
  "verdict": "VERIFIED"
}
````
