# Claim 1: embedding-conditional prior


---
<!-- trackio-cell
{"type": "markdown", "id": "cell_a00b3e4067cf", "created_at": "2026-07-30T08:08:15+00:00", "title": "Claim 1: embedding-conditional prior"}
-->
**PARTIAL - 1/2.** The release adds an embedding-dependent offset
to global parameters, but normalizes every non-zero update to a fixed norm.
The resulting map is scale-invariant away from zero and discontinuous at
zero, unlike `theta+Wz` in the paper.


---
<!-- trackio-cell
{"type": "code", "id": "cell_19ba98ee1605", "created_at": "2026-07-30T08:08:15+00:00", "title": "Claim 1: embedding-conditional prior evidence", "language": "python"}
-->
````output
{
  "global_norm": 5.0,
  "adaptation_scale": 0.2,
  "offset_z0": [
    0.0,
    0.0
  ],
  "offset_z_epsilon": [
    1.0,
    0.0
  ],
  "offset_z1": [
    1.0,
    0.0
  ],
  "offset_z2": [
    1.0,
    0.0
  ],
  "scale_invariance_error": 0.0,
  "jump_norm_at_zero": 1.0,
  "epsilon": 1e-09,
  "exact_tv_jump_for_unit_covariance_gaussians": 0.3829249225480261,
  "paper_linear_lipschitz_rhs_M1_W1_sigma1": 5e-10,
  "released_map_violates_linear_bound": true
}
````
