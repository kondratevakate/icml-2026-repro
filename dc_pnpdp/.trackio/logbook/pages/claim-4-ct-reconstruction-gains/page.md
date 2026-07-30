# Claim 4: CT reconstruction gains


---
<!-- trackio-cell
{"type": "markdown", "id": "cell_718bf1d6e887", "created_at": "2026-07-30T07:55:08+00:00", "title": "Claim 4: CT reconstruction gains"}
-->
**UNSUPPORTED - 0/2.** No CT data or cached metrics are released.
The quickstart runs DiffPIR with `w_tik=0`, while the paper reports positive
`rho=1e-5/sigma_t^2` for LACT. At zero penalty the exact released x-update
loses dependence on the consensus anchor.


---
<!-- trackio-cell
{"type": "code", "id": "cell_0bb088c3d77a", "created_at": "2026-07-30T07:55:08+00:00", "title": "Claim 4: CT reconstruction gains evidence", "language": "python"}
-->
````output
{
  "zero_penalty_anchor_difference": 0.0,
  "positive_penalty_anchor_difference": 0.5000000000000002,
  "zero_penalty_solution_error_vs_y": 0.0,
  "paper_requires_positive_rho": true,
  "release_example_w_tik": 0,
  "paper_reported_lact_w_tik": 1e-05
}
````
