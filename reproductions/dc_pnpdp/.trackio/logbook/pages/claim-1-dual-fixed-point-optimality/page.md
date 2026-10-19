# Claim 1: dual fixed-point optimality


---
<!-- trackio-cell
{"type": "markdown", "id": "cell_a9e10d06be14", "created_at": "2026-07-30T07:55:08+00:00", "title": "Claim 1: dual fixed-point optimality"}
-->
**VERIFIED - 2/2.** An independent scalar quadratic satisfies
consensus, both deterministic ADMM update equations, and the original
objective's stationarity condition with effective weight `lambda=rho*gamma`
to numerical precision.


---
<!-- trackio-cell
{"type": "code", "id": "cell_8b94878ffa03", "created_at": "2026-07-30T07:55:08+00:00", "title": "Claim 1: dual fixed-point optimality evidence", "language": "python"}
-->
````output
{
  "x": 1.4574759945130316,
  "u": 0.4663923182441702,
  "consensus_residual": 0.0,
  "data_update_residual": 3.3306690738754696e-16,
  "prox_update_residual": 5.551115123125783e-17,
  "original_objective_residual": 2.220446049250313e-16
}
````
