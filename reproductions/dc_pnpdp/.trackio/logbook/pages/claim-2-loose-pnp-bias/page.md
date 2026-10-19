# Claim 2: loose-PnP bias


---
<!-- trackio-cell
{"type": "markdown", "id": "cell_fc483913efc4", "created_at": "2026-07-30T07:55:08+00:00", "title": "Claim 2: loose-PnP bias"}
-->
**PARTIAL - 1/2.** The loose fixed-point equation is reproduced,
but an exact proximal example shows it is stationarity of a Moreau-envelope
objective, not the original regularized objective. The main theorem,
appendix statement, and proof use incompatible scalings.


---
<!-- trackio-cell
{"type": "code", "id": "cell_f63cf38b9694", "created_at": "2026-07-30T07:55:08+00:00", "title": "Claim 2: loose-PnP bias evidence", "language": "python"}
-->
````output
{
  "loose": {
    "x": 1.5096878363832076,
    "denoised": 1.1437029063509148,
    "fixed_point_equation_residual": 2.7755575615628914e-16,
    "original_objective_residual": 0.15224973089343352,
    "moreau_envelope_objective_residual": 2.7755575615628914e-16,
    "interpretation": "The released theorem's algebraic loose fixed-point equation holds, but for an exact proximal denoiser it is the stationarity equation of a Moreau-envelope objective, not the original regularized objective."
  },
  "scaling": {
    "main_theorem_omits_effective_lambda": true,
    "appendix_sets_lambda_rho_gamma": true,
    "loose_statement_uses_rho_over_sigma2": true,
    "loose_proof_uses_rho_times_residual": true
  }
}
````
