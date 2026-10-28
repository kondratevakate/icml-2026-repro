# Claim 3: negative-transfer theorem


---
<!-- trackio-cell
{"type": "markdown", "id": "cell_d0e67cdf5abd", "created_at": "2026-07-30T08:08:15+00:00", "title": "Claim 3: negative-transfer theorem"}
-->
**PARTIAL - 1/2.** A condition-number-aware constant can recover
the sufficient condition. Appendix A.2 instead sets `C=kappa0/kappa`,
dropping `cond(W)`; a diagonal 2D example satisfies the stated norm
condition while failing the parameter-space inequality used by the proof.


---
<!-- trackio-cell
{"type": "code", "id": "cell_171c8abda568", "created_at": "2026-07-30T08:08:15+00:00", "title": "Claim 3: negative-transfer theorem evidence", "language": "python"}
-->
````output
{
  "singular_values": [
    10.0,
    1.0
  ],
  "condition_number": 10.0,
  "kappa": 1.0,
  "kappa0": 0.5,
  "z_norm": 1.0,
  "embedding_error_norm": 0.1,
  "appendix_constant_kappa0_over_kappa": 0.5,
  "condition_number_aware_constant": 0.05,
  "appendix_embedding_condition_holds": true,
  "required_parameter_condition_holds": false,
  "counterexample": true
}
````
