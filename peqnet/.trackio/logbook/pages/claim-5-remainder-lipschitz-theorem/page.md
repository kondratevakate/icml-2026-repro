# Claim 5: remainder Lipschitz theorem


---
<!-- trackio-cell
{"type": "markdown", "id": "cell_27503623f58f", "created_at": "2026-07-29T15:43:54+00:00", "title": "Claim 5: remainder Lipschitz theorem"}
-->
**Paper claim.** "Theorem 4.2 establishes a finite Lipschitz bound on the CATE second-order remainder contrast in terms of trajectory-level policy MMD under the stated model, encoder, outcome and density-product regularity assumptions."

**Verdict - FALSIFIED AS A LIPSCHITZ GUARANTEE.**

The appendix proof has three independent failures as a uniform Lipschitz
argument:

1. The subsequence-MMD lemma defines `C` as the numerator/denominator ratio
   after fixing each policy pair. That proves only a pair-specific tautology,
   not one finite constant uniform over policies.
2. The method specifies approximate metric-MDS preservation of `MMD^2`; the
   proof substitutes exact equality to `MMD`.
3. Strict positivity and bounded LTMLE fluctuations do not imply that the
   targeting map is Lipschitz in the initial Q estimates.

For the third point, use identical initial values `Q_i=Q_j=0.5`, clever
covariate one, strictly positive treatment probability 0.5, and bounded
policy-specific fluctuations `epsilon_i=0.8`, `epsilon_j=-0.8`. Logistic
targeting gives `0.689974` and
`0.310026`. The initial difference and the claimed
right-hand side are zero, while the targeted difference is
`0.379949`.

The theorem may be repairable with a uniform marginal-to-trajectory MMD bridge,
an exact or controlled embedding-distortion bound, and a Lipschitz condition
on the policy-specific targeting map. Those assumptions are absent.


---
<!-- trackio-cell
{"type": "code", "id": "cell_1d739ef502dc", "created_at": "2026-07-29T15:43:54+00:00", "title": "C5 machine-readable evidence", "language": "python"}
-->
````output
{
  "checks": {
    "approximate_exact_mds_mismatch_detected": true,
    "bounded_targeting_counterexample_passes": true,
    "pair_specific_constant_detected": true
  },
  "targeting_counterexample": {
    "bounded_fluctuations": [
      0.8,
      -0.8
    ],
    "checks": {
      "claimed_lipschitz_implication_fails": true,
      "fluctuations_are_bounded": true,
      "initial_models_identical": true,
      "strict_positivity_holds": true,
      "targeted_models_differ": true
    },
    "claimed_rhs_for_any_finite_constant": 0.0,
    "initial_difference": 0.0,
    "initial_q_values": [
      0.5,
      0.5
    ],
    "strictly_positive_treatment_probability": 0.5,
    "targeted_difference": 0.37994896225522495,
    "targeted_q_values": [
      0.6899744811276125,
      0.31002551887238755
    ]
  },
  "verdict": "FALSIFIED AS A LIPSCHITZ GUARANTEE"
}
````
