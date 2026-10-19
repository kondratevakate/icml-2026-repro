# Claim 4: policy embedding pipeline


---
<!-- trackio-cell
{"type": "markdown", "id": "cell_4bb6a106d45e", "created_at": "2026-07-29T15:43:54+00:00", "title": "Claim 4: policy embedding pipeline"}
-->
**Paper claim.** "PEQ-Net represents each policy using Gaussian-kernel mean embeddings, pairwise MMD and metric MDS, then encodes the policy tail to condition shared Q-functions."

**Verdict - VERIFIED WITH QUALIFICATION.**

An independent implementation applied five threshold policies to
240 fixed synthetic histories, formed history-action
samples, computed biased empirical Gaussian-kernel MMD, and ran metric MDS on
the resulting dissimilarity matrix.

| Diagnostic | Result |
| --- | ---: |
| Distance-order rank correlation | 1.000 |
| Relative MDS stress against supplied `MMD^2` | 0.295 |
| Max error if embedding distance is called exact `MMD` | 0.222 |
| Finite, symmetric, zero diagonal | True |

The policy-distance ordering is perfectly preserved in this controlled run,
and the source algorithm independently confirms the reverse-time policy-tail
encoder and shared-Q conditioning path.

**Qualification.** The method defines the MDS input as `MMD^2` and promises
approximate preservation. The theorem proof later asserts exact equality
between embedding distance and `MMD`. The reproduced pipeline supports the
architecture claim but not that exact mathematical identity.


---
<!-- trackio-cell
{"type": "code", "id": "cell_20fdc589bfb7", "created_at": "2026-07-29T15:43:54+00:00", "title": "C4 machine-readable evidence", "language": "python"}
-->
````output
{
  "checks": {
    "all_values_finite": true,
    "mds_is_not_exact_identity_to_mmd": true,
    "policy_ordering_preserved": true,
    "symmetric_dissimilarity": true,
    "zero_diagonal": true
  },
  "embedding": {
    "checks": {
      "all_values_finite": true,
      "mds_is_not_exact_identity_to_mmd": true,
      "policy_ordering_preserved": true,
      "symmetric_dissimilarity": true,
      "zero_diagonal": true
    },
    "distance_ordering_correlation": 1.0,
    "embedding_coordinates": [
      [
        -0.008911504824772678,
        -0.02733337056359062
      ],
      [
        -0.002336745734407532,
        -0.007169696703265269
      ],
      [
        -0.00042654613399407753,
        -0.0013095070670614964
      ],
      [
        0.0023522186187848924,
        0.007216414434355551
      ],
      [
        0.009322578074389394,
        0.028596159899561838
      ]
    ],
    "max_error_if_treated_as_exact_mmd": 0.22166921530100492,
    "median_pairwise_distance": 2.3132520020037983,
    "mmd2_matrix": [
      [
        0.0,
        0.009939583534236962,
        0.019108178298730905,
        0.036021220805758425,
        0.07867802298581295
      ],
      [
        0.009939583534236962,
        0.0,
        0.0015936771385103299,
        0.008970988955112169,
        0.037079337345071406
      ],
      [
        0.019108178298730905,
        0.0015936771385103299,
        0.0,
        0.0032213619137267946,
        0.024366621335585203
      ],
      [
        0.036021220805758425,
        0.008970988955112169,
        0.0032213619137267946,
        0.0,
        0.010263087534899018
      ],
      [
        0.07867802298581295,
        0.037079337345071406,
        0.024366621335585203,
        0.010263087534899018,
        0.0
      ]
    ],
    "n_histories": 240,
    "paper_bandwidth_parameter": 0.21614592770994562,
    "relative_stress_against_mmd2": 0.2951098047391738,
    "seed": 20260729,
    "thresholds": [
      0.25,
      0.4,
      0.5,
      0.6,
      0.75
    ]
  },
  "qualification": "The pipeline is reproduced, but metric MDS only approximately preserves the supplied MMD-squared dissimilarities; it is not exact equality to MMD.",
  "verdict": "VERIFIED"
}
````
