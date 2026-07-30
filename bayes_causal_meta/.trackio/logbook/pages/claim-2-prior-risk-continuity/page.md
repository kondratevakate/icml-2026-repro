# Claim 2: prior-risk continuity


---
<!-- trackio-cell
{"type": "markdown", "id": "cell_ea2cd5fd6785", "created_at": "2026-07-30T08:08:15+00:00", "title": "Claim 2: prior-risk continuity"}
-->
**VERIFIED - 2/2.** Exact equal-covariance Gaussian total
variation stays below the stated Pinsker bound at five distances. The
expert/causal/OOD decomposition follows directly from the triangle
inequality. This verification applies to the paper's linear prior.


---
<!-- trackio-cell
{"type": "code", "id": "cell_3f77d1621e38", "created_at": "2026-07-30T08:08:15+00:00", "title": "Claim 2: prior-risk continuity evidence", "language": "python"}
-->
````output
{
  "rows": [
    {
      "embedding_distance": 0.0,
      "exact_total_variation": 0.0,
      "paper_bound_M1_W1_sigma1": 0.0,
      "bound_holds": true
    },
    {
      "embedding_distance": 0.01,
      "exact_total_variation": 0.003989406181481644,
      "paper_bound_M1_W1_sigma1": 0.005,
      "bound_holds": true
    },
    {
      "embedding_distance": 0.1,
      "exact_total_variation": 0.039877611676744924,
      "paper_bound_M1_W1_sigma1": 0.05,
      "bound_holds": true
    },
    {
      "embedding_distance": 1.0,
      "exact_total_variation": 0.3829249225480261,
      "paper_bound_M1_W1_sigma1": 0.5,
      "bound_holds": true
    },
    {
      "embedding_distance": 3.0,
      "exact_total_variation": 0.8663855974622838,
      "paper_bound_M1_W1_sigma1": 1.5,
      "bound_holds": true
    }
  ],
  "all_bounds_hold": true,
  "error_decomposition_is_triangle_inequality": true
}
````
