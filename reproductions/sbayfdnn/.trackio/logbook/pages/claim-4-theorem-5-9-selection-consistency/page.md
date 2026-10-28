# Claim 4: Theorem 5.9 selection consistency


---
<!-- trackio-cell
{"type": "markdown", "id": "cell_556d2f8af451", "created_at": "2026-07-30T07:12:50+00:00", "title": "Claim 4: Theorem 5.9 selection consistency"}
-->
**Verdict - INCONCLUSIVE (0/2).**

The posterior inclusion argument controls `q_j`, but the proof then transfers
the result to MAP plug-in `hat q_j` by citing equivalence under an "appropriate
choice of prior hyperparameters." That condition is not included among
Theorem 5.9's stated assumptions and is not established in the paper.

This is a proof gap, not a standalone counterexample to the conclusion, so no
falsification credit is claimed.


---
<!-- trackio-cell
{"type": "code", "id": "cell_6292db0ffe4a", "created_at": "2026-07-30T07:12:50+00:00", "title": "C4 machine-readable evidence", "language": "python"}
-->
````output
{
  "finding": "The proof introduces posterior-to-MAP equivalence under an appropriate hyperparameter choice not stated among Theorem 5.9 assumptions.",
  "id": "C4",
  "paper_anchors": {
    "map_transition": true,
    "posterior_q": true,
    "stated_assumptions": true
  },
  "score": 0,
  "verdict": "INCONCLUSIVE"
}
````
