# Claim 1: evidence accumulation and calibration


---
<!-- trackio-cell
{"type": "markdown", "id": "cell_64591a9395f2", "created_at": "2026-07-30T07:15:06+00:00", "title": "Claim 1: evidence accumulation and calibration"}
-->
**PARTIAL - 1/2.** A synthetic check reproduces the discounted
logit accumulation recurrence to `0.0e+00`.
The Appendix A proof is invalid as written: its assumption contributes
`2 * epsilon_suff`, but the displayed bound uses
`2 * epsilon_suff^2`. The released paper has no executable implementation.


---
<!-- trackio-cell
{"type": "code", "id": "cell_3fe3d71199a9", "created_at": "2026-07-30T07:15:06+00:00", "title": "Claim 1: evidence accumulation and calibration evidence", "language": "python"}
-->
````output
{
  "accumulation": {
    "scores": [
      0.6,
      0.8,
      0.7
    ],
    "beta": 0.5,
    "direct": 1.6418113179741898,
    "recurrent": 1.6418113179741898,
    "absolute_difference": 0.0,
    "states": [
      0.4054651081081642,
      1.5890269151739729,
      1.6418113179741898
    ]
  },
  "proof": {
    "assumption_term": "E[(p_S(S)-p*(tau))^2] <= epsilon_suff",
    "supported_sufficiency_term": "2 * epsilon_suff",
    "printed_sufficiency_term": "2 * epsilon_suff^2",
    "counterexample_epsilon": 0.2,
    "supported_numeric_upper_term": 0.4,
    "printed_numeric_term": 0.08000000000000002,
    "printed_term_follows_from_assumption": false
  }
}
````
