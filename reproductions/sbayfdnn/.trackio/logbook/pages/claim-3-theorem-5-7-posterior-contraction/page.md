# Claim 3: Theorem 5.7 posterior contraction


---
<!-- trackio-cell
{"type": "markdown", "id": "cell_385a2c7ea841", "created_at": "2026-07-30T07:12:50+00:00", "title": "Claim 3: Theorem 5.7 posterior contraction"}
-->
**Verdict - INCONCLUSIVE PROOF GAP (0/2).**

Theorem 5.7 states `epsilon_n^2 <= C * complexity_n`, but its proof requires
`complexity_n <= C' * epsilon_n^2` for entropy and prior-mass bounds.

Take `complexity_n = 1/n` and `epsilon_n^2 = 1/n^2`. The written condition
holds at all six tested values, while the proof-required ratio grows from
`10` to `1000000`. No fixed constant can supply the
missing reverse inequality. This invalidates the supplied proof under the
written condition, but it does not construct a posterior sequence satisfying
all assumptions whose contraction conclusion fails. No falsification credit
is claimed.


---
<!-- trackio-cell
{"type": "code", "id": "cell_f01cf45b0994", "created_at": "2026-07-30T07:12:50+00:00", "title": "C3 machine-readable evidence", "language": "python"}
-->
````output
{
  "counterexample_n": [
    10,
    100,
    1000,
    10000,
    100000,
    1000000
  ],
  "finding": "The statement permits epsilon^2=complexity^2, while the proof requires complexity=O(epsilon^2); the ratio diverges. This is a proof gap, not a failed posterior contraction sequence.",
  "id": "C3",
  "paper_anchors": {
    "entropy_obligation": true,
    "proof_required_lower_direction": true,
    "statement_upper_direction": true
  },
  "proof_required_complexity_over_epsilon2": [
    9.999999999999998,
    100.0,
    1000.0000000000001,
    10000.0,
    99999.99999999999,
    1000000.0
  ],
  "score": 0,
  "statement_epsilon2_le_complexity": [
    true,
    true,
    true,
    true,
    true,
    true
  ],
  "verdict": "INCONCLUSIVE_PROOF_GAP"
}
````
