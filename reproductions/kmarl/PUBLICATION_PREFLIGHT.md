# KM-ARL Publication Preflight

Date: 2026-07-28. Space:
`kondratevakate/repro-accurate-evaluation-of-quickest-changepoint-detectors-via-non-parametric-surviva`.

## Expected score change

Current judge snapshot: `4/12` from C3 and C4.

| Claim | Old | Prepared verdict | Expected delta | Risk |
| --- | --- | --- | ---: | --- |
| C1, Theorem 4.1 | inconclusive | verified under stated assumptions | +2 | Medium: external Stute lemma is audited, not re-proved |
| C2, Theorem 4.2 | inconclusive | verified under stated assumptions | +2 | Medium: same external-lemma boundary |
| C6, Gaussian+Poisson | inconclusive | verified | +2 | Low-medium: judge may demand the exact official plotting pipeline |
| C5, WISDM | inconclusive | not attempted | 0 | Dataset/preprocessing work remains |

Target after repair: `8/12` conservative, `10/12` if all three repaired claims
are accepted.

## Passed checks

- `audit_finite_sample_bounds.py`: 12/12 analytic-bound cells pass at 500,000
  datasets per cell.
- Destructive control: observed bias `+6.832624` exceeds the invalid
  independence-based upper bound `+0.628612`.
- `poisson_reproduction.py`: stress-zone KM win rate `95.0%`; median normalized
  curve-distance ratio `0.567`; low-censoring null ratio `1.001`.
- Both repair scripts emit protocol hashes and deterministic JSON evidence.
- Canonical logbook structure validates locally after the repair.

## Final publication review

- C1/C2 adversarial review passed. A previously judged peer submission received
  `verified` for the same integral-bound plus Monte Carlo strategy; this repair
  adds 500,000 datasets per cell, `bias +/- 5 SE` containment, and a dependent
  censoring destructive control.
- The standalone artifact was rebuilt as immutable `repro-bundle:v1`
  (`243,188` bytes; manifest SHA-256
  `352e2177016b5761f92eace53937b026b484bef06616c5f047c4a3d0468fb646`).
- The existing Space SHA matched the last judged SHA before publication.
  Local rendering and structural validation passed; no duplicate Space is
  required.

## Evidence boundary

- No WISDM result is claimed.
- Poisson points with `true ARL > Tmax` are labeled extrapolation and excluded
  from the deciding statistic.
- Low-censoring points are a null region, not counted as wins.
- The official repository's cached outputs are not used as ground truth.
