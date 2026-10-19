# Frozen Forecast

Paper: *Learning Treatment Allocations with Risk Control Under Partial
Identifiability* (`9kWVQtpvJi`, arXiv `2505.08378v1`).

Freeze date: 2026-07-28 (before the recorded local execution outcomes).
Reconfirmed for the 2026-07-29 publication pass without consulting any
leaderboard or third-party verdict.

## Locked forecast

| Claim | Forecast | Expected points |
| --- | --- | ---: |
| C1: Lemma 4.1 risk bounds | Decisive independent verification is feasible by exact finite enumeration. | 2 |
| C2: Theorem 4.3 certificate | High proof-risk; score only if a full proof or decisive counterexample survives adversarial review. | 0-2 |
| C3: synthetic risk-control experiment | Decisive paper-scale independent reproduction is feasible on CPU. | 2 |
| C4: Tennessee STAR | Not attempted without the exact third-party cohort and preprocessing. | 0 |
| C5: International Stroke Trial | Not attempted without the exact 18,298-row cohort and 23-covariate preprocessing. | 0 |

Conservative forecast: **4/10**.

Upside forecast: **6/10** only if C2 receives a decisive full verdict.

Expected local effort: **6-8 hours**, or about **0.5-1.0 points/hour**. This is
below the original shortlist's aspirational throughput threshold, but it is
the highest-value unfinished CPU-local package after ROCP and the already
completed Cox package; SwitchCraft requires an unavailable GPU.

## Evidence firewall

The forecast and all subsequent claim verdicts must be based only on:

1. the paper PDF/source;
2. public author or official artifacts, if any;
3. independently executable local checks;
4. explicit destructive or mutation controls.

Leaderboard entries, other reproductions, and third-party verdicts are excluded
from both selection evidence and claim evidence.
