# ROCP Publication Preflight

Prepared: 2026-07-28. Published: 2026-07-29.

Space:
`kondratevakate/repro-optimal-decision-making-based-on-prediction-sets`.
Published Space SHA: `92314dce6dfaace6c6a91e46035d5050127e913d`.

## Score forecast

| Claim | Prepared verdict | Expected points now | Full path |
| --- | --- | ---: | --- |
| C1 fixed-set minimax policy | verified | 2 | Complete locally |
| C2 optimal set construction | not attempted | 0 | Excluded from throughput |
| C3 ROCP coverage and risk | verified | 2 | Exact finite audit plus full cached coverage |
| C4 medical and safety-critical gains | verified | 2 | Full COVID Lambda0/Lambda1 and BDD runs |

Updated publication forecast: `6/8`.

Current status: canonical Space published; local evidence is ready for an
in-place Space update.

## Passed

- Four unit tests.
- 28,950 primal-LP versus closed-form comparisons; max error `1.42e-14`.
- Zero minimizing-policy-set failures.
- 14,850 exhaustive exchangeability orbit/alpha checks.
- Official commit pinned to `3ee0cf6e393d2e434368bc1fc7fe3abd03ed493f`.
- Two paper/code divergences reproduced on concrete inputs.
- Canonical Trackio logbook validator passes.
- COVID seeds 23-42 completed under both loss matrices.
- BDD split seeds 23-42 completed from the 10,000-record cached artifact.
- Claim validator passes all structural, coverage, and anchored C4 checks.

## Publication gate used

The following empirical gate is now complete:

1. COVID seeds 23-42, both `Lambda0` and `Lambda1`, seven alpha values, and all
   paper baselines.
2. BDD raw 10,000-record artifact, 20 deterministic splits, and all four paper
   metrics.
3. Paper-faithful candidate-wise `(n+1)` calibration is used or compared
   explicitly against the released global-beta implementation.
4. Exact miscoverage and critical-mistake deltas are reported, not only plots.

Known limitation: the released empirical implementation uses global beta,
while the paper-faithful candidate-wise Algorithm 1 is covered by the
independent finite exchangeability audit.

## Integrity hashes

```text
audit_claim1.py                 0FFD20DC7AD0E8CD7EE61F51C5A8746FD66F1361D4FE0882D95F560646438F7D
audit_claim3_exchangeability.py F2E57C4A090A40B53E18FAAD3B59F99BD5B52DDDE3B23F2CA508D933B52CD476
audit_official_divergence.py    867FA42E5B52733812BAB596D05D0E94341C6E2173FF5E67EE78434D7E9E9C1E
results/claim1.json             6C491D1237FFE339686D2D7A7B5F8FAD9F8CBE11CF105D3CCDC26087A4292D49
results/claim3_exchangeability.json 046CED382E67107179E8334E9816430BE7B0E5B7B894E508B97DA1B44BBD7226
results/official_divergence.json 6926EE0007BCC84362557BC6B79A60ED1D2CDAC4BC984AAF96E411032FCCA104
run_cached_empirics.py             85C085669ED9F43213C2E9280288515E2B325B85E5678E4C43F77F27A0DFDB8A
validate_cached_empirics.py        5BF7E3647C36C43E4DD9678B905C1F3B745F27776AB8F75B5526725AF1E856BC
EMPIRICAL_CLAIM_AUDIT.md           65C88E081CB4BA93D271C437EC5B7E7C1A6C3CDC51AD9D1F9E0142000B5382FC
results/cached/claim_validation.json 8DCF1E0F7F86B64930A789DF5A46C9566CCC87C38D76F287FAC912BC691EBA48
results/cached/covid/lambda0/summary.json 9115CDCDDCD8EBB2A87BDEC93BF9CF263A11F7F473B6DFBC77442012D44A2B71
results/cached/covid/lambda1/summary.json F9FB902013826A9B09A78BA26C519662A7999608ACB6AF94116126543B02F0FC
results/cached/bdd/default/summary.json E1CAB6A41C4AB386B9437EE0C8EB60B68F268F915247D44DD232B6AAA5398A4D
```
