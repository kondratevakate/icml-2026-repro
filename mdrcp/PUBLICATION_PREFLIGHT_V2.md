# MDRCP publication preflight v2

Date: 2026-07-29

Canonical Space:
`kondratevakate/repro-multi-distribution-robust-conformal-prediction`

Paper: Multi-Distribution Robust Conformal Prediction (`Vv4XRZDMM0`,
arXiv `2601.02998`)

## Locked score forecast

- Previous judged score: `2/12`.
- Forecast before publication: `5-6/12`.
- Point estimate: `6/12`.
- Expected gain: `+3 to +4`.
- C1: expected `2/2` from the previously judged finite-sample audit.
- C2: expected `2/2` from the new independent Theorem 2 audit.
- C3: expected `1-2/2`; the finite-alphabet instantiation is complete, but
  the judge may score theorem-instantiation evidence as toy.
- C4-C6: expected `0`; exact official protocols are not complete.

## Evidence gate

### C2: Theorem 2

- 500 independently generated finite-measure problems.
- Maximum primal-dual gap: `2.23e-8`.
- Maximum complementary-slackness error: `2.23e-8`.
- Maximum superlevel-structure violation: `1.26e-8`.
- At least one active source in `500/500` problems.
- Uniform-weight destructive control strictly suboptimal in `500/500`.

### C3: Theorem 3

- Eight sample sizes from 100 to 25,000 per source.
- 20 independent seeds per sample size.
- Sup shared-score error falls from `3.92` to `0.392`.
- Calibrated-set symmetric difference falls from `18.45` to `1.80`.
- Largest-sample set difference is below the three-outcome oracle boundary.
- Frozen-weight destructive control has `2.34x` larger shared-score error at
  the largest sample.

## Negative evidence and scope

- The official Linear command needs repository, `model`, and `notebook`
  directories on `PYTHONPATH`.
- A one-trial combined Linear probe produced no result artifact after about
  20 minutes on local Windows CPU and was stopped.
- The official 100-trial classification and regression panels are not claimed.
- FMoW data and checkpoints were not used.
- Peer logbooks informed target selection only; no peer code or outputs are
  used as evidence.

## Reproduction gate

Run from the publish bundle:

```powershell
cd .trackio\logbook\repro_mdrcp_v2
python run_all.py
```

Expected terminal statuses:

- `theorem2: passed`
- `theorem3: passed`

Protocol and evidence SHA-256:

| File | SHA-256 |
| --- | --- |
| `audit_theorem2.py` | `41B14D57FFA082F002B497B3CAA1ECE23209C54F6B7F23AC7BB47A5A8FD36857` |
| `audit_theorem3.py` | `ED69C76FB0071E0C64C2ABB82D9381D5D01D8C324677AFDFCE00052F724DA0D4` |
| `run_all.py` | `161BFF8E2E149696BC7F4FDBE3487C92F55D9E4A60715D63E8F8B3EDDA5C238C` |
| `results/theorem2.json` | `7D9BC834272AD6EB879F60DF692A586A48A7AF2665EA89135B9C3EE6BAFD7B2D` |
| `results/theorem3.json` | `E862C3E29BD1F18A1C48D5F61999C46B61CF76346B24BD8E77B92C883A772219` |

Official repository commit:
`6a7d5ba990c98aff3a9510bc40357254f3107e68`.

## Publication decision

GO for updating the existing canonical Space. Do not create a duplicate Space.
The publication must preserve C4-C6 as pre-verdict or not attempted.

## Publication verification

- Published canonical Space SHA:
  `178965c8297b9b25b29b8e7171425ec816faef77`.
- Runtime state: `RUNNING`.
- Rendered static endpoint: HTTP `200`.
- Remote C2 and C3 pages contain the intended verdicts.
- Remote protocol and result hashes match every SHA-256 value above.
- Challenge tags include `icml2026-repro` and `paper-Vv4XRZDMM0`.
