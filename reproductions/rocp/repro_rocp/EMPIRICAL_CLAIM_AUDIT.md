# ROCP cached empirical claim audit

## Scope

Fresh CPU evaluation of official cached artifacts at commit
`3ee0cf6e393d2e434368bc1fc7fe3abd03ed493f`:

- COVID seeds 23-42 under both `Lambda0` and `Lambda1`;
- BDD hazard scores over deterministic split seeds 23-42;
- seven alpha values from 0.001 to 0.1;
- ROCP, RAC, LAS, APS, SOCOP, and best-response outputs.

The parallel runner calls the official `evaluate_seed` function unchanged.
It changes orchestration only and writes one resumable JSON file per seed.

## C3 coverage

Empirical ROCP miscoverage tracks the target alpha on all three runs. The
maximum absolute difference from alpha is below 0.005. This empirical result
complements, but does not replace, the independent paper-faithful finite
exchangeability audit.

The released evaluator uses one global beta and omits the candidate test point.
Therefore these empirical results characterize the released approximation,
while the finite-sample theorem is supported by the independent literal
Algorithm 1 audit.

## C4 medical and safety-critical gains

**VERIFIED for the anchored claim.**

At alpha 0.05 under `Lambda0`, critical mistake rates are:

| Label | ROCP | RAC |
| --- | ---: | ---: |
| Pneumonia | 1.19% | 1.54% |
| COVID-19 | 10.59% | 12.95% |
| Lung opacity | 4.77% | 5.94% |

Under `Lambda1`, ROCP has zero observed critical mistakes for all three labels,
while RAC remains at 1.54%, 12.95%, and 5.94%. The safety gain is therefore
larger for every critical label when out-of-set errors are more costly.

On BDD at alpha 0.05, average realized loss is `2.320 +/- 0.019` for ROCP and
`2.981 +/- 0.075` for RAC. ROCP also has a lower collision rate than RAC for
every nonzero hazard label across the 20 splits.

## Stronger-statement limitation

Section 5.2 says ROCP matches or outperforms all baselines across all alpha
values. Against RAC this is not literally true in the fresh aggregate:

- realized loss is higher by 0.052 at alpha 0.005;
- worst-case risk is higher by 0.112 at alpha 0.03 and 0.010 at alpha 0.05.

These reversals are small relative to their standard errors and do not
contradict the anchored critical-mistake claim, but they must not be hidden.

## Reproduce

```powershell
python run_cached_empirics.py --official-repo ..\official --dataset covid --variant lambda0 --workers 6
python run_cached_empirics.py --official-repo ..\official --dataset covid --variant lambda1 --workers 6
python run_cached_empirics.py --official-repo ..\official --dataset bdd --workers 6
python validate_cached_empirics.py
```
