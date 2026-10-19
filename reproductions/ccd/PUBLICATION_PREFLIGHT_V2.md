# Conditional Coverage publication preflight v2

Date: 2026-07-29

Canonical Space:
`kondratevakate/repro-conditional-coverage-diagnostics-for-conformal-prediction`

Paper: Conditional Coverage Diagnostics for Conformal Prediction
(`vaApZm6MKM`, arXiv `2512.11779`)

## Locked score forecast

- Previous judged result: `4/4` across two exposed claims.
- Forecast before publication: `11-12/12`.
- Point estimate: `12/12`.
- Expected gain: `+7 to +8`.
- Main risk: C2 or C5 may be scored as toy because they replay official
  released outputs rather than retrain the underlying models.

## Claim gate

| Claim | Evidence | Gate |
| --- | --- | --- |
| C1 ERT metric family | Exact L1/L2/KL identities on 99 coverages and 100 predictions | PASS |
| C2 LightGBM power | 5,600 official rows; `68.929%` vs `33.671%`; `2.047x` | PASS, released-output replay |
| C3 synthetic convergence | Prior independent 8D run; 92% vs 5% of truth | PASS |
| C4 asymmetric split | Estimate `0.0445/0.0464`; analytic `0.0460/0.0488` | PASS |
| C5 classification split | 80 official rows; 8/8 groups diverge; additive residual `8.94e-8` | PASS, released-output replay |
| C6 Algorithm 1 | 20 seeds, five folds; null false signal `0.08886` to `0` | PASS |

## Artifact provenance

Official experiments commit:
`39a99dcad92205a15d93f2c5fec40c76540abf1c`.

Official CSV SHA-256:

- `classifier_results.csv`:
  `070257695F54BD4F2C0D2A8911A765AD49582DC776ED6B235723D4CD43A1230B`
- `classification_results.csv`:
  `E6261F6F8C26B5AF98BF0B6297E38460829F46086E58D9773D7E415E55871B83`

Protocol and result SHA-256:

| File | SHA-256 |
| --- | --- |
| `audit_metric_identities.py` | `6070D0811D2B8C68C86758C58AE918F4850C28EE33E8AA465D23488B723D9E90` |
| `audit_algorithm1_crossfit.py` | `38A01ABF9E69A7391CDC2A5A5178D62C884E157B9DDD5047AD62D998C676132D` |
| `audit_released_results.py` | `B02B687FA4F9FEE394BB918E7297DF21A8B1315661F45D6ACEAC629BF1AAA6C7` |
| `results/metric_identities.json` | `576B995AF6B145E5384A2AB5CD8576E642A8F94CB431114A86551435077E1FC6` |
| `results/algorithm1.json` | `DA3B59F811A85CEE94588EE28D7D12194D34B52F13C6CF0C0346E9C67428B05E` |
| `results/released_results.json` | `19ACFFAD671D0151171C506FCDE2E4D482AD4FB90EE741D848BE73C0DB88BEAC` |

## Reproduction gate

```powershell
cd .trackio\logbook\repro_ccd_v2
python run_all.py
```

All three audits return `status: passed`. The canonical logbook validator also
passes with six claim pages and the conclusion last.

## Publication decision

GO for updating the existing canonical Space. Do not create a duplicate.

## Publication verification

- Published canonical Space SHA:
  `4d6129fc368f2e93d86105849cc812a22a460408`.
- Runtime state: `RUNNING`.
- Rendered static endpoint: HTTP `200`.
- Remote `logbook.json` contains exactly six claim pages and conclusion last.
- Obsolete two-claim pages and Python bytecode were removed from the Hub.
- Remote protocol and result hashes match every SHA-256 value above.
- Challenge tags include `icml2026-repro` and `paper-vaApZm6MKM`.
