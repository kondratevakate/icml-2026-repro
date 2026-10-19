# Treatment-Risk Publication Preflight

Prepared: 2026-07-28. Fresh audit and publication: 2026-07-29.
Canonical Space:
`kondratevakate/repro-learning-treatment-allocations-with-risk-control-under-partial-identifiability`.

Published remote commit:
`b21e448c01075452e77cb47caca2da458e1e00b2`.

## Score forecast

| Claim | Prepared verdict | Expected points | State |
| --- | --- | ---: | --- |
| C1 Lemma 4.1 risk bounds | `VERIFIED` | 2 | Exact observational/trial audits complete |
| C2 Theorem 4.3 certificate | `PARTIAL` | 0-1 | Proof defect plus finite-grid repair |
| C3 synthetic risk control | `VERIFIED` | 2 | Main-text observational `1000 x 4` run complete; randomized-trial appendix excluded |
| C4 Tennessee STAR | `NOT ATTEMPTED` | 0 | Exact third-party cohort unavailable |
| C5 International Stroke Trial | `NOT ATTEMPTED` | 0 | Exact third-party cohort unavailable |

Defensible forecast: `4-5/10`. Do not claim six points from C2: Eq. (18) is
invalid, while the repair needs a finite ordered grid and conditional coverage.

## Passed

- Lemma 4.1: `153,600` exact population-risk checks and `115,200` exact
  treatment-risk checks; zero excess and zero `Gamma=1` equality error.
- Underestimated-`Gamma` controls violate observational and trial
  treatment-risk bounds by `+0.4` and `+0.5`.
- Eq. (18) counterexample: empirical equality `0.25`, fresh expectation `0.50`.
- Finite-grid repair: zero counterexamples across `1,396,054` UCB patterns.
- Bentkus Eq. (15): valid coverage in `165/165`; `0.98 x UCB` and raw-mean
  mutations fail.
- Synthetic full run: valid arms control `8/8` tolerances with minimum coverage
  `0.988`, `1.000`, and `0.998`.
- Invalid confounded `Gamma=1` arm controls only `5/8`; minimum coverage `0.257`.
- Standalone reproduction bundle is assembled.
- Standalone bundle rerun matches canonical JSON (excluding wall time) and all
  raw arrays element-for-element.
- Fresh 2026-07-29 rerun matches the canonical semantic JSON; all raw arrays
  are element-for-element identical with maximum absolute difference `0`.
- Canonical Trackio logbook is assembled and the local validator passes.
- Local Trackio render has no raw markup; the PNG asset link returns HTTP 200
  with `image/png`.
- Live static Space render has the scoped C3 disclosure, no horizontal overflow,
  and a working full-resolution PNG link.
- Remote executive-summary and PNG hashes match the local files exactly.

## Publication gate outcome

1. Fresh adversarial review completed. C1 retains its explicit finite-audit
   limitation; C3 is narrowed to the main-text observational four-arm protocol.
2. C2 remains `PARTIAL`; the continuum theorem is neither verified nor
   falsified.
3. All three standalone checks were rerun from `repro_treatment_risk/`.
4. Local and live Space renders pass, including the PNG and long claim titles.
5. The existing canonical Space was updated; no duplicate was created.

Final publication status: **PASS**.

## Integrity hashes

```text
96BF1C331C6EF1CEE97E49242E50D80CBF1D4386080C412416953D366AEBB379  audit_lemma41.py
596C673E8E586C1C871C6DA5E3EACD87F8D6AFF465E0F1886552EAF5396A0482  audit_theorem43.py
6DA03DDEF10115F23BAD05F713D6EB36482D8E8E679C8E1ECA2F3C114D6F253F  reproduce_synthetic.py
D651AA5C4F9085FD6D35A3E9E40C9786AE720402F733804535DF2C4B76A39F5C  evidence/lemma41_finite_audit.json
B12040B29FD9907CB7377D9E9359DDD13E978226D5EE5AA510F1501A3C30D7B2  evidence/theorem43_audit.json
9415B1DE4AA663C10B53A338979CB9D3F9509B6D219D0B3F534506740E6ABE12  evidence/synthetic_reproduction.json
78DD0EC2C03C13E717D855CE440FF6BCAB8A8219864282DF335B1691A309BDA0  evidence/synthetic_raw.npz
62986A815A0505B343718C2390E806648AC35EDB7983007F1CE0677FC3CD7B16  evidence/synthetic_coverage.png
```
