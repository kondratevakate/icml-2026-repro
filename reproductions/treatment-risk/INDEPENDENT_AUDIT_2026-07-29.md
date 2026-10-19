# Independent Audit - 2026-07-29

This publication-pass audit was run from the standalone
`repro_treatment_risk/` bundle after reconfirming the frozen forecast. Outputs
are in `fresh_evidence/`.

## Results

| Claim | Fresh result | Adversarial conclusion |
| --- | --- | --- |
| C1, Lemma 4.1 | `268,800/268,800` exact checks pass; both underestimated-Gamma mutations violate the bound | `VERIFIED` is defensible for the finite audit, with the page explicitly stating that it is not a symbolic proof over arbitrary measurable spaces |
| C2, Theorem 4.3 | Eq. (18) counterexample reproduced; finite-grid repair has zero failures in `1,396,054` UCB patterns; Bentkus passes `165/165` | Keep `PARTIAL`; neither full verification nor falsification of the continuum theorem is supported |
| C3, synthetic experiment | Valid arms control `8/8`, `8/8`, and `8/8`; invalid-Gamma control reaches only `5/8`, minimum coverage `0.257` | `VERIFIED` only for the main-text observational four-arm protocol; randomized-trial appendix is outside the verdict |
| C4, STAR | Not run | `NOT ATTEMPTED` |
| C5, IST | Not run | `NOT ATTEMPTED` |

The fresh JSON outputs match the canonical JSON semantically after removing
runtime/environment metadata. The fresh and canonical NPZ arrays are
element-for-element identical (`max_abs_difference = 0`), and the rendered PNG
hash is unchanged.

## Final forecast

The pre-publication forecast remains **4-5/10**:

- 2 points from C1;
- 0-1 possible partial credit from C2;
- 2 points from the restricted, explicitly scoped C3 verdict;
- 0 points from C4 and C5.
