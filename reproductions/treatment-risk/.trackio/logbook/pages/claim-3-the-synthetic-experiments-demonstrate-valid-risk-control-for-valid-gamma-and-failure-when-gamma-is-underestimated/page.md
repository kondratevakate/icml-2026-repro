# Claim 3: The synthetic experiments demonstrate valid risk control for valid Gamma and failure when Gamma is underestimated.


---
<!-- trackio-cell
{"type": "markdown", "id": "cell_316749629a60", "created_at": "2026-07-28T16:28:29+00:00", "title": "Claim 3: The synthetic experiments demonstrate valid risk control for valid Gamma and failure when Gamma is underestimated."}
-->
## Protocol

The complete observational synthetic pipeline was independently implemented:

- Eqs. (21)-(24) for the unconfounded and confounded data generators;
- `m=n=1000`, 200 nominal tolerances in `(0,0.5]`;
- all depth-one thresholds over both covariates, 200 bins, both orientations;
- Eq. (12) policy learning, Bentkus Eq. (15), prefix selection Eq. (14);
- eight target tolerances from `0.05` to `0.40`;
- 1000 independent datasets for each of four arms;
- exact closed-form evaluation of true interventional risks.

```powershell
python reproduce_synthetic.py --runs 1000 `
  --output evidence/synthetic_reproduction.json `
  --raw-output evidence/synthetic_raw.npz `
  --figure evidence/synthetic_coverage.png
```

Seed: `20260728`. Runtime: `237.3 s` on CPU. Protocol SHA-256:
`0f63020d04730202184324e417d50887728c5fc9eaf65418be80e81a1acdd97d`.

## Results

| Arm | Gamma valid? | Controlled tolerances | Minimum coverage | Mean treated fraction at `tau=0.40` |
| --- | --- | ---: | ---: | ---: |
| No confounding, `Gamma=1` | yes | 8/8 | 0.988 | 0.916 |
| No confounding, `Gamma=2` | yes, conservative | 8/8 | 1.000 | 0.759 |
| Confounded, `Gamma=2` | yes | 8/8 | 0.998 | 0.894 |
| Confounded, `Gamma=1` | no | 5/8 | 0.257 | 0.935 |

The invalid-`Gamma` arm has coverage `0.754`, `0.451`, and `0.257` at
`tau=0.15`, `0.20`, and `0.25`, respectively. Its 90th-percentile true
treatment risks are `0.1705`, `0.2448`, and `0.2750`, all above the requested
tolerances.

The check is not vacuous: at `tau=0.40`, the valid arms treat 75-91% of the
population. Closed-form treat-all treatment risks equal the paper's saturation
values, `0.25` without confounding and `0.275` with confounding.

## Declared reconstruction choices

The paper does not publish source code or fully specify FFT tie-breaking.
The reproduction exhaustively evaluates every depth-one binned threshold rather
than relying on an opaque greedy implementation. Eq. (12) uses the empirical
treated fraction on `D_m`; the certificate uses the exact population treated
mass available under the uniform synthetic DGP, keeping `p_pi` and `V_max`
independent of `D_n`. `V_max` is the tight analytic almost-sure bound over the
policy's treated region; a looser bound would only be more conservative. The
randomized-trial appendix experiment was not rerun, so this verdict is restricted
to the four-arm observational protocol in the main text.

## Verdict: VERIFIED

For the main-text observational protocol, all three valid sensitivity
assumptions meet the claimed 90% coverage at every tolerance, while the
deliberately invalid assumption fails in the predicted middle range using the
identical pipeline.
