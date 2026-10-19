# scRatio independent checks

Final prepared score: **7/12**, matching the frozen forecast.

| Claim | Score | Independent result |
| --- | ---: | --- |
| C1 ratio ODE | 2/2 | The released public estimator recovered an analytic two-dimensional Gaussian ratio with maximum absolute error `1.07e-7`. Source tracing matches the divergence and two score-correction terms in Proposition 4.1. |
| C2 Gaussian accuracy/runtime | 1/2 | A fresh seeded 1,200-step CPU run produced finite estimates and direct MSE `1.6455` versus naive MSE `5.1616`. Its one-shot timing was noisy (`0.94x`; earlier identical-seed execution was slightly above `1x`), so it is not used as runtime support. In the tracked CSV, direct estimation is faster and lower-MSE in all 30 displayed schedule/location/dimension comparisons. However, the scRatio rows contain three values while Figure 2 says five training runs; paper-scale retraining and raw predictions are absent. |
| C3 mutual information | 1/2 | The executed notebook preserves the qualitative high-dimensional result, but none of its five stochastic-0.25 MAEs match current Table 1 after rounding. Notebook values are `[0.155849, 0.288463, 0.492947, 1.027249, 0.455203]`; Table 1 reports `[0.03, 0.09, 0.07, 0.11, 1.16]`. Referenced ratio arrays are external. |
| C4 differential abundance | 1/2 | Independently parsed executed outputs reproduce Table 2 to its displayed precision and put scRatio first on five of six metrics; MELD leads CSP at high DA. The underlying per-cell predictions remain external. |
| C5 batch correction | 1/2 | Executed notebooks consistently reconstruct `(90261, 12)` and `(89701, 12)` corrected/uncorrected LLR matrices for 431 and 248 condition groups. The clone has no corresponding NPZ arrays, so the magnitude decrease cannot be numerically recomputed from the clone alone. |
| C6 treatment response | 1/2 | From 23 printed ComboSciPlex pairs, independent recomputation gives Pearson `0.94493` and Spearman `0.90711` between log mean absolute ratio and classifier separation. The PBMC notebook contains the expected IFN-omega/APRIL/IL-10 analysis and executed figures, but not the model, raw ratios, or numerical group summary. |

Additional checks:

- 19 application notebooks contain executed outputs and no stored error output.
- The official package has no repository test directory; the audit bundle adds
  four focused tests, all passing.
- PDF pages 1, 6, 7, 8, and 9 were rendered and visually inspected.
- The fresh smoke run used local PyTorch `2.10.0+cpu`, not the paper's pinned
  PyTorch 2.5.1 environment, and is intentionally treated as partial evidence.
- The 3,302,638,232-byte Zenodo archive matches its declared MD5. Its 126 ZIP
  members expand to 7,031,582,791 bytes under `data_scRatio/` (plus macOS
  metadata); the archive was inventoried without bulk extraction.
