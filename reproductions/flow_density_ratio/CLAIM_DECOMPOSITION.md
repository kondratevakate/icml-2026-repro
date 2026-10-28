# scRatio claim decomposition

- **C1 ratio dynamics and implementation:** Proposition 4.1 gives a single
  ODE for the conditional log-density ratio, and the release should implement
  its divergence, vector-field, and score terms in a callable inference path.
- **C2 Gaussian accuracy and efficiency:** on the closed-form shifted-Gaussian
  benchmarks, scRatio should beat the naive and TSM/CTSM baselines across the
  reported dimensions and estimate ratios faster than two independent
  likelihood solves.
- **C3 mutual-information estimation:** scRatio should attain best or
  second-best MAE across the five reported dimensions, including the
  high-dimensional stochastic-path result.
- **C4 differential-abundance estimation:** on the semi-synthetic PBMC68k
  experiment, scRatio should lead the reported AUC/NAR/correlation metrics and
  remain competitive on correct-sign proportion.
- **C5 batch-correction evaluation:** absolute batch-conditioned log-ratios
  should decrease after scVI correction on both NeurIPS 2021 and C. elegans.
- **C6 treatment-response applications:** scRatio scores should correlate with
  classifier separation on ComboSciPlex and recover the reported
  donor/cytokine response pattern in the approximately 10-million-cell PBMC
  cohort.

