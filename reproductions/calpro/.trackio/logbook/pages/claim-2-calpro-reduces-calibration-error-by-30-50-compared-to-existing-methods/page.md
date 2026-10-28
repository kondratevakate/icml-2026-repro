# Claim 2: CalPro reduces calibration error by 30-50% compared to existing methods


---
<!-- trackio-cell
{"type": "markdown", "id": "cell_f8cad3bb4457", "created_at": "2026-07-18T16:28:13+00:00", "title": "Claim 2: CalPro reduces calibration error by 30-50% compared to existing methods"}
-->
**Setup.** Regression calibration via the probability integral transform (PIT): under a calibrated Gaussian predictive, `pit = Phi((y-mu)/sigma)` is Uniform[0,1]; **ECE** = mean over levels q of `|P(pit<=q) - q|`. We compare the evidential adaptive `sigma(x)` against a homoscedastic baseline (same `mu`, constant `sigma` = calibration RMSE), in-distribution and under shift. Code: `claim2_ece.py`. CPU.

| set | ECE baseline | ECE evidential | reduction |
| --- | --- | --- | --- |
| in-distribution | 0.0611 | 0.0113 | **81.5%** |
| shifted | 0.1427 | 0.1050 | 26.4% |

**Verdict — Claim 2 reproduced in direction.** The evidential head substantially lowers calibration error (paper claims 30–50%): **81.5% in-distribution**, well past their range; **26.4% under strong covariate shift**, a bit below — consistent with the over-inflation caveat from Claim 1 (the NIG uncertainty grows conservatively out-of-distribution). Lower calibration error, driven by uncertainty that adapts to the input.
