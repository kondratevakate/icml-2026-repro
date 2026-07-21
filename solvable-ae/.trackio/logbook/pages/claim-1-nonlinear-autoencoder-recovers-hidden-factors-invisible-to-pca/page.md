# Claim 1: Nonlinear autoencoder recovers hidden factors invisible to PCA


---
<!-- trackio-cell
{"type": "markdown", "id": "cell_d742a8332800", "created_at": "2026-07-18T17:16:56+00:00", "title": "Claim 1: Nonlinear autoencoder recovers hidden factors invisible to PCA"}
-->
**Setup.** Spiked-cumulant model (Eq. 1-2): visible spike u* (in the covariance) + hidden v* (uncorrelated, appearing only at correlation exponent k*). Single-neuron tied autoencoder `xhat=(w/sqrt d) sigma(w^T x/sqrt d)` (Eq. 4), full-batch Adam. Recovery = cosine theta_v of w with v*; floor = d^-1/2. Paper-scale d=800, alpha in {15,30}, 800 epochs. Code `verify_solvable_ae.py` (numpy, analytic gradient). CPU.

**k* = 2** (floor 0.035):

| activation | theta_v (a=15) | theta_v (a=30) | verdict |
| --- | --- | --- | --- |
| linear / PCA | 0.042 | 0.031 | no v* (floor) |
| **ReLU** | **0.134** | **0.184** | **RECOVERS v*** |
| tanh | 0.041 | 0.028 | no v* |
| quadratic | 0.030 (theta_u=0.12) | 0.035 (theta_u=0.04) | no v* (misses u* too) |

**k* = 3** (floor 0.035):

| activation | theta_v (a=15) | theta_v (a=30) | verdict |
| --- | --- | --- | --- |
| linear / PCA | 0.015 | 0.006 | no v* |
| ReLU | 0.029 | 0.011 | no v* (floor) |
| **tanh** | 0.041 | **0.058** | above floor, marginal |
| quadratic | 0.026 | 0.036 | no v* |

**Verdict — reproduced, honestly two-sided.** The ReLU-recovers-at-k*=2 half of the Table-1 flip is convincing (ReLU 0.134->0.184; others at floor). At k*=3 the flip is only directional: tanh is the sole above-floor activation (0.058, ~1.65x floor) while ReLU/linear collapse, but it does not clear a 2x-floor bar — matching the paper's own Fig. 7 (k*=3 recovery sits just above the d^-1/2 line). Corroborations not targeted: PCA == linear AE to the decimal (Prop. 2.1); quadratic misses even u* (Table 1).
