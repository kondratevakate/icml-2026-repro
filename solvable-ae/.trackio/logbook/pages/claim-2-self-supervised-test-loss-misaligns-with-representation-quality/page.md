# Claim 2: Self-supervised test loss misaligns with representation quality


---
<!-- trackio-cell
{"type": "markdown", "id": "cell_b1a86282a407", "created_at": "2026-07-18T17:16:56+00:00", "title": "Claim 2: Self-supervised test loss misaligns with representation quality"}
-->
**Setup.** Linear autoencoder (= reconstruction-loss optimum) vs the v*-recovering nonlinear autoencoder (ReLU at k*=2, tanh at k*=3), on held-out data: test reconstruction MSE vs a downstream task `y = sign(x . v*)` predicted by `sign(x . w)`. d=800, alpha=30. Code `verify_solvable_ae.py`. CPU.

**k* = 2** (nonlinear = ReLU):

| model | test recon MSE | downstream error | theta_v |
| --- | --- | --- | --- |
| linear | **799.12** (lower) | 0.497 (worse) | 0.031 |
| ReLU | 800.11 | **0.451** | 0.205 |

**k* = 3** (nonlinear = tanh):

| model | test recon MSE | downstream error | theta_v |
| --- | --- | --- | --- |
| linear | **799.13** (lower) | 0.480 (worse) | 0.037 |
| tanh | 799.46 | **0.467** | 0.018 |

**Verdict — reproduced.** In both regimes the linear autoencoder has the strictly lower test reconstruction loss yet the higher downstream error: the self-supervised metric moves opposite to representation quality. Convincing at k*=2 (downstream gap ~4.6 pp, widening from d=300); directional-but-marginal at k*=3 (tanh only weakly recovers v* this seed). The Dice-vs-ICC pattern in self-supervised learning, confirmed on a solvable model.
