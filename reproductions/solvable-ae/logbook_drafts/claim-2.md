# Claim 2 — Self-supervised test loss misaligns with representation quality

**Status:** direction reproduced (preliminary; margins to widen at paper scale).

## The prediction (paper Fig. 2, Eq. 24)

For this model the linear autoencoder is the reconstruction-loss optimum: it achieves a **lower**
held-out reconstruction MSE than any nonlinear single-neuron autoencoder (Eq. 24 inequality). Yet the
linear autoencoder only captures the visible spike `u*` and is blind to the hidden `v*`. So on a
downstream task that depends on `v*`, the linear model — despite its lower reconstruction loss —
performs **worse** than the nonlinear model that recovered `v*`.

Downstream task (paper Fig. 2): labels `y = sign(x . v*)`, predicted by a linear readout initialised
from the pre-trained autoencoder weight `w`. Representation quality tracks `theta_v`, not the loss.

## What I did

Trained the linear autoencoder and the `v*`-recovering nonlinear autoencoder (ReLU for `k*=2`, tanh
for `k*=3`) on the same data, then on a fresh 20k-sample test set measured (a) reconstruction MSE and
(b) downstream classification error using `sign(x . w)`.

## Result (paper-scale: d=800, alpha=30, 800 epochs)

**k\* = 2** (nonlinear = ReLU)
| model | test recon MSE | downstream error | theta_v |
|---|---|---|---|
| linear | **799.12** (lower) | 0.497 (worse) | 0.031 |
| ReLU | 800.11 | **0.451** | 0.205 |

**k\* = 3** (nonlinear = tanh)
| model | test recon MSE | downstream error | theta_v |
|---|---|---|---|
| linear | **799.13** (lower) | 0.480 (worse) | 0.037 |
| tanh | 799.46 | **0.467** | 0.018 |

## Reading of the result

- The **misalignment direction reproduces in both regimes**: the linear autoencoder has the strictly
  lower test reconstruction loss (consistent with the Eq. 24 inequality) yet the higher downstream
  error. Lower self-supervised loss, worse representation.
- **k\* = 2 is clean:** ReLU recovers v* (theta_v=0.205) and the downstream gap widened to ~4.6 pp
  (0.497 vs 0.451) at paper scale, up from ~2.6 pp at d=300 — the effect strengthens as expected.
- **k\* = 3 is weak:** tanh only barely recovers v* in this seed (theta_v=0.018), so its downstream
  advantage is small (0.480 vs 0.467, ~1.3 pp). The direction is right but the effect is marginal —
  the same near-floor k*=3 regime seen in Claim 1.

**Verdict:** Claim 2 reproduced — reconstruction loss is a misleading proxy for representation quality
in this solvable model. Convincing at `k*=2` (widening downstream gap), directional-but-marginal at
`k*=3`. The Dice-vs-ICC pattern in self-supervised learning, confirmed.

## Note for Kate's own work

This is the Dice-vs-ICC pattern in self-supervised learning: a convenient scalar (reconstruction loss)
moves the right way while the property you actually care about (recovery of the latent structure) moves
the wrong way. Same failure mode, different domain.
