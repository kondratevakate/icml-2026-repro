# Reproduction: A Solvable Model Where Nonlinear Autoencoders Learn Structure Invisible to PCA

**Paper:** Conde Mendes, Bardone, Koller, Medina Moreira, Erba, Troiani, Zdeborova.
*A Solvable High-Dimensional Model Where Nonlinear Autoencoders Learn Structure Invisible to PCA While Test Loss Misaligns With Generalization.* ICML 2026.
OpenReview `wm3ABfhE7P` · arXiv 2602.10680.

**Reproducer:** kondratevakate · independent from-the-paper reimplementation (no code was released).
**Compute:** CPU only, numpy, no GPU. One evening.

## What the paper claims

A tractable high-dimensional **spiked-cumulant model** plants two latent factors in the data:
a **visible** direction `u*` that shows up in the covariance, and a **hidden** direction `v*`
that is statistically dependent on the visible latent but *uncorrelated* with it, so it appears
only in higher-order moments. The "hardness" is set by the **correlation exponent** `k*` — the
smallest `k` with `E[lambda^k nu] != 0`.

Two claims are reproduced here:

- **[Claim 1](claim-1.md) — Nonlinear beats PCA, and the recovery is activation-selective.**
  A minimal single-neuron nonlinear autoencoder provably recovers **both** spikes, while PCA and
  linear autoencoders recover only `u*`. Which nonlinearity succeeds depends on matching its
  Hermite spectrum to `k*` — giving a sharp **ReLU <-> tanh flip** between `k*=2` and `k*=3`
  (paper Table 1).

- **[Claim 2](claim-2.md) — Self-supervised test loss misaligns with representation quality.**
  The linear autoencoder achieves a **lower** test reconstruction loss yet a **worse** downstream
  task performance than the nonlinear autoencoder that recovers `v*` (paper Fig. 2). Reconstruction
  loss is the wrong yardstick for representation quality.

## Why this reproduction

The second claim is a clean instance of a metric systematically disagreeing with the quantity of
real interest — reconstruction loss improves while the useful latent structure is lost. Verifiable
strictly, on synthetic data, with no training of large models: either the planted hidden direction
is recovered (nonzero cosine similarity above the `d^{-1/2}` floor) or it is not.

## Method in one screen

Data (Eq. 1-2): `x = lambda u*/sqrt(d) + S (nu v*/sqrt(d) + z)`, with `u*,v*,z ~ N(0,I_d)` and a
rank-1 whitening `S` that removes `v*` from the covariance. Exact planted latent laws transcribed
from the paper:

- `k*=2` (Fig. 1 / Sec. E): `lambda ~ N(0,1)`, `nu = -sqrt(2) if |lambda| < Phi^{-1}(0.75) else +sqrt(2)`.
- `k*=3` (Sec. E.4): `lambda ~ N(0,1)`, `nu = sign(lambda) sign(|lambda| - sqrt(2 ln 2))`.
  The threshold `sqrt(2 ln 2)` is exactly what forces `E[lambda nu]=E[lambda^2 nu]=0` while `E[lambda^3 nu] != 0`.

Model (Eq. 4): single-neuron tied autoencoder `xhat = (w/sqrt(d)) sigma(w^T x / sqrt(d))`,
squared reconstruction loss (Eq. 6), full-batch Adam, lr=0.1, init `w ~ N(0,I_d)`.

Recovery metric: cosine similarity `theta_s = |w . s| / (||w|| ||s||)`; the `d^{-1/2}` line is the
"no recovery" floor (paper Fig. 7).

Code: [`verify_solvable_ae.py`](../verify_solvable_ae.py).
