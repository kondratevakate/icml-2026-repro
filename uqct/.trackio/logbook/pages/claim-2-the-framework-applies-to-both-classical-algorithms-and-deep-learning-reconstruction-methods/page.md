# Claim 2: The framework applies to both classical algorithms and deep learning reconstruction methods


---
<!-- trackio-cell
{"type": "markdown", "id": "cell_3c2cc0345705", "created_at": "2026-07-18T17:24:07+00:00", "title": "Claim 2: The framework applies to both classical algorithms and deep learning reconstruction methods"}
-->
**Reproduced for the classical case.** The SLM confidence-sequence construction is reconstruction-agnostic: beta_t is defined from any predictable reconstruction sequence xhat_s. We instantiate it with a classical single-point (log-domain SART) reconstruction and confirm the coverage guarantee holds (see Claim 1: empirical coverage 1.000 at delta in {0.1, 0.05}). This demonstrates the framework applies to classical algorithms on a faithful Beer-Lambert / Poisson / Radon setup, CPU, no checkpoints.

**Deep-learning case — NOT reproduced (blocked).** The paper also instantiates the same construction with U-Nets, U-Net ensembles, and diffusion models (as multi-point mixing distributions mu_s). Reproducing that needs their released checkpoints or GPU training; flagged as a blocker rather than presented as reproduced. The construction itself is identical — only the mixing distribution changes — so the classical reproduction validates the mechanism the deep case reuses.
