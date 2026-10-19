# Claim 1: Deep reconstruction methods yield tighter confidence regions than classical while maintaining coverage guarantees


---
<!-- trackio-cell
{"type": "markdown", "id": "cell_d4ce305d2019", "created_at": "2026-07-18T17:24:07+00:00", "title": "Claim 1: Deep reconstruction methods yield tighter confidence regions than classical while maintaining coverage guarantees"}
-->
This claim has two parts: (a) **coverage is maintained**, and (b) **deep methods give substantially tighter regions than classical**. We reproduce **(a) on the classical path**; (b) needs their deep checkpoints.

**Setup (from the paper).** Forward model = parallel-beam Radon `R_alpha`, Beer-Lambert mean counts `lambda = I0 exp(-R_alpha x)`, measurement `y ~ Poisson(lambda)`. SLM confidence sequence (Prop. 3.2, single-point mixing): `nll(x,s) = -sum(y logL - L)`, `L_t(x*) = sum nll(x*)`, `beta_t = sum nll(xhat_{s-1})` with a predictable log-domain SART reconstruction; `C_t = {x : L_t(x) <= beta_t + log(1/delta)}`. The Poisson log(y!) cancels between L_t and beta_t (no scipy needed). Phantom r=32, 40 views, I0=1e4, 300 Poisson realisations per delta. Code `verify_uqct_coverage.py` (numpy, CPU).

| delta | nominal 1-delta | empirical coverage |
| --- | --- | --- |
| 0.10 | 0.900 | **1.000** (300/300) |
| 0.05 | 0.950 | **1.000** (300/300) |

**Part (a) reproduced:** the SLM confidence sequence covers x* in every run, satisfying `P(for all t: x* in C_t) >= 1-delta`. Coverage is a conservative 1.000 — exactly the paper's motivation: a classical single-point reconstruction yields loose beta_t, hence wide-but-valid regions.

**Part (b) — NOT reproduced (blocked, not toy-substituted).** Showing deep reconstructions give *substantially tighter* regions requires the paper's trained U-Net / ensemble / diffusion models (data + checkpoints available on request, not yet released) or GPU training. Flagged as a blocker.
