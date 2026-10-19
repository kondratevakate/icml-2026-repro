# Claim 1: Architecture EPDE MELP dLIF ERG


---
<!-- trackio-cell
{"type": "markdown", "id": "cell_311eb6164302", "created_at": "2026-07-22T12:42:40+00:00", "title": "Claim 1: Architecture EPDE MELP dLIF ERG"}
-->
**Paper claim (Architecture section).** LERD combines an Event Posterior
Differential Equation (EPDE), a Mean-Evolving Lognormal Process (MELP), a
differentiable leaky-integrate-and-fire (dLIF) prior, and an Event-Relational
Graph (ERG) to infer latent neural events and their cross-channel relations
directly from multichannel EEG.

**Verdict: descriptive - transcribed and partially implemented (not a
confirm/refute claim).**

This is a structural description, not an empirical result. The spec (Appendix F,
Algorithms 1-4) was transcribed and found internally consistent: encoder
(EEGNet-style temporal-spatial factorization) -> EPDE/MELP (K=3 lognormal mixture,
reparameterized sampling, Algorithm 1) -> ODE Euler evolution (Algorithm 2) ->
dLIF rate prior with refractory gating -> ERG adjacency from cross-channel event
lags (Algorithm 3) -> GCN -> fused classifier (Algorithm 4).

The encoder and classifier (the "No prior" path) are implemented in `lerd_eeg/`;
the dLIF/ERG prior modules are specified but not yet coded, pending the GPU run
that would exercise them (see Claim 4). No aspect of the architecture description
was found to be inconsistent with the reported shapes (Table F.7).
