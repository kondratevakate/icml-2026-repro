## Claim 1 — Fourier embedding (`verify_claim1.py` → `results/claim1.json`)

**Anchor:** v2 §3.1, Eq. (4): ⟨c|π⟩ = ∫ Fc(−z) Fπ(z) dϱ(z), F an isometry on L²(ℝᵈ;ϱ).

**Setup.** Discrete grid (unitary DFT is the exact discrete Fourier isometry). Checked
(a) Parseval/inner-product preservation, (b) the pairing identity for a random density
w.r.t. ϱ, (c) the pairing identity for a *genuine* coupling π ∈ Π(µ,ν) obtained by Sinkhorn
scaling (marginal error 1.4e−17).

**Numbers.** isometry relative error `1.5e−16`; norm preservation `1.6e−16`;
time-domain vs frequency-domain pairing: random density `0.2509738933825768` vs
`0.2509738933825770` (rel. gap `6.6e−16`); OT coupling `−0.05839465248784993` vs
`−0.05839465248785002` (rel. gap `1.4e−15`).

**Mutation.** Replacing F by a non-unitary, high-frequency-truncated transform breaks the
identity: relative gap `1.03e+3` (and un-normalised DFT inflates the norm by 7×).

**Verdict: verified.** The embedding claim is an exact algebraic identity and reproduces to
machine precision; the mutation confirms the test is sensitive to the isometry property.

---
<!-- trackio-cell
{"type": "markdown", "id": "cell_<built-in function hash>", "created_at": "2026-01-01T00:00:00+00:00", "title": "Claim 1 \u2014 Fourier embedding (`verify_claim1.py` \u2192 `results/claim1.json`)"}\n-->
