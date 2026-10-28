# Claim 1: Conflict-aware aggregation dynamically resolves inter-view conflicts


---
<!-- trackio-cell
{"type": "markdown", "id": "cell_bd65e2d1deaf", "created_at": "2026-07-18T17:26:59+00:00", "title": "Claim 1: Conflict-aware aggregation dynamically resolves inter-view conflicts"}
-->
**Setup.** No code was released (upstream repo is README-only), so this is an independent reimplementation of the paper's aggregation equations, transcribed from the PDF (Eq. 4-9): belief/uncertainty masses `b_k, u` with `sum_k b_k + u = 1` (Eq. 5); conflict degree C (Eq. 6); combined belief (Eq. 9) and combined uncertainty (Eq. 8). We construct two opinions, sweep their conflict C from 0 to 1, and check Propositions 1-2 two ways: `u_eq8` (Eq. 8 exactly as printed) and `u_norm = 1 - sum_k b_combined` (forced consistent with Eq. 5). Code `verify_prop.py` (numpy, CPU, seconds).

**Result.**
- **Proposition 1** (C -> 0, integrating a *consistent* opinion lowers uncertainty): **holds** for both `u_eq8` and `u_norm`.
- **Proposition 2** (C -> 1, integrating a *conflicting* opinion with higher uncertainty raises it): **holds for `u_norm`**, but **fails for `u_eq8`** — Eq. 8's C->1 limit is `u_a u_b/(u_a+u_b)`, which is *less* than `u_o` (a decrease), contradicting Prop. 2. The paper's own proof of Prop. 2 uses a limit with a factor 2, matching `u_norm = 1 - sum b` (Appendix A.5 normalization), not the printed Eq. 8.

**Verdict — reproduced, with a finding.** The design principle behind Claim 1 (consistent opinions sharpen, conflicting opinions blunt confidence) is confirmed under the paper's own normalization constraint. But **Equation 8 as printed appears to be a typo** (missing a normalization factor ~2): it satisfies Prop. 1 but not Prop. 2. This is a partial refutation at the level of the printed equation with the substance of the claim confirmed.
