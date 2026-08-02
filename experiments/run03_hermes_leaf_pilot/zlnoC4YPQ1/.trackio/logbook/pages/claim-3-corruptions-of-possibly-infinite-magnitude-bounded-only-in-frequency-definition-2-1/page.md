# Claim 3 — corruptions of possibly infinite magnitude, bounded only in frequency (Definition 2.1)

---
<!-- trackio-cell
{"type": "markdown", "id": "cell_66cc53b674c5", "created_at": "2026-08-02T04:30:00+00:00", "title": "Claim 3 \u2014 corruptions of possibly infinite magnitude, bounded only in frequency (Definition 2.1)"}
-->
**Source:** Definition 2.1 (Sec 2.2); mechanism Definition 3.1 (Sec 3) + Lemma D.5 (Sec 4.1);
brittleness of the GP alternative Appendix H.
**Verdict: verified** (for the posterior-level mechanism; see Evidence boundary).

Exhaustive enumeration: 10 seeds × all 8 possible corrupted indices × 13 magnitudes
|c| ∈ {10⁰,…,10¹²} = **1040 configurations**. For each, deviation of the posterior mean from the
uncorrupted GP posterior mean, sup over a 101-point grid:

* RCGP (P-IMQ, g = 0, L = 1.96, c = 1): **sup deviation = 1.4212** over all 1040 configurations,
  and it *converges* as |c| → ∞ — maximum absolute drift between |c| = 10⁶ and |c| = 10¹² is
  **9.93e-07**. Normalised by σ_uc the sup is **1.5437** (finite, i.e. of the Lemma D.5 form
  C_w √T_c σ_uc);
* standard GP on the same data: **sup deviation = 4.983e+11**, and increasing |c| by 10⁶ increases
  the deviation by a factor of **≥ 999998** (exactly linear in c, as Appendix H predicts).

**Mutation test:** replacing the P-IMQ weight by the constant weight W (= removing the robustness
mechanism; formally J_w = I, m_w = 0) makes the deviation **4.536e+11** at |c| = 10¹² — unbounded,
exactly as predicted when the mechanism is disabled.
