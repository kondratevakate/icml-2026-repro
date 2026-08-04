## Claim 1 — Theorem 5.1 (FPA revenue monotonicity for tCPA)

**What the paper claims:** for tCPA bidders without budget using uniform
bidding with μ=1 in a first-price auction, refining the prediction model never
decreases platform revenue: `Rev(M_A) ≥ Rev(M_B)` whenever `M_A` refines `M_B`.
Revenue has closed form `Rev(M) = Σ_C w_C · max_i t_i·p_i,C`.

**Reproduction.** Built 2000 random calibrated model pairs (coarse partition,
refined by splitting each cluster), computed revenue from first principles,
confirmed `Rev(M_A) ≥ Rev(M_B)` in **every** instance (min margin ≈ 0, never
negative within 1e-9). An illustrative example is recorded in the JSON.

**Mutation test (property must break).** (a) Constructed `M_A` *not*
refining `M_B` → revenue can strictly decrease (observed). (b) Replaced
calibrated predictions with random values (calibration broken) → the Jensen
step is invalid and revenue can decrease (observed). So both theorem conditions
(refinement + calibration preservation) are essential.

**Verdict: verified.** Source: Theorem 5.1, §5.2.1, proof Appendix A.2.

---

---
<!-- trackio-cell
{"type": "markdown", "id": "cell_<built-in function hash>", "created_at": "2026-01-01T00:00:00+00:00", "title": "Claim 1 \u2014 Theorem 5.1 (FPA revenue monotonicity for tCPA)"}\n-->
