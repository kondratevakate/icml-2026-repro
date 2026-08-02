# Claim 6 — D-head: DCL, not architecture, enforces the dual bound

---
<!-- trackio-cell
{"type": "markdown", "id": "cell_9d973235baa5", "created_at": "2026-08-02T04:00:00+00:00", "title": "Claim 6 \u2014 D-head: DCL, not architecture, enforces the dual bound"}
-->
**Paper source:** Section 3.4, Eqs. (3)–(4); Table 1 row "D"; Conclusion.
**Script:** `verify_claim6.py` → `results/claim6.json`. Command: `.venv/bin/python verify_claim6.py`.

The claim has three parts, all tested:

1. **Conservation survives averaging.** The Eq. (3) averaged update has max relative drift
   **4.377e-16** (float64) over 40 configs — each branch is conservative, so their mean is.
2. **No strict dual-bound guarantee.** With independent (untrained) branch logits, **6/6** seeds
   produce violations: mean lb rate **3.27 %**, ub rate **2.65 %**, max magnitude **0.479** on a
   [0,1] field. This matches the paper's own statement that the averaged update "does not
   theoretically guarantee satisfaction of both bounds unless the two branches agree exactly" —
   the anchored claim is therefore confirmed rather than contradicted.
3. **DCL is the enforcing mechanism.** Minimizing L_DCL (Eq. 4) directly on the head logits
   (Adam, lr 0.05, 200 steps, 32×32 grid, r = 2, float64) drives L_DCL from 0.195 to
   **1.04e-8** and the violation rate to **exactly 0.00 %** on **6/6** seeds — "near-zero
   violations without post-hoc clipping", as claimed.

**Mutation test:** flip the sign of the objective (maximize branch disagreement) for the same
200 steps. L_DCL rises to 15.5, **0/6** seeds reach zero violations, and the max violation
magnitude grows to **2.90** (6× the untrained level). Violations track DCL, not the optimizer.

**Supporting analysis:** the per-cell violation is provably at most ½|Δu^out − Δu^in|; this held
on all 20 tested seeds. Across an ensemble of 90 random states spanning six logit scales, the
Spearman correlation between √L_DCL and the violation magnitude is **ρ = 0.903 (p = 4.8e-34)**,
with 48/90 states showing exactly zero violation at low disagreement.
**Verdict: verified.**

---
