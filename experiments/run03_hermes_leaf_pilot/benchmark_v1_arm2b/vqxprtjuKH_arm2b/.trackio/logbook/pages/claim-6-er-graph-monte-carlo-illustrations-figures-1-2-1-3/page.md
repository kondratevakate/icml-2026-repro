## Claim 6 — ER-graph Monte-Carlo illustrations (Figures 1–2, §1.3)

**Verdict: inconclusive.**

n=8 Erdős–Rényi graphs (p = 1/8 … 8/8, 3 graphs per p, 16 000 MC samples) were simulated for independent, positively-, and negatively-correlated Gaussians. The qualitative picture of the paper's figures is reproduced (per-set OPT is increasing in p; the optimal allocation concentrates mass onto few variables), but the **strict** concavity of per-set OPT in p and monotonic increase of concentration are not robustly confirmed.

Quoted evidence (`results/claim6.json`):
- `"all_concave": false`, `"all_concentration_increasing": false`
- Independent: `"increasing": true` (per_set_opt 0.219 → 0.524) but `"concave": false`; `"second_differences": [0.0389, -0.0655, -0.0029, 0.0009, -0.0042, -0.0087]` — includes positive (convex) segments.
- Independent `"concentration_increasing": false` (top-2 mass `[0.886, 0.744, 0.710, 0.684, 0.676, 0.606, 0.624, 0.495]` falls, not rises); negative setting does show `"concentration_increasing": true`.
- `"top2_mass"` profiles confirm heavy concentration onto 1–2 variables (e.g. positive p=0.125: `[0.754, 0.158, 0.050, …]`, top-2 = 0.912).

**Mutation test** (replace optimal allocation by uniform): `"property_breaks": true` — uniform allocation gives top-2 mass exactly 0.25 for every p and destroys the concentration pattern. The concentration effect is genuine, but the paper's figures are explicitly illustrative; at n=8 the exact concavity claim is not reproduced, so the verdict is **inconclusive**.

---

---
<!-- trackio-cell
{"type": "markdown", "id": "cell_<built-in function hash>", "created_at": "2026-01-01T00:00:00+00:00", "title": "Claim 6 \u2014 ER-graph Monte-Carlo illustrations (Figures 1\u20132, \u00a71.3)"}\n-->
