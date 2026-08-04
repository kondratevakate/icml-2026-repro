# Claim 5 — Section 5, Figures 2 and 4 (baselines + "q ≥ 10 gives the sublinear→linear transition")

---
<!-- trackio-cell
{"type": "markdown", "id": "cell_3fbaaa2b7ea6", "created_at": "2026-08-02T04:30:00+00:00", "title": "Claim 5 \u2014 Section 5, Figures 2 and 4 (baselines + \"q \u2265 10 gives the sublinear\u2192linear transition\")"}
-->
**Verdict: `inconclusive`** (split, and *not* supported as stated for the q ≥ 10 half).

*Source:* Section 5, Figure 2 (toy problem, "Condition in (4) suggests q ≥ 10 to ensure linear
convergence") and Figure 4 (comparison with PADMM, IPDS-ADMM, IADMM).
*Script:* `verify_claim5.py` → `results/claim5.json`.

**(i) q ≥ 10 transition.** The toy problem is fully specified in Section 5 but μ_x, μ_z and ρ are
not reported, so we swept q over 10 values × 6 hyperparameter settings × 6 seeds and took, per q,
the **best** (most favourable to the claim) worst-case contraction factor:

| q | 0.5 | 1 | 2 | 5 | 8 | 10 | 12 | 15 | 20 | 50 |
|---|---|---|---|---|---|---|---|---|---|---|
| best worst-case factor | 0.2296 | 0.1667 | 0.0475 | 0.0378 | 0.0588 | 0.0317 | 0.0250 | 0.0245 | 0.0287 | 0.3364 |

**All 5 tested values of q < 10 are already linear** (factors 0.03–0.23; median below q=10 is
0.0588 vs 0.0287 at/above q=10 — the same order of magnitude), and q = 50 is *worse* than q = 10.
We found **no sublinear regime below q = 10 and hence no transition at q = 10** under any of the
six hyperparameter settings. Under a *fixed* ρ the ordering is in fact reversed (larger q slower).
Because the paper does not report μ_x, μ_z, ρ or the initialisation of Figure 2, we cannot exclude
a setting in which the figure's transition appears, so the honest verdict is `inconclusive` rather
than `falsified`: the specific numeric threshold "q ≥ 10" is **not reproducible from the
information published**.

**(ii) Baseline comparison (Figure 4): `inconclusive`, deliberately no numbers.** No code release,
and the paper reports neither the problem instances/dimensions of the three panels, nor the
baselines' hyperparameters (proximal / dual step sizes for PADMM, IPDS-ADMM, IADMM), nor
initialisations or seeds. Any re-implementation would benchmark our own baseline choices, not the
authors', so no comparison figure was produced.

---
