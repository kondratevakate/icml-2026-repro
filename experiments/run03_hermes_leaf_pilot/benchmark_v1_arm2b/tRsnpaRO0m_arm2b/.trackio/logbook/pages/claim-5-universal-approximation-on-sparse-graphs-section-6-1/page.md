## Claim 5 — universal approximation on sparse graphs (Section 6.1)

Script `verify_claim5.py` → `results/claim5.json`. Verdict: **verified**. Mutation test: yes.

Random-feature MPNN (fixed random message/update weights, ridge readout), 220 train / 120 test
bofop graphs, widths 8→256. Uniform approximation ⇒ **sup-norm** test error, not RMSE.

| width | 8 | 16 | 32 | 64 | 128 | 256 |
|---|---|---|---|---|---|---|
| sup error, continuous DIDM target | 0.0516 | 0.0113 | 0.0083 | 0.0063 | 0.0067 | **0.0068** |
| sup error, **discontinuous** target (mutation) | 0.519 | 0.483 | 0.521 | 0.534 | 0.593 | **0.617** |

Final sup error is **0.47 % of the target range** (0.877) and falls by ~8× from the smallest width.

**Mutation.** The target `1[mean degree > 3]` is discontinuous in the DIDM metric and its sup error
**plateaus at ≈ 0.5–0.6 ≈ the jump size** at every width — uniform approximation fails exactly where
the theorem's continuity hypothesis fails.

**Iteration recorded honestly.** A first version sampled degrees from `{2,3,4}`, so the "discontinuous"
target was on a *well-separated* sample and was learned to 0.009 sup error — the mutation measured
nothing and the claim was marked `inconclusive` at that point. Fixed by sampling a **continuum** of
bofop DIDMs (4-regular graph with a random fraction of edges deleted) so samples straddle the
threshold arbitrarily closely. The sampler was wrong, not the criterion.

---
<!-- trackio-cell
{"type": "markdown", "id": "cell_<built-in function hash>", "created_at": "2026-01-01T00:00:00+00:00", "title": "Claim 5 \u2014 universal approximation on sparse graphs (Section 6.1)"}\n-->
