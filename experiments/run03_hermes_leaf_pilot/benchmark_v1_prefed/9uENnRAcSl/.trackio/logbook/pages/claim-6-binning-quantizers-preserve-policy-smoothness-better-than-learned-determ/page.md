## Claim 6 — Binning quantizers preserve policy smoothness better than learned; deterministic experts violate RTVC more (§4.1, empirical)

*Source:* arXiv:2603.20538, Section 4.1 (empirical claim; **no released dataset/code**).
*Type:* empirical → reproduced with a controlled **synthetic** stand-in. *Script:* `verify_claim6.py` → `results/claim6.json`. *Seed:* 20260321.

**Verdict: toy.**

The paper is theory-focused and releases no standalone experiment for this claim (supported only by
Prop 3.2). We reproduce the *qualitative phenomenon* with a synthetic setup (state ∈ ℝ², action ∈
[−2,2]; deterministic expert `π(x)=clip(Lx,−2,2)` vs stochastic `N(mean,σ)`; K=16 binning vs k-means
learned quantizer; RTVC test via `κ(r)=1{r>δ0}`):

- quantizer avg error: binning = 0.0624, learned-on-deterministic = 0.0472, learned-on-stochastic = 0.0505
- RTVC violation rates: binning_det = **0.1482**, learned_det = **0.2107** (learned > binning ✓);
  binning_sto = **0.0000**, learned_sto = **0.0000** (det > sto ✓)
- `prediction_bin_better_than_learned = true`, `prediction_deterministic_worse_than_stochastic = true`

**Mutation test (increase expert stochasticity σ 0.3 → 2.0):** learned-det violation drops
**0.2107 → 0.0000**, confirming the test is sensitive to the smoothness/determinism structure the
claim depends on. `passed = true`.

Honest boundary: this is a *toy* (synthetic) reproduction of the empirical phenomenon, not the
paper's unreleased data. The direction/ordering matches the claim, but full verification would
require the paper's actual empirical experiment.

---

---
<!-- trackio-cell
{"type": "markdown", "id": "cell_<built-in function hash>", "created_at": "2026-01-01T00:00:00+00:00", "title": "Claim 6 \u2014 Binning quantizers preserve policy smoothness better than learned; deterministic experts violate RTVC more (\u00a74.1, empirical)"}\n-->
