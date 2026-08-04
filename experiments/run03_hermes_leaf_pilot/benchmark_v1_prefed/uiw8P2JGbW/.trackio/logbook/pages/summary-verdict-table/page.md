## Summary verdict table

| # | Anchored claim | Verdict | Source | Key reproduction |
|---|---------------|---------|--------|-----------------|
| 1 | FPA revenue monotonicity for tCPA (μ=1, no budget) | **verified** | Thm 5.1 / App. A.2 | Rev(M_A) ≥ Rev(M_B) on 2000/2000 refined pairs |
| 2 | Convexity of f + mean-preserving spread (proof mechanism) | **verified** | Thm 5.1 proof / App. A.2 | 0 Jensen violations; MPS & Jensen hold per cluster |
| 3 | μ=1 optimal & welfare-monotonicity corollary | **verified** | Thm 5.2 / Cor. 5.3 | Revenue ↑ in μ, max at 1.0; welfare mono 2000/2000 |
| 4 | VCG/SPA tCPA non-monotone, 6.2% both | **verified** | Thm 5.8 / App. B.1 | 3.23 → 3.03, drop 6.20% rev / 6.19% welf |
| 5 | Budget breaks FPA monotonicity, 16.8% | **verified** | Thm 5.10 / App. B.2 | 5.5268 → 4.5977, drop 16.81% |
| 6 | LP lifting monotonicity + Table 1 (3 settings) | **verified** | Thm 5.11 / App. B.5; Table 1 | Lift feasible+equal objective; 3 setting families |

**All 6 anchored claims reproduced and verified.** No GPU/network required;
every computation is from first principles in `common.py` + `verify_claim<N>.py`.
Full numeric detail is in `results/claim<N>.json`.

---

---
<!-- trackio-cell
{"type": "markdown", "id": "cell_<built-in function hash>", "created_at": "2026-01-01T00:00:00+00:00", "title": "Summary verdict table"}\n-->
