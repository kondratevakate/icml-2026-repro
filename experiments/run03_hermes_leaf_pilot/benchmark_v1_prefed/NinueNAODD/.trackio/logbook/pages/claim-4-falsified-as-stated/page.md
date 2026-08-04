## Claim 4 — — falsified (as stated)

**Verdict:** table 1, sec. 4.1

*Source:* Table 1 + Sec. 4.1. *Script:* `verify_claim4.py` (15 min).

Ranking the three methods by Kendall's τ using the paper's **own** Table 1 numbers:

`HUMAINE 1 · MT-Bench 2 · WD 2 · HiFiC 1 · ConHa 1 · IHQ-all 1 · IHQ-scr **3** · IHQ-unscr 1`

- "first on 5 of 8" → **true** (5 firsts).
- "100% top-1 on MT-Bench, WD, HiFiC" → **true** per Table 1 (not independently reproduced — datasets unavailable).
- "second on the remaining three" → **false**: BBQ is second on only 2, and **third on IHQ-screened** (BBQ 0.9204 < Bayes-BT 0.9211 < Crowd-BT 0.9238). The conjunctive claim is false against its own source table.
- **Mutation** (audit): shaving 0.01 off BBQ's HUMAINE τ drops the first-place count 5 → 4. ✔ the count is sensitive, not a tautology.
- Partial empirical check on the 3 IHQ splits (200 bootstraps, τ vs the full-data BBQ ranking — a reference that *favours* BBQ): BBQ best on IHQ-all (0.9320), but Crowd-BT best on IHQ-scr (0.9135 vs 0.9098) and IHQ-unscr (0.8758 vs 0.8704). Consistent with the paper's own finding that BBQ is not first on IHQ-screened.

---
<!-- trackio-cell
{"type": "markdown", "id": "cell_<built-in function hash>", "created_at": "2026-01-01T00:00:00+00:00", "title": "Claim 4 \u2014 \u2014 falsified (as stated)"}\n-->
