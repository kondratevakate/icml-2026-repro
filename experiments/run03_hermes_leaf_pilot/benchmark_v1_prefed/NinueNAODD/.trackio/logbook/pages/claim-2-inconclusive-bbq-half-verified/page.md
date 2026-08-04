## Claim 2 — — inconclusive (BBQ half verified)

**Verdict:** fig. 3 / sec. 4.4

*Source:* Fig. 3 / Sec. 4.4; dataset shapes App. D Table 6. *Script:* `verify_claim2.py` (25 s).

- BBQ wall-clock, plain NumPy: IHQ-scr **0.069 s** (121 it), IHQ-unscr **0.098 s** (115 it), IHQ-all **0.393 s** (200 it), and a synthetic **HUMAINE-shaped** set (104,781 comparisons, 1,977 raters, 27 items) **1.38 s** (25 it). → "converges within seconds" ✔ on everything tested.
- The "≈15 minutes for Crowd-BT" figure is implementation-specific. Our Crowd-BT costs **4.6 s/epoch** on the HUMAINE-shaped set (≈4.6 min for 60 epochs) — same order of magnitude, but it is *not* Google's code and not the real HUMAINE data, so this neither confirms nor refutes the number.
- **Mutation:** subsampling to 20 raters (~1% of data) cuts BBQ runtime 47× (1.38 s → 0.029 s), confirming the timing is data-size driven rather than a fixed cost. ✔

---
<!-- trackio-cell
{"type": "markdown", "id": "cell_<built-in function hash>", "created_at": "2026-01-01T00:00:00+00:00", "title": "Claim 2 \u2014 \u2014 inconclusive (BBQ half verified)"}\n-->
