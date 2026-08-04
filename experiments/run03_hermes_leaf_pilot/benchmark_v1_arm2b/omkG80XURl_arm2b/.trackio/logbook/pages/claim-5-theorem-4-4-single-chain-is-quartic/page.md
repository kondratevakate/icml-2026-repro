## Claim 5 — Theorem 4.4 (single-chain is quartic)
`verify_claim5.py` → `results/claim5.json`. Same instances, same ε target, both algorithms run side by side. Stepsizes follow each theory's floor: α ∝ ε·η (double) and α ∝ ε·η³ (single, since its floor is Õ(ατ/(η′η²))).
- η 0.0731→0.0410: `T_single` 7 548 → 75 826, `T_double` 4 595 → 13 935.
- Fits: single **−3.76** (R²=0.994) vs double **−1.91** (R²=0.996); exponent gap **1.85**. Predicted −4 vs −2.
- The contrast itself is the mutation: the only structural change is replacing the independent second chain φ(ŝ_t) by the running average w_t, and the exponent degrades by ~2 — the decorrelation cost the theorem attributes it.
- Verdict: **verified**.

---
<!-- trackio-cell
{"type": "markdown", "id": "cell_<built-in function hash>", "created_at": "2026-01-01T00:00:00+00:00", "title": "Claim 5 \u2014 Theorem 4.4 (single-chain is quartic)"}\n-->
