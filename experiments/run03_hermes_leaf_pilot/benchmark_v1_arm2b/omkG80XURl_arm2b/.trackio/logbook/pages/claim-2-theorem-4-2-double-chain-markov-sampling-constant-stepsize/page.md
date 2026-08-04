## Claim 2 — Theorem 4.2 (double-chain, Markov sampling, constant stepsize)
`verify_claim2.py` → `results/claim2.json`. Same instances with mixing slowed (`gap=0.6`), states drawn from two *independent trajectories* of P.
- η swept 0.0731→0.0213, α = Θ(ε·η): hitting times 11 564 → 99 543, log-log **exponent −1.67** (R²=0.992) — same rate as under i.i.d. sampling (−1.64), so Markov noise does not degrade the complexity.
- Fixed point under Markov noise: max distance to θ* **0.0665**, spread 0.0294.
- **Mutation:** coupled (single) chain under Markov sampling → distance to θ* **1.774**.
- Verdict: **verified**.

---
<!-- trackio-cell
{"type": "markdown", "id": "cell_<built-in function hash>", "created_at": "2026-01-01T00:00:00+00:00", "title": "Claim 2 \u2014 Theorem 4.2 (double-chain, Markov sampling, constant stepsize)"}\n-->
