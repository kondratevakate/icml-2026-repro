## Claim 4 — Section 1 / Table 1 (quartic → quadratic)
`verify_claim4.py` → `results/claim4.json`.
- Double-chain measured η-exponents: **−1.64** (i.i.d.) and **−1.67** (Markov) — quadratic or better.
- Executable quartic reference: the single-chain / decorrelation-limited algorithm in the same harness gives **−3.76** (claim 5). Gap ≈ 2 in the exponent, exactly the quartic→quadratic improvement asserted.
- Diagnostic (not a gate): a discounted-TD sweep with η_disc=(1−γ)σ_min(Φ'DΦ) gave exponent −0.94 (R²≈1.0), i.e. ≤2 — the average-reward method is not worse than discounted TD in this harness, but η_disc is only a proxy so this does not by itself establish "matching".
- **Scope caveat:** the *prior-work* algorithms of refs [11,17,21] were **not** re-implemented; the quartic side of the comparison is reproduced via the paper's own single-chain variant, which Theorem 4.4 places at quartic. The literature-attribution part of the claim is not independently checkable here.
- Verdict: **verified** for the executable content (their method quadratic; a decorrelation-limited method quartic), with the caveat above.

---
<!-- trackio-cell
{"type": "markdown", "id": "cell_<built-in function hash>", "created_at": "2026-01-01T00:00:00+00:00", "title": "Claim 4 \u2014 Section 1 / Table 1 (quartic \u2192 quadratic)"}\n-->
