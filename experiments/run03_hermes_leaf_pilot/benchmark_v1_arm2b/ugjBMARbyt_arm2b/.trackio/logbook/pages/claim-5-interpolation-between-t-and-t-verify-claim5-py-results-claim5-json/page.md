## Claim 5 — Interpolation between Õ(√T) and Õ(T) (`verify_claim5.py` → `results/claim5.json`)

**Anchor:** v2 Theorem 5.3 with ζ(n) = 1 − n^{−q}, and Corollary 5.4.

**Numbers.** Analytic exponent (q+2)/(2q+2) is monotone decreasing in q, equals `0.976` at
q = 0.05 (→ 1, i.e. O(T)) and `0.515` at q = 32 (→ 1/2, i.e. Õ(√T)). Numerically evaluating the
**full** Thm-5.3 expression over T ∈ {10³…10⁶} gives fitted log-log slopes
`0.982, 0.972, 0.935, 0.880, 0.807, 0.734, 0.678, 0.642, 0.611` for
q = `0.05, 0.1, 0.25, 0.5, 1, 2, 4, 8, 32` — each ≥ the analytic exponent and converging to it
from above (the excess is the log factor on a finite range).
Simulation: instances with γ_i ∝ i^{−(q+1)/2} run with n_t = ⌈t^{1/(q+1)}⌉ stay below the certified
bound at every T, and the realised exponent is ordered as predicted (q = 0.1 worse than q = 2).

**Mutation.** Freezing the truncation order at n_t = 1 (never growing the basis) gives linear
regret: `37.1, 71.0, 138.8, 274.4, 545.5`, slope `0.970`.

**Verdict: verified.**

---
<!-- trackio-cell
{"type": "markdown", "id": "cell_<built-in function hash>", "created_at": "2026-01-01T00:00:00+00:00", "title": "Claim 5 \u2014 Interpolation between \u00d5(\u221aT) and \u00d5(T) (`verify_claim5.py` \u2192 `results/claim5.json`)"}\n-->
