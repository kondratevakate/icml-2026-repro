## Summary

| # | Claim (source) | Verdict | Mutation test |
|---|---|---|---|
| 1 | Agents = distributions, epistemic utility `U(o) = log P(o)` (Sec. 2, Def. 7, Prop. 32) | **verified** | linear score `W_i = P_i` → softmax recovery and welfare-gap identity both break |
| 2 | Thm 10: strict unanimity impossible under **linear** pooling, any \|O\| | **verified** | same search with **log** pooling → 9 strictly unanimous configs found |
| 3 | Thm 9: strict unanimity achievable under log pooling for \|O\| ≥ 3 (vs. Thm 8 binary, Thm 10 linear) | **verified** | collapse to \|O\| = 2 → dense grid + random search max min Δ ≈ −8e−7 ≤ 0 |
| 4 | Thm 19: manifesting Luigi under a stability budget forces weight onto an anti-aligned (Waluigi) component | **verified** | inflate ε to 5·δ‖v_H‖ → T₁ < 0, bound vacuous; all-aligned witnesses → conclusion void |
| 5 | Thm 21: manifest-then-suppress beats pure Luigi reinforcement | **verified (conditional)** | w ∈ S₀ (u = 0) or u ⟂ g_A → gap exactly 0 (≤1.7e−18) |
| 6 | Thm 14: parental compositional benefit need not pass to child subagents | **verified** | cloning split (t = 0) → both children inherit Δ = +0.0914 > 0 |

No claim was falsified. One claim (5) is verified only under a hypothesis that the main text omits — see below.

---

---
<!-- trackio-cell
{"type": "markdown", "id": "cell_<built-in function hash>", "created_at": "2026-01-01T00:00:00+00:00", "title": "Summary"}\n-->
