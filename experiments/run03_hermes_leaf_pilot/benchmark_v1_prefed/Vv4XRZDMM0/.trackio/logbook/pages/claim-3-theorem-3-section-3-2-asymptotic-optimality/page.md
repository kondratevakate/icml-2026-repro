## Claim 3 — Theorem 3 (Section 3.2), asymptotic optimality

**Statement tested.** With `ĥ = Σ_k λ̂_k(x) f̂_k(y|x)`, score `s_k = −ĥ`, the paper's randomized
p-values and `Ĉ^{(n)}(x) = {y : max_k p^{(k)} ≥ α}`:
`limsup_n ||Ĉ^{(n)}| − |C*|| ≤ ρ(T)`, `T = {(x,y) : h*(x,y) = 1}`, and if ρ(T)=0 then
`ρ(Ĉ^{(n)} △ C*) → 0`.

**Script.** `verify_claim3.py` → `results/claim3.json` (runtime 21.3 s). K = 3, |X| = 5, |Y| = 12,
α = 0.1; oracle λ*(x), h*(x,·), `{h*>1}` and `T(x)` computed exactly per x by LP; `f̂_k` from
multinomial training counts and `λ̂(x)` by re-solving the LP on `f̂` (both consistent, as Thm 3 assumes);
n per source ∈ {100, 400, 1600, 6400, 25600}, 8 worlds, 120 randomizations each.

**Numbers.**
- Boundary mass in these worlds is **not** negligible: mean `ρ(T) = 2.875` labels per x — the LP optimum is
  fractional on `T`, so the theorem's slack term is genuinely active. The theorem's inequality
  `||Ĉ| − |C*|| ≤ ρ(T)` held in **8/8 worlds at all 5 sample sizes (40/40 cells)**.
- Symmetric difference to `{h*>1}` **restricted to the complement of T** (the part Thm 3 controls):
  **1.2823 → 0.4696 → 0.4654 → 0.0694 → 0.0546** for n = 100 … 25 600. Converging to 0 as claimed.
- Finite-sample validity survives throughout: min worst-source coverage ≥ **0.874** across all cells
  (Theorem 1 does not depend on the score, as the paper stresses).

**Mutation tests** (same worlds, only the score-learning mechanism broken):
| mutation | symmetric difference outside T at n = 25 600 | predicted |
|---|---|---|
| main (consistent λ̂, f̂) | **0.0546** | → 0 ✔ |
| λ̂ frozen at a uniform vector (violates `sup_x‖λ̂−λ*‖→0`) | **0.3577** (flat from n = 1600 on) | does not vanish ✔ |
| score built from a single source density `f̂_0` | **1.6233** (flat) | does not vanish ✔ |

**Verdict: verified** — convergence happens only when the premises of Thm 3 hold, and stalls at a
positive plateau when either premise is broken, while coverage stays valid in every arm.

---

---
<!-- trackio-cell
{"type": "markdown", "id": "cell_<built-in function hash>", "created_at": "2026-01-01T00:00:00+00:00", "title": "Claim 3 \u2014 Theorem 3 (Section 3.2), asymptotic optimality"}\n-->
