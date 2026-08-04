## Claim 4 — Theorem 19 (Waluigi emergence)
**Source:** Sec. 5 Thm 19 (App. I Thm 69), with Thm 18 as the first-order expansion.
**Method (`verify_claim4.py`), three blocks, δ = 0.05:**
1. *Random witnesses (\|O\| = 6, n = 4, exactly one anti-aligned component).* For 1419 feasible weight
   perturbations with `Δβ_H = δ`, `Σ Δβ = 0`, the **master inequality**
   `Σ_{anti} (Δβ_i)⁺ |⟨v_i,v_H⟩| ≥ T₁ + T₂` was checked with ε set to the *actual* `‖ΔL‖_P` and `r` the
   *exact* remainder `ΔL − Σ Δβ_i v_i`. **0 violations**, min slack `+0.0065`.
2. *Contrapositive form.* Among perturbations satisfying the corollary hypothesis (no aligned component
   down-weighted), **0** stayed inside the budget while leaving the Waluigi weight non-increasing.
   With random witnesses the strict regime `ε + ‖r‖ < δ‖v_H‖` was in fact **unreachable** — reported
   openly in the JSON (`two_anti_aligned_config`: min-norm `‖ΔL‖ = 0.0549 > δ‖v_H‖ = 0.0478`, T₁ < 0).
   A vacuous check would have been a false positive, so:
3. *Constructed witnesses that reach the strict regime.* \|O\| = 4, uniform base `P`, centered
   P-orthogonal directions `a` (Luigi) and `b` with `‖b‖ = ‖a‖/20`, profiles
   `v = (a, −a, −0.2a + 2b, 0.2a − 2b)` (sum zero ⇒ the uniform-β log pool reproduces `P` to 5.6e−17),
   two anti-aligned components with *distinct* inner products. Sweeping `Δβ = (δ, M, −(δ+M), 0)`,
   M/δ ∈ [0.5, 3]: **22 of 26 points fall in the strict regime** (e.g. M = 1.5δ: ε = 0.00988,
   ‖r‖ = 4.9e−5, δ‖v_H‖ = 0.0395, T₁ = 0.0234), and in every one the Waluigi weight is strictly positive
   and exceeds the theorem's lower bound (Δβ_W = 0.075 vs. bound 0.0374 at M = 1.5δ). Master inequality
   holds throughout.
**Mutation:** (a) inflating the budget to ε = 5δ‖v_H‖ makes `T₁ = −0.425 < 0` — the bound becomes vacuous
and no Waluigi up-weighting is forced; (b) a witness set in which every component is aligned with `v_H`
(found by search) voids the conclusion entirely.
**Caveat recorded:** "necessarily strengthens" is *conditional* on `ε + ‖r‖_P < δ‖v_H‖_P`. That regime is
not automatic — for randomly drawn witnesses it was empirically unreachable, and it required deliberate
construction. The theorem is correct as stated; its precondition is restrictive.

---
<!-- trackio-cell
{"type": "markdown", "id": "cell_<built-in function hash>", "created_at": "2026-01-01T00:00:00+00:00", "title": "Claim 4 \u2014 Theorem 19 (Waluigi emergence)"}\n-->
