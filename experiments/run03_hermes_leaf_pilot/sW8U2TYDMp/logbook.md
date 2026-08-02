# Reproduction logbook — sW8U2TYDMp

**Paper:** *Probabilistic Modeling of Latent Agentic Substructures in Deep Neural Networks* —
Su Hyeong Lee, Risi Kondor, Richard Ngo. OpenReview `sW8U2TYDMp`, arXiv **2509.06701v2** (ICML 2026).
**Arm:** arm1 (autonomous). **Hardware:** CPU only (WSL2), numpy 2.5.1 / scipy / sympy in local `.venv`.
**Global seed:** `20260802` (`common.SEED`; each script uses `default_rng(SEED + claim_index)`).
**Reproduce:** `.venv/bin/python verify_claim<N>.py` → writes `results/claim<N>.json`, stdout mirrored in `log_claim<N>.txt`.

## Nature of the paper and of this reproduction

The paper is **purely theoretical**: no datasets, no models, no released code, no experiments.
"Reproduction" therefore means *independent numerical verification of the mathematical content*:
each anchored claim was re-derived from first principles (definitions of linear/logarithmic pooling,
log-score welfare, P-centered log profiles) and then checked numerically — by exhaustive/dense search
for impossibility claims, by explicit construction for existence claims, and by closed-form-vs-brute-force
agreement for the variational (Waluigi) claims. Every verified claim also gets a **mutation test**: a
perturbation of the setup under which the claimed property must (and does) break.

Numbering note: the task/OpenReview numbering (Thms 8, 9, 10, 14, 19, 21) is the **main-text** numbering
of v2; the corresponding appendix statements are Thms 33/34, 37, 46, 69, 75.

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

## Claim 1 — epistemic utility is the log score
**Source:** Sec. 2 (`U_i(o) = log P_i(o)`), Def. 2 (log pool), Def. 7 + Prop. 32 (welfare gap).
**Method (`verify_claim1.py`, 4000 trials, \|O\| = 5, n = 3):** checked that (a) the log score is strictly
proper, (b) `softmax(log P) = P`, (c) `softmax(Σ β_i log P_i)` equals the logarithmic pool exactly, and
(d) the identity `Δ_i = H(P_i) − H(P) − KL(P‖P_i)`.
**Result:** properness held in all trials; max errors `3.3e−16` (b), `0.0` (c), `1.8e−15` (d).
**Mutation:** replacing the log score with the linear score `W_i = P_i` breaks softmax recovery
(err 0.49) and the identity (err 3.79) — i.e. the whole apparatus is specific to the log score.

## Claim 2 — Theorem 10 (linear-pool impossibility)
**Source:** Sec. 3.1 Thm 10 (App. E Thm 37).
**Method (`verify_claim2.py`):** re-derived the impossibility as a two-step inequality
`Σ_i β_i Δ_i ≤ −H(P) + Σ_i β_i H(P_i) ≤ 0` (Jensen on `Σβ_i log P_i ≤ log Σβ_i P_i`, then concavity of
entropy). Checked both steps and the conclusion over 6000 random configurations
(\|O\| ∈ 2..6, n ∈ 2..5, Dirichlet concentrations 0.3/1/3), plus 60 Nelder–Mead restarts directly
maximising `min_i Δ_i` over beliefs and weights.
**Result:** max Jensen slack `0.0`, max entropy slack `0.0`, max β-weighted Δ-sum `−2.3e−7`,
max `min_i Δ_i` over random trials `−3.6e−6`, optimiser best `1.7e−21` (numerically zero, attained only
at the degenerate all-agents-identical point). Impossibility holds.
**Mutation:** identical search under the **logarithmic** pool finds 9 strictly unanimous configurations,
best `min_i Δ_i = 0.0571` — the property is specific to linear pooling. The "random dictatorship"
intuition is exactly the Jensen step; note the *mechanism* wording is interpretive, what is verified is
the impossibility itself.

## Claim 3 — Theorems 8/9/10 frontier
**Source:** Sec. 3.1 Thms 8, 9, 10.
**Method (`verify_claim3.py`):** random search over sparse Dirichlet beliefs for a log pool with all
`Δ_i > 0`, for every (\|O\|, n) with \|O\| ∈ {3,4,5,6}, n ∈ {2,3,4}; then the |O| = 2 mutation with a
dense 400 × 57 × 99 grid over `(x₁, x₂, β)` plus 40 000 random draws.
**Result:** strict unanimity found for **every** size; best `min_i Δ_i = 0.2254` at \|O\| = 3, n = 2
(explicit witness stored in `results/claim3.json → witness_K3_n2`), and 0.145–0.293 across the grid.
**Mutation:** for \|O\| = 2, best `min_i Δ_i = −8.4e−7` (grid) and `−2.6e−8` (random) — never positive,
matching Thm 8.
*Honesty note:* an earlier run with an 8 000-draw budget missed the K = 5, n = 4 cell (reported −0.107);
that was a **search-budget artifact**, not a falsification — it disappeared at 30 000 draws with two
concentrations. Only the fixed run is recorded in the JSON.

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

## Claim 5 — Theorem 21 (Waluigi shattering)
**Source:** Sec. 5.1 Thm 21; App. J Lemma 20/71, Prop. 72, Cor. 74, Thm 75.
**Method (`verify_claim5.py`, \|O\| = 8, dim S₀ = 3, ε = 0.02, 1932 trials):** used Lemma 20
(`P′(A) − P(A) = ⟨ΔL, g_A⟩_P + o(·)`) and the closed form `M(S) = ε‖Proj_S g_A‖_P`, verified against
brute-force constrained maximisation inside the ball (max relative error 2.6e−2 with 3000 random
directions — consistent with random-search slack, the closed form is always the upper envelope).
**Result:** whenever `⟨g_A, u⟩_P ≠ 0`, the gap `M(S₁) − M(S₀)` is strictly positive in **all** trials
(median 4.7e−4, minimum 4.6e−11).
**Mutation:** (a) `w ∈ S₀` (so `u = 0`) → gap exactly `0.0`; (b) `u` built orthogonal to `g_A` → gap
`1.7e−18`. So the strictness *requires* the hypothesis `u ≠ 0` and `⟨g_A, u⟩_P ≠ 0`.
**Verdict: verified (conditional).** Corollary 74 states that hypothesis explicitly and is fully
reproduced. The **main-text wording of Theorem 21** (and of the abstract) asserts
`M(P′_shatter) − M(P′_pure) > 0` while only supposing that `w` is "an anti-aligned direction" — dropping
the `⟨g_A, u⟩ ≠ 0` condition. Mutation (b) is an explicit counterexample to that literal statement, so
the main-text phrasing is an over-claim relative to the proof; the underlying result is sound.

## Claim 6 — Theorem 14 (benefit does not descend to subagents)
**Source:** Sec. 4.1 Thm 14 (App. G Thm 46), with Lemma 13 (compatible splitting).
**Method (`verify_claim6.py`, \|O\| = 4, n = 2):** found a parent with `Δ_{P₁}(P) = +0.0914 > 0`, then
split it compatibly as `P_{1,1} ∝ P₁ e^{tφ}`, `P_{1,2} ∝ P₁ e^{−tφ}` with α = (½, ½) and a mean-zero
tilt φ, scanning t ∈ [0, 3].
**Result:** compatibility (`P₁ ∝ P_{1,1}^{½}P_{1,2}^{½}`) holds to 1.7e−16 and Lemma 13 pool invariance
to 1.1e−16 across the whole scan. At **t = 0.29** the counterexample appears:
`Δ_parent = +0.0914`, `Δ_child1 = −0.00239 < 0`, `Δ_child2 = +0.156` — the parent strictly benefits while
one child is strictly harmed, exactly the tilt mechanism described in the paper.
**Mutation:** the cloning split (t = 0, `P_{1,1} = P_{1,2} = P₁`) leaves both children with
Δ = +0.0914 > 0 — the failure disappears, consistent with Lemma 48 (cloning preserves welfare).

---

## Limitations / honesty statement
* No GPU work, no data, no model was needed or attempted — the paper contains no empirical component,
  so there is nothing to reproduce beyond the mathematics. The "Waluigi effect" claims are verified as
  statements about the paper's probabilistic model, **not** as claims about any real LLM; the paper itself
  says so ("Scope of the interpretation", Sec. 5).
* Impossibility claims (2, 3-binary) are verified by dense/exhaustive numerical search plus a re-derived
  analytic argument, which is strong evidence but not a formal proof check (no Lean/Coq formalisation).
* Claim 4's strict regime required constructed rather than random witnesses (documented above); claim 5's
  strictness is conditional on a hypothesis the main text omits (documented above).
* All searches are seeded (`20260802`) and rerun deterministically.

## Files
```
common.py             shared pooling / welfare-gap utilities, seed
verify_claim1..6.py   one script per anchored claim (each with its mutation test)
results/claim1..6.json numeric results, verdicts, witnesses, mutation outcomes
log_claim1..6.txt     stdout of each run
```
