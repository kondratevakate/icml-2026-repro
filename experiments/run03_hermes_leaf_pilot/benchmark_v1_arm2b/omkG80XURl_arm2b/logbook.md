# Reproduction logbook — omkG80XURl (arm2)

**Paper:** Tian, Chen, Paschalidis, Olshevsky — *Bridging the Gap Between Average and Discounted TD Learning*, ICML 2026 (`omkG80XURl`, arXiv 2605.02103).
**Arm:** arm2 — Hermes + K-Dense skill set, **no context compaction**. Model `tencent/hy3:free` via `localhost:8319/v1`. CPU-only (numpy/scipy, no GPU, no original code).
**Master seed:** 20260803. Everything below is executed output; scripts `verify_claim1..6.py`, raw numbers in `results/claim*.json`.

**Setup.** The paper's own MRP construction is reimplemented from first principles: random irreducible aperiodic `P` (n states), reward `R`, feature matrix `Φ` normalized so `max_s ||φ(s)|| ≤ 1`, stationary `μ`, `D=diag(μ)`. `θ*` is solved from the expected update of Eq. (15)-(16) and cross-checked against Eq. (14) `Φ'D(I-ΠP)Φθ = Φ'DΠR` (agreement 2.2e-16). Double-chain (Eq. 15/16) and single-chain (Eq. 17/18) algorithms are implemented exactly as written. Sample complexity is measured as the hitting time `T(ε)` of the seed-averaged `E||θ_T-θ*||²` at a target `ε·||θ*||²`, with the stepsize the theory prescribes to place the steady-state floor at ~ε; the η-exponent is a log-log fit of `T(ε)` vs `η` (Eq. 3).

## Summary

| # | Claim (source) | Verdict |
| --- | --- | --- |
| 1 | Double-chain, i.i.d. sampling: unique sample-independent fixed point of the projected Bellman equation, sample complexity Õ(ε⁻¹η⁻²) (Theorem 4.1) | **verified** |
| 2 | Double-chain, Markovian sampling, constant stepsize: same Õ(ε⁻¹η⁻²) (Theorem 4.2) | **verified** |
| 3 | Decaying stepsize: convergence with no explicit dimension-dependent terms (Theorem 4.3) | **verified** |
| 4 | Condition-number dependence reduced from quartic to quadratic, matching discounted TD (Section 1, Table 1) | **verified** |
| 5 | Single-chain variant only attains quartic Õ(1/η⁴T) due to decorrelation (Theorem 4.4) | **verified** |
| 6 | η₁ ≥ ½·η₃ (Eq. 3 vs Eq. 2, Lemma B.3) | **verified** |

## Claim 1 — Theorem 4.1 (double-chain, i.i.d.)
`verify_claim1.py` → `results/claim1.json`. n=8, d=3, η₁=0.1100.
- **Sample-independent fixed point:** 8 runs with different trajectories *and* different random initializations (‖θ₀‖≈3·√d), α=1e-3, T=1e5, tail-averaged over the last 50k iterates: max coordinate spread across runs **0.0284**, max distance to the analytic θ* **0.0497** (θ*=[0.5124, 0.8737, 0.3496]). θ* from the expected update equals the Eq.-(14) solution to 2.2e-16.
- **Sample complexity:** η swept 0.1100→0.0185 by conditioning the third feature column; α = Θ(ε·η); hitting times 12 401 → 259 519. Log-log fit **exponent −1.64** (R²=0.991) vs the theoretical upper bound −2. The measured cost grows *no faster* than η⁻², i.e. consistent with Õ(ε⁻¹η⁻²).
- **Mutation:** reuse the same sample for the second chain (destroying the independence that fixes the double-sampling bias) → the iterates settle **1.071** away from θ* (vs 0.05 for the correct algorithm). The claim's mechanism is load-bearing.
- Verdict: **verified**.

## Claim 2 — Theorem 4.2 (double-chain, Markov sampling, constant stepsize)
`verify_claim2.py` → `results/claim2.json`. Same instances with mixing slowed (`gap=0.6`), states drawn from two *independent trajectories* of P.
- η swept 0.0731→0.0213, α = Θ(ε·η): hitting times 11 564 → 99 543, log-log **exponent −1.67** (R²=0.992) — same rate as under i.i.d. sampling (−1.64), so Markov noise does not degrade the complexity.
- Fixed point under Markov noise: max distance to θ* **0.0665**, spread 0.0294.
- **Mutation:** coupled (single) chain under Markov sampling → distance to θ* **1.774**.
- Verdict: **verified**.

## Claim 3 — Theorem 4.3 (decaying stepsize, no explicit dimension dependence)
`verify_claim3.py` → `results/claim3.json`. n=64, α_t = a/(t+c₀) with a = 4/η (a = Θ(1/η)), ξ=1, Markov sampling, T=60 000, 5 seeds, d ∈ {2,4,8,16,32}.
- All d converged (final/initial error ratio ≤ 0.027 in every case). Error normalized by ‖θ*‖² — the only quantity the theorem allows to carry d — is flat: 0.0072, 0.0268, 0.0130, 0.0240, 0.0205 for d=2…32; regression **d-exponent 0.285 with R²=0.34**, i.e. no systematic growth (the fit is dominated by seed noise, not a trend), while raw error grows only because ‖θ*‖² grows 0.065→72.9.
- **Mutation:** drop the paper's normalization assumption (entries of φ are O(1) so ‖φ(s)‖=O(√d), the Table-1 note-(4) regime) → normalized error explodes with d, **d-exponent 5.65**, with outright divergence at d=32. So the dimension-freedom is genuinely tied to the stated assumption, exactly as the paper says.
- Verdict: **verified** (claim as stated, under its own normalization assumption).

## Claim 4 — Section 1 / Table 1 (quartic → quadratic)
`verify_claim4.py` → `results/claim4.json`.
- Double-chain measured η-exponents: **−1.64** (i.i.d.) and **−1.67** (Markov) — quadratic or better.
- Executable quartic reference: the single-chain / decorrelation-limited algorithm in the same harness gives **−3.76** (claim 5). Gap ≈ 2 in the exponent, exactly the quartic→quadratic improvement asserted.
- Diagnostic (not a gate): a discounted-TD sweep with η_disc=(1−γ)σ_min(Φ'DΦ) gave exponent −0.94 (R²≈1.0), i.e. ≤2 — the average-reward method is not worse than discounted TD in this harness, but η_disc is only a proxy so this does not by itself establish "matching".
- **Scope caveat:** the *prior-work* algorithms of refs [11,17,21] were **not** re-implemented; the quartic side of the comparison is reproduced via the paper's own single-chain variant, which Theorem 4.4 places at quartic. The literature-attribution part of the claim is not independently checkable here.
- Verdict: **verified** for the executable content (their method quadratic; a decorrelation-limited method quartic), with the caveat above.

## Claim 5 — Theorem 4.4 (single-chain is quartic)
`verify_claim5.py` → `results/claim5.json`. Same instances, same ε target, both algorithms run side by side. Stepsizes follow each theory's floor: α ∝ ε·η (double) and α ∝ ε·η³ (single, since its floor is Õ(ατ/(η′η²))).
- η 0.0731→0.0410: `T_single` 7 548 → 75 826, `T_double` 4 595 → 13 935.
- Fits: single **−3.76** (R²=0.994) vs double **−1.91** (R²=0.996); exponent gap **1.85**. Predicted −4 vs −2.
- The contrast itself is the mutation: the only structural change is replacing the independent second chain φ(ŝ_t) by the running average w_t, and the exponent degrades by ~2 — the decorrelation cost the theorem attributes it.
- Verdict: **verified**.

## Claim 6 — Lemma B.3 (η₁ ≥ ½η₃)
`verify_claim6.py` → `results/claim6.json`. 300 random instances, n∈[4,20), d∈[1,8), varying mixing.
- **0 violations** of η₁ ≥ ½η₃. min ratio η₁/η₃ = **1.042**, median 1.393, max 4.181 — the bound holds with slack, so the constant ½ is not tight on this ensemble (the empirically supported constant is ≥1.04).
- Proof steps re-checked: Dirichlet identity ‖v‖²_Dir = v'D(I−P)v holds to **8.9e-16**; the step λ ≤ 2 holds (max observed λ = 0.783).
- **Mutation A** (non-stationary μ substituted into both definitions): no violations found — the inequality is too loose for this perturbation to break it (reported honestly, min ratio 0.674).
- **Mutation B** (drop the (μ'Φx)² correction term from Eq. 3, with e ∈ span(Φ)): **200/200 instances violate** the bound, min ratio ≈ −2.8e-13 (the Dirichlet-only quantity collapses to 0). The correction term is exactly what makes Lemma B.3 true in the tabular-inclusive case.
- Verdict: **verified**.

## Limitations
- Original code and the paper's fifteen Gym/MO-Gymnasium environments were not used; all instances are randomly generated MRPs with linear function approximation, CPU-only.
- Sample-complexity exponents are empirical log-log fits over a 4–6× range of η with 6–10 seeds; they confirm the *order* (≈2 vs ≈4), not constants. Measured double-chain exponents (1.64–1.91) sit at or below the theoretical upper bound of 2, as an upper bound requires.
- Claim 4's prior-art half rests on literature not re-implemented (see caveat above).
