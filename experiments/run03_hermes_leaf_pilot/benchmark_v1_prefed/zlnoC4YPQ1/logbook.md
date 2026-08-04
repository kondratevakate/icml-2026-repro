# logbook.md — reproduction of "Robust Bayesian Optimisation with Unbounded Corruptions"

Paper: arXiv 2511.15315v2 (OpenReview zlnoC4YPQ1). All work in this directory, CPU only,
Python 3.12 (`.venv`), numpy/scipy/sympy/mpmath. Paper read once → `notes_paper.md`.
Official code repository was **not** used (no `notes_code.md`): all five anchored claims were
addressed from the paper's own equations plus a from-scratch implementation (`rcgp.py`, `bo.py`).

Timing: first tool call 02:14 (+04) 2026-08-02, finished ~03:30 → **elapsed ≈ 1h20m**
(hard budget 8h, soft 4h, per-claim 2h — all respected). LLM turns used: ~20 (budget 60).

Artifacts: `plan.md`, `notes_paper.md`, `rcgp.py`, `bo.py`, `verify_claim1..5.py`,
`results/claim1..5.json` (+ `.log`), `check_reproducibility.py`.

---

## Claim 1 — Theorem 4.2 (FC-RCGP-UCB), sublinear regret for α < 1/4
**Source:** Theorem 4.2, Sec 4.3; sublinearity discussion Sec 4.3 "Conditions for Sublinear Regret",
Appendix D.4.3, Table 1 (Appendix E).
**Verdict: verified** (as an implication: stated bound ⇒ stated α threshold).

Substituting T_c = T^α, Ψ(T_c) = √(1+T_c(1+T_c)) (Sec 4.1, κ = σ_noise² = 1), γ_T = Õ(1),
β'_T = Õ(1) into R_T = Õ(Ψ(T_c)(1+√T_c)√(β'_T T(γ_T+T_c))) gives, symbolically (sympy),

* growth exponent **e(α) = 2α + 1/2**, hence e(α) < 1 ⟺ **α < 1/4** (`alpha_threshold_rbf = 1/4`);
* this coincides exactly with the paper's intermediate form R_T = Õ(T_c²√T)
  (`intermediate_matches_bound = true`);
* numeric exponents: α=1/10 → 0.6999, α=1/5 → 0.9000, α=1/4 → 1.0000, α=1/3 → 1.1667 (not sublinear),
  α=1/2 → 1.5000;
* all **8 Matérn/domain-case entries of Table 1 for FC** reproduce to grid resolution 1/4000
  (e.g. η=1/3, cases 1&2: 0.222 vs 2/9; η=1/2, case 3: 0 vs 0 — no permissible α).

**Mutation test:** removing the observability penalty Ψ (set Ψ≡1) changes the exponent to α + 1/2
and the threshold to **α < 1/2**, which is exactly the paper's "Ideal" column in Table 1
(`matches_paper_ideal = true`, `verdict_changed = true`). The threshold therefore genuinely comes
from Ψ, not from an artefact of the algebra.

## Claim 2 — Theorem 4.3 (A2-RCGP-UCB), sublinear regret for α < 1/7
**Source:** Theorem 4.3, Sec 4.3; Appendix D.5.5; Table 1 (Appendix E).
**Verdict: verified** (same implication sense as claim 1).

R_T = Õ((1+T_c Ψ(T_c)²)√(β'_T T(γ_T+T_c))) with T_c = T^α gives exponent **e(α) = 7α/2 + 1/2**,
so sublinear ⟺ **α < 1/7 = 0.142857…**; matches the paper's intermediate R_T = Õ(T_c^{7/2}√T).
Numeric: α=1/7 → 1.0000, α=1/6 → 1.0833 (not sublinear), α=1/4 → 1.3750, α=1/3 → 1.6667.
All 8 Matérn entries of Table 1 for A2 reproduce (grid 1/7000), incl. η=1/4 case 3: 0.08329 vs 1/12.
**Mutation test:** Ψ≡1 ⇒ exponent 3α/2 + 1/2 ⇒ **α < 1/3**, again exactly the paper's "Ideal" column.

## Claim 3 — corruptions of possibly infinite magnitude, bounded only in frequency (Definition 2.1)
**Source:** Definition 2.1 (Sec 2.2); mechanism Definition 3.1 (Sec 3) + Lemma D.5 (Sec 4.1);
brittleness of the GP alternative Appendix H.
**Verdict: verified** (for the posterior-level mechanism; see Evidence boundary).

Exhaustive enumeration: 10 seeds × all 8 possible corrupted indices × 13 magnitudes
|c| ∈ {10⁰,…,10¹²} = **1040 configurations**. For each, deviation of the posterior mean from the
uncorrupted GP posterior mean, sup over a 101-point grid:

* RCGP (P-IMQ, g = 0, L = 1.96, c = 1): **sup deviation = 1.4212** over all 1040 configurations,
  and it *converges* as |c| → ∞ — maximum absolute drift between |c| = 10⁶ and |c| = 10¹² is
  **9.93e-07**. Normalised by σ_uc the sup is **1.5437** (finite, i.e. of the Lemma D.5 form
  C_w √T_c σ_uc);
* standard GP on the same data: **sup deviation = 4.983e+11**, and increasing |c| by 10⁶ increases
  the deviation by a factor of **≥ 999998** (exactly linear in c, as Appendix H predicts).

**Mutation test:** replacing the P-IMQ weight by the constant weight W (= removing the robustness
mechanism; formally J_w = I, m_w = 0) makes the deviation **4.536e+11** at |c| = 10¹² — unbounded,
exactly as predicted when the mechanism is disabled.

## Claim 4 — zero-cost robustness at T_c = 0 (Theorem 4.1)
**Source:** Theorem 4.1, Sec 4.3; Appendix F; mechanism Definition 3.1/Sec 3.
**Verdict: verified.**

(a) Symbolic: Ψ(0) = **1.0**, and both Theorem 4.2 and Theorem 4.3 bounds collapse at T_c = 0 to
`sqrt(T)*sqrt(betaprime)*sqrt(gamma)` = O(√(T β'_T γ_T)), the GP-UCB rate (both equalities exact).

(b) Exact numeric, uncorrupted Forrester benchmark (Sec 5.3 / Fig 3 setting; 5 initial points,
30 iterations, 10 seeds, σ_noise² = 1): with the plateau condition satisfied (L = 10) and T_c = 0,
FC-RCGP-UCB and A2-RCGP-UCB reproduce GP-UCB **bit-identically**:
max |query difference| over all seeds/iterations = **0.0** for both, max |cumulative-regret
difference| = **0.0** for both; mean cumulative regret **34.9909** for all three algorithms.

**Mutation test:** violating the plateau condition (L = 0.05) makes the FC trajectory diverge from
GP-UCB on **10/10 seeds**, with max |cumulative-regret difference| **34.7054** — i.e. the
equivalence is caused by the plateau, not by coincidence.

## Claim 5 — corrupted Forrester experiment: RCGP methods beat GP-UCB / Student-t / DiagnosticsGP
**Source:** Sec 5.3 and Figure 1 (+ configuration in Appendix I.4.2).
**Verdict: toy** — reproduced with a from-scratch CPU proxy of the paper's BoTorch pipeline
(fixed RBF hyperparameters instead of weighted LOO-CV fitting, grid argmax, own DiagnosticsGP).
Direction of the claim reproduces under the paper's recommended settings, but it is *not*
statistically resolved against GP-UCB and it flips under another configuration the paper endorses.

Setup: 1-D Forrester, σ_noise² = 1, 5 uncorrupted quasi-random initial points, 10 seeds,
greedy clairvoyant adversary (low = −10, high = 25), budget T_c = ⌈T^{1/3}⌉.
Two settings, because the main text and the appendix disagree on the adversary radii:
A = Sec 5.3 text (T = 100, near 0.2 / far 0.5, T_c = 5); B = Appendix I.4.2 (T = 30, near 0.1 /
far 0.4, T_c = 4). Mean cumulative regret over the 10 seeds (± s.e.):

| method | A (T=100) | B (T=30) |
|---|---|---|
| FC-RCGP-UCB (L = 1.96, T_c = 0 in β) | **219.27 ± 27.40** | **148.42 ± 8.10** |
| A2-RCGP-UCB (L = 1.96, T_c = 0 in β) | 224.62 ± 29.18 | 151.47 ± 8.51 |
| GP-UCB | 247.37 ± 33.34 | 157.59 ± 8.28 |
| Student-t Process UCB | 438.82 ± 17.58 | 191.62 ± 2.99 |
| DiagnosticsGP | 980.93 ± 60.80 | 255.02 ± 17.95 |
| FC-RCGP-UCB (L = 1.96, adaptive T_c estimate) | 793.61 ± 2.41 | 235.02 ± 3.26 |
| A2-RCGP-UCB (L = 1.96, adaptive T_c estimate) | 778.99 ± 3.82 | 215.22 ± 6.03 |

* Under the App I.1 configuration "c = 1, L = 1.96, T_c = 0 inside β_t" (both explicitly endorsed
  by the authors) the claimed ordering holds in **both** settings: RCGP < GP-UCB < Student-t <
  DiagnosticsGP. The advantage over GP-UCB is **11.4 %** (A) / **5.8 %** (B) of mean regret — well
  inside one standard error, so 10 seeds do not resolve it; the advantage over Student-t and
  DiagnosticsGP is large and unambiguous.
* Under the *other* configuration the paper endorses (T_c estimated as the number of points outside
  the plateau, which drives Ψ(T_c) and hence over-exploration), both RCGP methods become **worse
  than every baseline except DiagnosticsGP** (793.61 / 778.99 vs GP-UCB 247.37). The claim is
  therefore configuration-sensitive in this reimplementation.
* Discrepancy with the paper: Sec 5.3 states GP-UCB suffers the *highest* cumulative regret; here
  DiagnosticsGP is the worst baseline in both settings (980.93 / 255.02) — most likely because my
  DiagnosticsGP uses a MAD z-score filter rather than the paper's variational Student-t-likelihood
  GP detector.
* Uncorrupted control (Fig 3, T = 30, 10 seeds): A2 34.70, GP-UCB 34.99, FC 40.90, Student-t 42.29,
  DiagnosticsGP 233.43 → the Fig 3 statement (RCGP ≈ GP-UCB, better than Student-t/DiagnosticsGP)
  reproduces.

**Mutation test:** setting L = ∞ (plateau everywhere ⇒ the RCGP degenerates to a plain GP) makes
both RCGP curves land **exactly** on GP-UCB (247.37312941 in A, 157.58672285 in B, equal to
GP-UCB to <1e-9) and worsens their regret relative to the intact algorithms — so the improvement is
attributable to the P-IMQ down-weighting mechanism and to nothing else in the pipeline.

---

## Summary table

| claim | verdict | one-line evidence |
|---|---|---|
| 1 (Thm 4.2, α<1/4) | **verified** | symbolic exponent of the stated bound is 2α+1/2 ⇒ threshold exactly 1/4; all 8 Table-1 FC entries reproduce; mutation (Ψ≡1) gives the paper's ideal 1/2 |
| 2 (Thm 4.3, α<1/7) | **verified** | exponent 7α/2+1/2 ⇒ threshold exactly 1/7; all 8 Table-1 A2 entries reproduce; mutation (Ψ≡1) gives the paper's ideal 1/3 |
| 3 (unbounded magnitude, freq-only) | **verified** | over 1040 configurations RCGP posterior deviation ≤ 1.4212 and converges as \|c\|→10¹² (drift 9.9e-07) while the GP's grows linearly to 4.98e+11; mutation to constant weights restores divergence |
| 4 (T_c=0 zero-cost, Thm 4.1) | **verified** | Ψ(0)=1 collapses both bounds to √(β'Tγ) symbolically, and FC/A2 reproduce GP-UCB bit-identically on 10 seeds (query diff 0.0, regret 34.9909); mutation L=0.05 diverges on 10/10 seeds |
| 5 (corrupted Forrester ordering) | **toy** | own CPU proxy: FC 219.27 / A2 224.62 < GP-UCB 247.37 < Student-t 438.82 < DiagnosticsGP 980.93 (T=100), but the gap to GP-UCB is within 1 s.e. and the ordering flips under the paper's adaptive-T_c option (793.61) |

## Evidence boundary — what this evidence does NOT cover

1. **Claims 1–2 verify an implication, not a proof.** I checked that the *stated* bounds of
   Theorems 4.2/4.3 imply exactly the stated α-thresholds (and Table 1), with Ψ as defined in
   Sec 4.1. I did **not** verify the proofs in Appendix D (Lemmas D.1–D.31): the correctness of
   C_w scaling, of Lemma D.5, of Lemma D.11 (the Ψ bound), and of the plateau high-probability
   event are all taken as given. A flawed lemma would not be detected here.
2. **No empirical verification of the regret *rates*.** Nothing here measures R_T ~ T^{2α+1/2} on
   actual runs; the horizons used (T ≤ 100) are far too short to identify an asymptotic exponent.
3. **Claim 3 is a posterior-level, not regret-level, result.** I showed the RCGP posterior mean
   stays bounded under |c| → 10¹² with T_c = 1 corruption in a fixed 8-point design; I did not run
   the full BO loop with infinite-magnitude corruptions at the T_c = O(T^α) frequency, nor verify
   the constant C_w of Lemma D.5 numerically. Magnitudes above 10¹² were not tested (float64).
4. **Claim 4's exact equivalence is conditional on my fixed hyperparameters.** With per-iteration
   weighted-LOO-CV hyperparameter fitting (what the paper actually does) the RCGP and GP objectives
   differ, so bit-identity would not be expected; the theorem-level statement is asymptotic anyway.
5. **Claim 5 is a reimplementation, not the authors' code.** No BoTorch/GPyTorch, no LOO-CV
   hyperparameter optimisation, fixed RBF lengthscale 0.15 / outputscale 1, 201-point grid argmax,
   MAD-based DiagnosticsGP, Student-t process with ν = 5, 10 seeds. Absolute regret values are
   therefore **not** comparable to Figure 1; only orderings are. The paper reports no numeric
   regret values at all, so no digit-level anchor exists for this claim.
6. **Statistical power.** 10 seeds (the paper's own count) give standard errors of 8–33 on
   cumulative regrets of 150–250; the RCGP-vs-GP-UCB gap is not significant at that resolution.
   No paired test across seeds was performed beyond reporting per-seed values in `results/claim5.json`.
7. **Untested claims/experiments.** The CIFAR-10 hyperparameter-optimisation experiment (Sec 5.4,
   ≈8 h, GPU) and the Lunar-Lander-3 experiment (Sec 5.5, ≈8–12 h) were not attempted — they are
   outside the CPU/time budget and are not among the anchored claims.
8. **Adversary ambiguity.** The main text (near 0.2 / far 0.5, 100 iterations) and Appendix I.4.2
   (near 0.1 / far 0.4, 30 iterations) specify different experiments; both were run rather than
   guessing which produced Figure 1.
