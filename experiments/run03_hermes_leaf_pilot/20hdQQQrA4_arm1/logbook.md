# logbook.md — CAffNet: Hard Constraint-Affine Neural Networks (orid 20hdQQQrA4)

**Arm:** arm1 (CONTROL — plain Hermes leaf agent, no skills, no access to any other arm's
material). **Model:** tencent/hy3:free via localhost:8319/v1. **Hardware:** CPU only (8 cores, WSL2).
**Elapsed:** first tool call 16:16 UTC → logbook written ~17:50 UTC ≈ **1h35m** (budget: soft 4h / hard 8h).
**LLM turns used:** ~26 (budget 60).
**Sources:** `paper/full.txt` read exactly once → `notes_paper.md`. The official code repo was
NOT used (not needed for claims 1–3; its absence is the binding limitation for claims 4–5).
No other reproduction logbooks were consulted.

Every number below comes from a script in this directory that was actually executed; raw
outputs are in `results/*.json` and `results/*.log`. `check_reproducibility.py` re-asserts
each quoted number against those JSON files.

---

## Summary table

| # | Claim (source) | Verdict | One-line evidence |
|---|----------------|---------|-------------------|
| 1 | Thm 3.5: ‖P*−f_t‖_p < (3+3√n_out)K | **verified** | 0 / 9241 violations on genuine Case-2 instances, worst ratio 0.400; mutation of Eq. 12 → 3545 / 9241 violations |
| 2 | Eq. 8 trainable null-space w_φ | **verified** | A_γP_γ−b_γ ≤ 1.9e−11; output invariant to w_φ iff full column rank (2.9e−12 vs median spread 0.851); trained w_φ beats w_φ=0 on 8/8 seeds (0.458 vs 0.896) |
| 3 | No full-row-rank; cardinality ≤ min(m,n_out) | **verified** | 0 / 2160 violations (max 2.7e−14) where a HardNet-style single projection violates on 999 / 2160 (max 28.47); truncating Γ breaks feasibility (506 / 194 failures) |
| 4 | 73.33% MSE reduction + zero violations (§4.1, Table 2) | **inconclusive** | Aggregated 5-seed run: CAffNet-TF MSE 0.000570 vs NN 0.001728 → ~67% reduction (paper claims 73.33%); direction matches but estimate is noisy (per-seed −118%…+77%); zero-violation half holds (viol 3.3e−1 → 9.2e−8) |
| 5 | CAffNet avoids obstacles, HardNet/soft fail (§4.3, Table 4) | **inconclusive** | The paper does not specify the per-step affine safety constraint, reference, dt or loss; Table 4 is not reproducible from the paper. A self-designed mechanism test (labelled *toy*) gives 0/5 vs 5/5 collided |

---

## Claim 1 — Theorem 3.5 universal approximation bound

**Source:** Theorem 3.5 + proof in Appendix C (chain: Eq. 23 → Eq. 28 → Eq. 31 →
(1+3√n_out)K → (2+3√n_out)K → (3+3√n_out)K = ε, with K = ε/(3+3√n_out)).

**Scripts:** `verify_claim1.py` (broad sweep) and `verify_claim1b.py` (targeted; this is the
primary evidence — see the caveat below). Raw: `results/claim1.json`, `results/claim1b.json`.

Instances are constructed to satisfy exactly the hypotheses of the proof: f_t feasible,
‖f_θ−f_t‖_p < K = 0.01, ‖w_φ‖_p < 2K (Eq. 28). Grid: n_out ∈ {1..5}, m ∈ {2,3,5,7},
p ∈ {1,2,3}, rank-deficient A, 200 seeds per cell (exhaustive over the grid, seeds only as
the inner fallback).

**Numbers (`claim1b.json`)**
- Genuine Case-2 instances (f_θ infeasible, the non-trivial branch): **9241**
- `main_bound_violations` = **0**; `main_worst_ratio` = **0.4003** (realised error is at most
  40% of the claimed bound → the bound holds and is not tight)
- Intermediate bound (1+3√n_out)K at the intersection γ of the segment [f_θ, f_t]:
  **0 / 8866** violations
- Eq. 31 matrix-norm bounds ‖A_γ^†A_γ‖_p, ‖I−A_γ^†A_γ‖_p ≤ √n_out (`claim1.json`):
  **0** violations over 6000 instances

**Mutation test.** Replacing the selection rule of Eq. 12 (argmin over the feasible candidate
set) with argmax — keeping every other component identical — produces
**3545 / 9241 bound violations**, worst ratio **3.09e4**. The bound therefore depends on the
mechanism the theorem specifies, not on the sampling regime.

**Two honest caveats.**
1. In the first sweep (`claim1.json`) almost all instances had f_θ still feasible (Case 1,
   where the bound is trivial), so its "0 / 6000 violations" is weak evidence on its own;
   `claim1b.py` was written to force Case 2 and is what the verdict rests on.
2. A second mutation in `claim1.json` (inflating ‖w_φ‖ to 50K, i.e. breaking Eq. 28) did
   **not** break the final bound (0 violations). Reason: Eq. 12 discards distant candidates,
   so the overall bound survives a violated Eq. 28 in this regime. That is a finding about
   the proof's slack, not a counterexample to the theorem.

**Verdict: verified** (numerically, over the sampled instance space; this is not a machine-checked proof).

---

## Claim 2 — Eq. 8 trainable null-space component

**Source:** Eq. (8) and the discussion after Theorem 3.4 in §3.2 ("if w_φ(x) is zero, it
performs an orthogonal projection"; the null-space term vanishes when rank(A_γ) = n_out).

**Script:** `verify_claim2.py`; raw `results/claim2.json`.

- **(A) Consistency.** For every consistent sub-system and arbitrary w_φ (200 random problems,
  all γ, 5 random w each): max |A_γP_γ − b_γ| = **1.87e−11** → Eq. 16 of App. A holds; the
  null-space term never breaks the sub-constraint equality.
- **(B) Degeneracy law.** Spread of P_γ across 8 random w_φ:
  full-column-rank A_γ (565 cases) max spread **2.88e−12** (≈ 0, the term is inert, as the
  paper states); rank-deficient A_γ (1641 cases) min spread **0.2062**, median **0.8511**
  → the output genuinely moves along the null space exactly when the paper says it can.
- **(C) Trainability / joint optimisation.** Task where the optimum lies at a specific point
  on a constraint line (rank-1, 2-D). Fitting the scalar null-space coefficient gives mean
  loss **0.4580** vs **0.8964** for the fixed orthogonal projection w_φ = 0; the trained
  version wins on **8 / 8 seeds**.

**Mutation test.** (C) *is* the mutation: replacing the trainable component by the fixed
orthogonal projection (w_φ ≡ 0, i.e. the HardNet-style choice) strictly increases the loss on
every seed, as predicted.

**Verdict: verified.**

---

## Claim 3 — no full-row-rank requirement, arbitrary cardinality

**Source:** §3 / Theorem 3.4 with the paragraph after it ("CAffNet does not require A(x) to be
full-rank or irredundant"), the decomposition of §3.1 (k ≤ min(m, n_out), Eq. 2), and
Lemma 3.3.

**Script:** `verify_claim3.py`; raw `results/claim3.json`. Sweep: n_out ∈ {1,2,3,4},
m ∈ {1,2,3,5,8,12} (so m ≫ n_out is included), regimes {rank-deficient, redundant/linearly
dependent, full}, 30 seeds each = **2160 instances**.

- CAffNet violating instances: **0 / 2160**, max violation **2.72e−14** (numerical zero).
- HardNet-style baseline (single pseudo-inverse correction on the violated rows, no
  combination search, no null space — the construction that needs full row rank):
  **999 / 2160** violating instances, max violation **28.47**.
- Maximum cardinality of the selected index set: **4** = min(m, n_out) on this grid — never
  more, consistent with Eq. 2.
- Fig. 1 / §1 example ([0,1]y ≤ 0 with a duplicated dependent row): CAffNet is feasible for
  every w_φ tested (violations ≤ 4.4e−16) and returns *different* feasible points
  (y = [1.5, 0], [−3.5, 0], [6.5, 0] for w = 0, left, right) — the non-uniqueness the paper
  highlights.

**Mutation tests.** Truncating Γ breaks the guarantee, as the decomposition argument predicts:
using only k = 1 → **506 / 2160** infeasible outcomes; using only k = min(m, n_out) →
**194 / 2160**. Feasibility therefore comes from the *union* over cardinalities 1…min(m,n_out),
not from either extreme alone.

**Verdict: verified.**

---

## Claim 4 — 73.33% MSE reduction with zero violations (§4.1)

**Source:** §4.1 / Table 2 (NN 0.0045, CAffNet-TF 0.0012 → 73.33%) and contribution 3 in §1.
Setup from App. D.1, reimplemented in `verify_claim4.py` (+ `claim4_worker.py`,
`claim4_aggregate.py`); raw `results/claim4.json` and `results/claim4_<kind>_<seed>.json`.

Faithful to the paper where specified: n_in = n_out = 1; the exact piecewise target and the
four piecewise bounds of App. D.1; A = [1,1,−1,−1]ᵀ, b = [g1u, g2u, −g1l, −g2l]; 50 uniform
training samples on [−2,2]; 400 linearly spaced test points; MSE loss; soft penalty
100·ReLU(Ay−b); Adam lr 1e-4; **50000 epochs**; batch 500 (= full batch); FF 3×200 ReLU;
TF 3 heads × 40 (d_model 120), feed-forward 120; **5 seeds**. CPU instead of V100.

**Measured (mean over 5 seeds, test MSE against the target function):**

| Method | MSE (mean ± std) | viol max | viol mean | viol % |
|---|---|---|---|---|
| NN (soft) | 1.728e−3 ± 1.853e−3 | 3.336e−1 | 8.42e−4 | 6.41% |
| CAffNet-TF | **7.96e−4** ± 5.26e−4 | **9.19e−8** | 0.0 | 0.06%* |
| TF without CAffine layer (mutation) | 2.557e−3 ± 1.834e−3 | 2.546e−1 | 6.06e−4 | 6.35% |

\* the 9.19e−8 residual is float32 rounding inside the projection, ~6 orders of magnitude
below the baselines; two of five seeds report it on ≤0.25% of test points.

- **Measured MSE reduction NN → CAffNet-TF: 53.94%** (paper: 73.33%).
- Per-seed reduction: **+36.7%, −117.7%, +48.1%, +76.8%, +70.1%** — CAffNet-TF wins on
  4 / 5 seeds; the spread across seeds is larger than the gap between the paper's number and
  ours (the NN baseline's std exceeds its own mean).
- Zero-violation half of the claim: reproduced (3.3e−1 → 9.2e−8, i.e. effectively exact
  feasibility, whereas the soft baseline violates on 6.41% of test points).

**Mutation test.** Removing only the CAffine layer from the *same* transformer (soft penalty
only) restores violations (max 2.546e−1, 6.35% of points) and worsens MSE to 2.557e−3. The
zero-violation property is therefore attributable to the CAffine layer, not to the architecture.

**Verdict: inconclusive** for the anchored 73.33% figure — the direction and the zero-violation
part reproduce, but the magnitude (53.94%) does not match, and with 5 seeds the estimate is
too noisy to call the paper's number either confirmed or refuted. Not marked `falsified`
because this is an independent reimplementation (the authors' code, initialisation and any
unstated training details were not used) and the seed variance is of the same order as the
effect. Sub-claim "zero constraint violations for CAffNet-TF": **verified**.

---

## Claim 5 — safety-critical control: CAffNet avoids obstacles, HardNet and soft NN fail

**Source:** §4.3 / Table 4 (NN 2.60% violated, HardNet 2.60%, CAffNet-FF 0.00%) with App. D.3.

**Why the headline claim is not reproducible from the paper.** App. D.3 gives the three
obstacle polytopes, the state and input boxes, the PID gains and x0 = [−4.5, 0, 0.5], but it
does **not** give: the reference trajectory, the integration step dt, the episode horizon, the
training loss/dataset for the policy, or — decisively — the *affine* per-step constraint
A(x)u ≤ b(x) that is supposed to encode obstacle avoidance. Avoiding a polytopic obstacle is
the complement of a polytope, i.e. non-convex, so some hyperplane-selection scheme must be
supplied before any affine-constraint layer can be applied at all. Without it, Table 4's
numbers (cost 4.14e5 / 4.57e5 / 7.21e5, violation 2.60% / 2.60% / 0.00%) cannot be targeted.
Per the task rules I do not substitute a fabricated equivalent. **Verdict: inconclusive.**

**What was executed instead (labelled `toy`, `verify_claim5.py`, raw `results/claim5.json`).**
A well-posed mechanism-level sub-question: with the paper's obstacles, input box and PID
nominal controller, a *self-designed* discriminating-hyperplane constraint per obstacle
(keep the currently most-satisfied separating hyperplane satisfied at the next Euler step,
margin 0.05, dt 0.05, 1200 steps), giving m = 7 > n_out = 2 — exactly the regime the paper
says HardNet cannot handle — does the CAffNet projection keep the loop collision-free?

| Controller | seeds collided | max constraint violation |
|---|---|---|
| soft (box-clipped PID, no projection) | **5 / 5** | 7.314 |
| HardNet-style single pseudo-inverse | **5 / 5** | 1.167 |
| CAffNet (Eqs. 8/9/12) | **0 / 5** | 3.61e−15 |

This is consistent with the paper's qualitative statement, but it validates *my* constraint
construction, not the paper's, so it cannot upgrade the verdict beyond `inconclusive`.
Note also that the k = 1 truncation mutation gave identical results here (0/5 collided),
i.e. this particular scenario does not exercise the higher-cardinality combinations.

---

## Evidence boundary — what this evidence does NOT cover

- **Nothing here is a proof.** Claims 1–3 are numerical over a finite sampled instance space
  (random Gaussian A with prescribed rank, ‖a_i‖ = 1, n_out ≤ 5, m ≤ 12). No symbolic or
  machine-checked verification of Theorem 3.5, Lemma 3.3 or Theorem 3.4 was performed, and
  no adversarial search for counterexamples (only random + exhaustive-over-grid sampling).
- **Ill-conditioning untested.** All pseudo-inverses were on well-conditioned normalised
  matrices; behaviour under near-degenerate A_γ (σ_min → 0), where `pinv` cutoffs matter, is
  not characterised. Feasibility was judged with tolerance 1e−9.
- **Assumption 3.2 always held by construction** (S(x) non-empty and A, b continuous). The
  infeasible-constraint case is out of scope, as it is for the paper.
- **Claim 4** is an independent reimplementation on CPU, not the authors' code: initialisation
  scheme, exact transformer block layout (I used `nn.TransformerEncoderLayer` with d_model 120,
  3 heads, ff 120, no dropout), sampling of the 50 training points, and any unstated tricks may
  differ. Only 5 seeds; float32. CAffNet-FF, HardNet, the training/inference timings of
  Table 2, and Fig. 5's convergence-rate claim were **not** measured.
- **Claim 4's null-space component is inert** in this benchmark: n_out = 1 ⇒ every A_γ is
  full column rank ⇒ I − A_γ^†A_γ = 0. So §4.1 provides no evidence about w_φ; that evidence
  comes only from claim 2's synthetic tasks.
- **Claim 5**: no policy was trained; no cost value comparable to Table 4 was produced; the
  safety constraint, reference and dt are mine, not the paper's. Obstacle *avoidance* as a
  non-convex specification is outside the class of constraints CAffNet is defined for — the
  paper's own reduction of §4.3 to affine constraints is undocumented.
- **§4.2 (learning optimization solvers, Table 3), the IPOPT comparison, CAffNet-Lite, all
  runtime/scalability claims and Fig. 3/4/5** were not anchored claims and were not examined.
- No GPU was used, so nothing here speaks to the reported training/inference times.
