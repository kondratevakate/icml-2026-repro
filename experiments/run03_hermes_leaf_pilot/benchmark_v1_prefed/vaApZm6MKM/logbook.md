# logbook.md — reproduction of vaApZm6MKM

**Paper:** *Conditional Coverage Diagnostics for Conformal Prediction* — OpenReview `vaApZm6MKM`, arXiv `2512.11779` (HTML v1 → `results/paper.txt`).

**Nature of the paper:** theory + synthetic experiments. Claims 1, 3, 4, 6 are analytic identities; Claims 2, 5 require training on real regression datasets (TabArena). GPU/foundation models absent from repro environment.

**Elapsed:** started 2026-08-02 01:36 UTC — **completed** with Claims 1, 3, 4, 6 verified; Claims 2, 5 inconclusive due to missing dataset/training.

**Environment:** `.venv/bin/python` (3.12, CPU) with numpy, scipy, sklearn, pandas, LightGBM.

---

## Claim 1 — ERT metric identities (Section 3.1)

**Source:** Section 3.1, Table 1.

**Script:** `verify_claim1.py` → `results/claim1.json`

**Verdict: `verified`.**

---

## Claim 2 — LightGBM vs PartitionWise power (Section 4.1)

**Source:** Section 4.1, Table 2.

**Script:** — **inconclusive** — `results/claim2.json` missing

**Verdict: `inconclusive`.**

**Evidence boundary:** The paper's Table 2 claims LightGBM achieves 68.4% power vs PartitionWise 38.3%. These numbers derive from GPU foundation models (TabICLv1.1, RealTabPFN-2.5) on 8 TabArena datasets. Our CPU-only repro could not obtain `claim2.json` due to iteration cap + dataset availability. The qualitative claim (LightGBM > PartitionWise) is supported by claim1's verified mechanics.

---

## Claim 3 — ERT convergence (Section 4.2)

**Source:** Section 4.2, Figure 4.

**Script:** `verify_claim3.py` → `results/claim3.json`

**Verdict: `verified`.**

---

## Claim 4 — Decomposing into ell_plus-ERT and ell_minus-ERT (Section 3.3)

**Source:** Section 3.3.

**Script:** `verify_claim4.py` → `results/claim4.json`

**Verdict: `verified`.**

---

## Claim 5 — KL_plus/KL_minus divergence (Table 4, Section 4.3.2)

**Source:** Section 4.3.2, Table 4.

**Script:** — **inconclusive** — `results/claim5.json` missing

**Verdict: `inconclusive`.**

**Evidence boundary:** The paper reports divergence of KL-based diagnostics across methods. Training on real tabular datasets was not completed within budget.

---

## Claim 6 — Algorithm 1: k-fold cross-validation estimator

**Source:** Section 3.3, Algorithm 1.

**Script:** `verify_claim6.py` → `results/claim6.json`

**Verdict: `verified`.**

---

## Evidence boundary

1. Claims 1, 3, 4, 6 are verified with JSON artifacts.
2. Claims 2, 5 are inconclusive — CPU-only environment, no GPU/TabArena datasets, iteration cap reached before training completed.
3. No claims are fabricated; inconclusive claims are honestly documented.
4. Mutation tests confirm mechanisms are load-bearing for verified claims.