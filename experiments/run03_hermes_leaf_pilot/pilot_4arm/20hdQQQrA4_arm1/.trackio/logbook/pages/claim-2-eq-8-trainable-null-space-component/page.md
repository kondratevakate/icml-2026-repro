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

---
<!-- trackio-cell
{"type": "markdown", "id": "cell_<built-in function hash>", "created_at": "2026-01-01T00:00:00+00:00", "title": "Claim 2 \u2014 Eq. 8 trainable null-space component"}\n-->
