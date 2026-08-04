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

---
<!-- trackio-cell
{"type": "markdown", "id": "cell_<built-in function hash>", "created_at": "2026-01-01T00:00:00+00:00", "title": "Claim 3 \u2014 no full-row-rank requirement, arbitrary cardinality"}\n-->
