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

---
<!-- trackio-cell
{"type": "markdown", "id": "cell_<built-in function hash>", "created_at": "2026-01-01T00:00:00+00:00", "title": "Claim 5 \u2014 Theorem 21 (Waluigi shattering)"}\n-->
