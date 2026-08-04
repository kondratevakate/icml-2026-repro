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

---
<!-- trackio-cell
{"type": "markdown", "id": "cell_<built-in function hash>", "created_at": "2026-01-01T00:00:00+00:00", "title": "Claim 6 \u2014 Theorem 14 (benefit does not descend to subagents)"}\n-->
