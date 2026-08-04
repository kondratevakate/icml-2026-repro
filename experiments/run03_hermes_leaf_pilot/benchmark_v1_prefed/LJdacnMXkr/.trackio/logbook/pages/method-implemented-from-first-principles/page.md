## Method (implemented from first principles)

Symmetric Sinkhorn scaling (Thm 4.1): given a symmetric, entrywise-positive kernel `K`
and masses `m`, find `d>0` with `S = diag(d) K diag(d)` symmetric and `S1 = m`; iterate
`d ← sqrt(d·m / (K d))`. Then `P = diag(m)^{-1} S` satisfies `P1 = 1` (mass conservation),
`P > 0`, and `M P = S = S^T`, i.e. `P` is self-adjoint w.r.t. `⟨u,v⟩_m = Σ mᵢuᵢvᵢ`.
Spectra are computed on the symmetrized conjugate `M^{1/2} P M^{-1/2}`.

---
<!-- trackio-cell
{"type": "markdown", "id": "cell_<built-in function hash>", "created_at": "2026-01-01T00:00:00+00:00", "title": "Method (implemented from first principles)"}\n-->
