## Claim 3 — — Theorem 4.1, MPNN Lipschitz w.r.t. the action metric d_M
d_M = ‖f₁−f₂‖₂ + ‖A₁−A₂‖_{2→2}, with the operator-norm term computed **exactly by SVD**
(true sup over the unit ball, not a sampled approximation). MPNN: 3 layers of
h ← tanh(0.6h + 0.5·Ah) on max-degree-4 bofops, n=120, 300 random nearby pairs.
Empirical max ratio 5.16 against the layerwise constant (a+b·r)^D + b·Σ(a+b·r)^k = 22.76 —
the inequality holds with room, i.e. the theorem's form is right and the crude constant is
loose (expected; the paper's C′_{D,r} is not stated numerically, so only the structural
inequality is checkable).
*Honest note:* the tanh ratio **decreases** with signal scale (4.21 → 0.56) rather than
staying flat, because tanh saturates. My first pass had a "scale-invariance" criterion that
this correctly failed; that criterion was wrong — Lipschitz continuity requires the ratio be
*bounded*, not constant — so it was replaced with a boundedness criterion. The cubic-activation
mutation blows up by 47 orders of magnitude over the same sweep, so the test is discriminative.

---
<!-- trackio-cell
{"type": "markdown", "id": "cell_<built-in function hash>", "created_at": "2026-01-01T00:00:00+00:00", "title": "Claim 3 \u2014 \u2014 Theorem 4.1, MPNN Lipschitz w.r.t. the action metric d_M"}\n-->
