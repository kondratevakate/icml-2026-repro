## Claim 1 — — graphops are self-adjoint, positivity-preserving, and unify dense + sparse
Ω = [0,1] discretised into n=400 uniform atoms; (Af)(x) = ∫W(x,y)f(y)dμ(y) → (1/n)·Wf.
Tested a dense graphon kernel `0.5(1+cos 2π(x−y))` and a sparse ~3-regular graph mapped
to the same space via W = n·A. Over 200 random probe pairs: ⟨Af,g⟩−⟨f,Ag⟩ ≤ 1.1e-16 and
no negative output for non-negative input, for **both**. That is the "single class of limit
objects" content of Def. 3.1, and it holds. Mutations target one property each and each
breaks only its own property (the signed-kernel mutation stays self-adjoint at 3e-18 —
a good sign the tests are independent).

---
<!-- trackio-cell
{"type": "markdown", "id": "cell_<built-in function hash>", "created_at": "2026-01-01T00:00:00+00:00", "title": "Claim 1 \u2014 \u2014 graphops are self-adjoint, positivity-preserving, and unify dense + sparse"}\n-->
