## Claim 1 — graphops are self-adjoint, positivity-preserving (Definition 3.1)

Script `verify_claim1.py` → `results/claim1.json`. Verdict: **verified**. Mutation test: yes.

`n = 300` atoms. Two instances audited with the *same* independent probes (200 random pairs each):

| instance | self-adjointness error `|⟨Af,g⟩−⟨f,Ag⟩|` | positivity violation |
|---|---|---|
| dense graphon `0.5(1+cos 2π(x−y))` | ≈ 1e-17 | 0.0 |
| sparse graph as graphop `W = n·Adj` (4-regular) | 2.22e-16 | 0.0 |

Both pass ⇒ they are members of the **same** operator class, which is precisely the "unifying limit
object" content of Definition 3.1. Note `op_norm_inf_to_1 = 4.0` for the sparse instance — the
kernel blows up like `n` but the operator stays bounded, as the framework requires.

**Mutation.** An antisymmetric signed kernel breaks *both* properties (self-adjointness error
1.24e-2, positivity violation 6.9e-2). A *symmetric* signed kernel breaks positivity only
(violation 7.5e-2) while self-adjointness stays at 3.5e-18 — confirming the two probes are
**independent** and not cross-contaminating.

---
<!-- trackio-cell
{"type": "markdown", "id": "cell_<built-in function hash>", "created_at": "2026-01-01T00:00:00+00:00", "title": "Claim 1 \u2014 graphops are self-adjoint, positivity-preserving (Definition 3.1)"}\n-->
