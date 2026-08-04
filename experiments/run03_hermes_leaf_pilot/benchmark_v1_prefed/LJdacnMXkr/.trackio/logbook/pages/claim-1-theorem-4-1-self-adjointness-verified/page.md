## Claim 1 — Theorem 4.1 (self-adjointness) — **verified**
`verify_claim1.py` → `results/claim1.json`

6 configurations (Gaussian & exponential kernels, n = 80/150/200, d = 2/3, non-uniform
masses `m ~ U(0.5,1.5)`).

- max `‖MP − (MP)^T‖_∞` = **2.8e-17**; random-vector check `|⟨Pu,v⟩_m − ⟨u,Pv⟩_m| / |⟨Pu,v⟩_m|` ≈ 2e-16
- max row-sum error = 6.9e-13; min entry > 0 (3.1e-14 worst case, strictly positive)
- λ ∈ [4.5e-4, 1.0] in every case

**Mutation.** (a) Row-normalize instead of Sinkhorn → self-adjointness residual **0.046** (broken)
and λ_min = −0.023. (b) Multiplicatively perturb `K` into an asymmetric kernel → residual **0.036**.
The property is specific to the symmetric Sinkhorn scaling, as claimed.

---
<!-- trackio-cell
{"type": "markdown", "id": "cell_<built-in function hash>", "created_at": "2026-01-01T00:00:00+00:00", "title": "Claim 1 \u2014 Theorem 4.1 (self-adjointness) \u2014 **verified**"}\n-->
