## Claim 1 — Theorem 4.1, diagonal rescaling → mass-weighted self-adjointness — **verified**

Script `verify_claim1.py`, results `results/claim1.json`.

Construction: given symmetric positive `W` and a prescribed positive mass vector `m`,
iterate `d ← sqrt(d·m /(W d))` until `diag(d)W diag(d)1 = m`; set `S = diag(d)W diag(d)`,
`T = diag(m)^{-1} S`.

- Gaussian 2D, n=120: Sinkhorn iters=42; row-sum err=8.7e-13; `<Tx,y>_m − <x,Ty>_m` (rel)=2.2e-16; `T = diag(a)W diag(b)` residual=8.0e-17
- exponential 3D, n=100: Sinkhorn iters=42; row-sum err=5.2e-13; `<Tx,y>_m − <x,Ty>_m` (rel)=0.0; `T = diag(a)W diag(b)` residual=9.8e-17
- Gaussian 5D, n=80: Sinkhorn iters=42; row-sum err=5.1e-13; `<Tx,y>_m − <x,Ty>_m` (rel)=0.0; `T = diag(a)W diag(b)` residual=1.3e-16


(Iteration counts here are to machine tolerance 1e-12, not the 0.1% target of Claim 3.)

**Mutations.** (M1) breaking kernel symmetry: m-self-adjointness error rises to
1.3e-2 / 1.3e-3 / 2.8e-2. (M2) replacing the Sinkhorn diagonal by row normalization:
error 3.2e-1 / 6.1e-2 / 1.7e-1. Both break the property → the test is discriminating.

---
<!-- trackio-cell
{"type": "markdown", "id": "cell_<built-in function hash>", "created_at": "2026-01-01T00:00:00+00:00", "title": "Claim 1 \u2014 Theorem 4.1, diagonal rescaling \u2192 mass-weighted self-adjointness \u2014 **verified**"}\n-->
