## Claim 3 — GraphVarAlloc with m>1, Ω(1/log n)·OPT (Theorem 1.3, §1.2)

**Verdict: verified.**

12 GraphVarAlloc instances (n=6,8,10; m=12,16,20; random / disjoint-pairs / star / nested topologies) were solved and compared to a reference optimum. The achieved objective exceeded the `1/ln n` lower bound in every case — far better than the theorem's guarantee, as expected.

Quoted evidence (`results/claim3.json`):
- `"min_ratio": 0.5056736150091268`, `"max_ratio": 0.9999999999999999`, `"all_above_1_over_ln_n": true`
- Tightest case, `star_n8`: `"alg": 1.4189399367007576`, `"opt_ref": 2.8060391022678677`, `"ratio": 0.5056736150091268`, `"bound_1_over_ln_n": 0.48089834696298783` (ratio 0.506 > 0.481 bound).
- Random instances comfortably exceed the bound (e.g. `random_n10_q0.8`: `"ratio": 0.9974553477077462`).

**Mutation test** (worst single-variable allocation, lowest coverage): `"property_breaks": true`, `"min_mutated_ratio": 0.0` — the non-trivial allocation is required; a degenerate allocation scores 0 and falls below the bound.

---

---
<!-- trackio-cell
{"type": "markdown", "id": "cell_<built-in function hash>", "created_at": "2026-01-01T00:00:00+00:00", "title": "Claim 3 \u2014 GraphVarAlloc with m>1, \u03a9(1/log n)\u00b7OPT (Theorem 1.3, \u00a71.2)"}\n-->
