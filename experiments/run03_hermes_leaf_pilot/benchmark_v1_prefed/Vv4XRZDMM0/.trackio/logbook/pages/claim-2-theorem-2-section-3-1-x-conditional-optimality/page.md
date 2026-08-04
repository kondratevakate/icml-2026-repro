## Claim 2 — Theorem 2 (Section 3.1), X-conditional optimality

**Statement tested.** For fixed x the program (4) has optimum
`C*(x) = {y : h_{λ*}(x,y) > 1} ∪ S(x)`, `h_λ = Σ_k λ_k(x) f_k(y|x)`, with complementary slackness
(i) `λ_k*>0 ⇒` exact 1 − α, (ii) `λ_k*=0 ⇒ ≥ 1 − α`, (iii) some `λ_{k*}* > 0`.

**Script.** `verify_claim2.py` → `results/claim2.json` (runtime ≈ 81 s). With μ = counting measure on a
finite Y-grid, (4) is an LP; solved with `scipy.optimize.linprog` (HiGHS) and its exact duals.
**Exhaustive enumeration of the design axes**: K ∈ {2,3,4,5} × m ∈ {3,…,10} × α ∈ {0.05, 0.1, 0.2}
× 30 random density draws = **2880 instances**.

**Numbers.**
- Superlevel-set solution reproduces the LP optimum in **2880/2880** instances, max size gap **3.1e-14**.
- Strong duality gap (primal − dual (5)) ≤ **4.0e-14**.
- Complementary slackness (i)/(ii)/(iii): **2880 / 2880 / 2880**.
- "Exact 1 − α coverage for at least one source": **2880/2880**.

**Mutation tests.**
| mutation | result |
|---|---|
| λ* → uniform λ with the same ℓ1 mass | superlevel set infeasible in **2680/2880**, infeasible *or* strictly larger in **2880/2880** |
| invert the superlevel condition (`{h_{λ*} < 1}`) | infeasible in **2880/2880** |
| drop the largest active constraint | LP optimum strictly shrinks in **2847/2880** (the remaining 33 are ties where another source binds identically) — i.e. active constraints really are the binding ones |

**Verdict: verified** (to solver precision, on the discrete-μ instantiation of the theorem).

---

---
<!-- trackio-cell
{"type": "markdown", "id": "cell_<built-in function hash>", "created_at": "2026-01-01T00:00:00+00:00", "title": "Claim 2 \u2014 Theorem 2 (Section 3.1), X-conditional optimality"}\n-->
