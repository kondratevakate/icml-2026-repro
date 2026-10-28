# Claim 2: Theorem 2 dual optimality


---
<!-- trackio-cell
{"type": "markdown", "id": "cell_mdrcp_v2_c2", "created_at": "2026-07-29T08:10:00+00:00", "title": "Claim 2: Theorem 2 dual optimality"}
-->
**Anchored claim (verbatim).** "Theorem 2 (Section 3.1) characterizes the
population-optimal prediction set as a superlevel set of a shared score
function `h_lambda*(x,y) = sum_k lambda_k*(x) f_k(y|x)`, with complementary
slackness ensuring exact `1-alpha` coverage for at least one source."

**Verdict: VERIFIED.**

The audit independently solves the randomized prediction-set relaxation in
Eq. (4) with `scipy.optimize.linprog`; it does not call the paper's optimizer
or another participant's code. For each solution it extracts the LP dual
multipliers and checks the paper's dual objective in Eq. (5), the KKT
superlevel structure, and complementary slackness.

Scope: 500 seeded finite-measure problems, each with 3 source distributions,
24 outcomes, and `alpha=0.1`. Finite outcome spaces with counting measure are
explicitly covered by Section 3.1.

| Check | Result |
| --- | ---: |
| Primal/dual instances | 500 |
| Maximum primal-dual gap | `2.23e-8` |
| Maximum complementary-slackness error | `2.23e-8` |
| Maximum superlevel-structure violation | `1.26e-8` |
| Instances with at least one active source | `500/500` |
| Uniform-weight destructive control strictly suboptimal | `500/500` |

The optimized shared score includes every outcome with `h_lambda>1`, excludes
every outcome with `h_lambda<1`, and randomizes only on the `h_lambda=1`
boundary. Each instance has 2-3 boundary outcomes, which directly exercises
the optional boundary set `S(x)` in the theorem instead of silently assuming
the optimum is deterministic.

Rerun:

```powershell
cd repro_mdrcp_v2
python audit_theorem2.py --output results/theorem2.json
```
