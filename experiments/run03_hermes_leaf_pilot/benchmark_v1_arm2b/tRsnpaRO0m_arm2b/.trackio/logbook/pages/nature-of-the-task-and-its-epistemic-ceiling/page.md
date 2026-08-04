## Nature of the task and its epistemic ceiling

All six anchored claims are **mathematical** — two definitions, one theorem, one corollary and two
derived results about operators on probability spaces. There is **no released code, no dataset and
no GPU requirement**, and — critically — **the paper states no numeric value to match for any
anchored claim**. "Reproduction" therefore means *independent numerical verification of the
mathematical content* by **faithful finite-dimensional discretisation**: Ω is modelled as `n`
uniform atoms of mass `1/n`, so `(Af)(x)=∫W(x,y)f(y)dμ` becomes `(W @ f)/n`, a sparse graph enters
the same class as `W = n·Adj`, the fiber measure `ν_x(Ω)` becomes vertex degree, DIDMs become joint
(degree, neighbour-degree) histograms and the mover's distance is computed as an **exact W₁ linear
program** (scipy HiGHS, ℓ¹ ground metric — no entropic approximation).

Ceiling, stated per verdict: claims 1–3 are **conclusive at the level of the finite model** (they
are definitional/numeric identities checked to machine precision or against an explicit constant);
claims 4–6 are **evidence, not proof** — a finite ε-net is a surrogate for a topological statement,
and a finite-`m` rate fit is a surrogate for an asymptotic bound. Every verified claim carries a
mutation test that removes the relevant hypothesis and must break the property.

---
<!-- trackio-cell
{"type": "markdown", "id": "cell_<built-in function hash>", "created_at": "2026-01-01T00:00:00+00:00", "title": "Nature of the task and its epistemic ceiling"}\n-->
