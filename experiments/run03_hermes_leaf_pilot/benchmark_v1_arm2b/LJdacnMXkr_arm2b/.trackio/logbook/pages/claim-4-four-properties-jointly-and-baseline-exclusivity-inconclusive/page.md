## Claim 4 — four properties jointly, and baseline exclusivity — **inconclusive**

Script `verify_claim4.py`, results `results/claim4.json`.

For a *prescribed* non-uniform mass measure `m` (n=150, Gaussian 2D/3D, exponential 2D):

- Sinkhorn operator: **4/4** in every case (sym. rel. err ≤ ~1e-16, mass err ≤ 1e-12,
  entrywise positive, spectrum in [0,1]).
- Row normalization `D^{-1}W` scored against the prescribed `m`: **3/4** (not
  m-self-adjoint, rel. asymmetry 0.24).
- Symmetric normalization `D^{-1/2}WD^{-1/2}`: **2/4** (mass error 0.42, not
  m-self-adjoint).

**Why not "verified":** the primary content holds, but the exclusivity sub-clause does
not survive as literally stated. Row normalization evaluated against **its own degree
measure** satisfies all four properties simultaneously (sym. err 2.2e-16, mass err
4.4e-16, positive, spectrum [5.4e-4, 1.0]) — a genuine counter-example recorded in the
results file. The Sinkhorn advantage is that it attains the four properties for an
*arbitrary prescribed* mass measure, which is a weaker statement than the claim.

**Mutation.** An entrywise-positive but indefinite symmetric kernel
(`1 + 0.9 cos(π|x−y|²)`, min eigenvalue −13.1) yields a Sinkhorn operator with
eigenvalue −0.110 → spectral damping fails. Positive-definiteness of the kernel, not
just positivity of entries, is required for the [0,1] spectrum.

---
<!-- trackio-cell
{"type": "markdown", "id": "cell_<built-in function hash>", "created_at": "2026-01-01T00:00:00+00:00", "title": "Claim 4 \u2014 four properties jointly, and baseline exclusivity \u2014 **inconclusive**"}\n-->
