## Evidence boundary

What this evidence does **not** cover:

1. **No trained CAffNet-TF.** Claims 1–3 and 5 use the layer/operator directly; claim 4 uses
   a 1-32-32-2 numpy MLP. The paper's transformer variant is untested here.
2. **No GPU, no paper-scale training.** 1500 Adam steps vs the paper's GPU-scale schedule.
   Every reported magnitude (the 9.22% reduction, the control costs) is a reduced-scale
   surrogate, not a reproduction of the tables.
3. **No paper data.** The piecewise benchmark and the control task are independent
   reimplementations from the claim text; the paper's own datasets are not released in the
   claim bundle. A different benchmark instantiation could move every magnitude.
4. **No official code executed**, so the reimplementation has no cross-check against the
   authors' implementation. Section 3's equations are the only source.
5. **The `w_scales` safeguard.** When no feasible candidate exists at full null-space scale
   the implementation shrinks `w` (1 → 0.5 → 0.25 → 0). With `w = 0` the enumeration is
   exactly the active-set enumeration of the Euclidean projection, so feasibility is
   guaranteed whenever the polyhedron is non-empty; whether the paper's layer uses the same
   safeguard is not determinable from the claim text.
6. **Dimensions and cardinalities reached** are small: `n_out ≤ 6`, `m ≤ 6`. The cardinality
   clause was checked on a 6 × 5 grid only.
7. **Claim 1's bound is an upper bound**, and the measured ratio (1.71) sits far below it; a
   tighter constant is not tested, and a *failure* to reach the bound is not evidence about
   its sharpness.
8. **Claim 5's collision contrast is untested** (see above), and the dynamics are a
   single-integrator simplification; the paper's system is not specified in the claim text.
9. **Monte-Carlo resolution.** Claim 4 uses 3 seeds; the 9.22% figure has no reported
   standard error and should not be compared to 73.33% as if it were a measurement of the
   same quantity.
10. **float64 only.** Machine-precision quantities (0.0, 1e-16) are float64 artefacts, not
    exact algebra.