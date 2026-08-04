# Reproduction task — Efficient Neural Controlled Differential Equations via Attentive Kernel Smoothing

OpenReview: https://openreview.net/forum?id=e6hVbhHEXh
Area: Deep Learning
Anchored claims (6):

1. On CharacterTrajectories, MVC-CDE with GP smoothing requires 95.17 ± 7.41 function evaluations (NFE), versus 633.11 ± 28.31 for linear-spline and 595.91 ± 9.99 for cubic-spline neural CDEs (Table 1, Section C.1).

2. On SpokenArabicDigits, MVC-CDE with GP smoothing requires 95.40 ± 0.89 NFE versus 2237.40 ± 20.02 for linear spline and 1003.80 ± 24.19 for cubic spline, corresponding to total training times of 7.32 ± 0.15s versus 106.15 ± 3.76s and 57.90 ± 1.73s respectively (Table 1, Section C.1).

3. On UWaveGestureLibrary, MVC-CDE with GP achieves 218.60 ± 10.04 NFE and 12.29 ± 0.27s training time, versus 4340.60 ± 216.07 NFE / 176.19 ± 5.13s for linear spline and 1076.60 ± 33.57 NFE / 53.22 ± 2.25s for cubic spline (Table 1, Section C.1).

4. MVC-GP demonstrates a speedup ranging from 4.3× to 14.5× compared to standard spline-based neural CDE methods across the evaluated datasets (Section 4.2).

5. Theorem 3.1 shows that NFE is strictly determined by the L_{1/(p+1)}-quasi-norm of the (p+1)-th derivative of the control path, and Corollary 3.3 shows NFE scales as h^{-1} for smoothing-based paths with lengthscale h (Section 3.2, Section 3.4).

6. Figure 4 shows that under additive noise, smoothing-based methods (kernel/GP) maintain almost constant NFE, while interpolation-based (linear/cubic spline) methods show a drastic increase in both NFE and error rate (Section 4.3).

## Your job (autonomous)
Reproduce each anchored claim on CPU with numpy/scipy/sympy (no GPU). For every claim:
- Write `verify_claim<N>.py` that computes the claim's quantity from first principles.
- Save numeric result to `results/claim<N>.json`.
- Run a MUTATION test (perturb the setup; the claimed property must break or shift) for every verified claim.
- Write verdict: verified / falsified / toy / inconclusive (honest if data/GPU blocks full repro).
- Record exact source (section/equation/theorem) and a reproducible seed.

Hard budget: 8h total, 4h soft, 2h per claim. Write `logbook.md` when done.
