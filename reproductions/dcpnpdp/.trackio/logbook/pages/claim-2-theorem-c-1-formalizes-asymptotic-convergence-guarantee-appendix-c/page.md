# Claim 2: Theorem C.1 formalizes asymptotic convergence guarantee (Appendix C)


---
<!-- trackio-cell
{"type": "markdown", "id": "cell_2d72ce852beb", "created_at": "2026-07-20T08:31:34+00:00", "title": "Claim 2: Theorem C.1 formalizes asymptotic convergence guarantee (Appendix C)"}
-->
**Claim (verbatim, anchored):** "Theorem C.1 shows that dual-coupled fixed points satisfy the first-order optimality condition 0 in the subdifferential of f(x*) + phi(x*) of the original reconstruction problem, formalizing the asymptotic convergence guarantee (Appendix C, Theorem C.1)."

**Setup.** Same synthetic subspace-manifold construction as Claim 1: exact proximal denoiser `prox_shrink(v) = Pv + s(I-P)v` for the regularizer phi(x) = (1/2gamma) dist(x,M)^2, satisfying the paper's own Assumption D.1 (denoiser-proximal equivalence) exactly and non-vacuously. Theorem C.1 states: at a fixed point (x*,z*,u*) of the deterministic backbone, under Assumption D.1, x* satisfies `0 in grad f(x*) + d phi(x*)` with effective weight lambda = rho*gamma — a genuine first-order optimality *characterization of fixed points*, contrasted against the loose-coupled fixed point which instead satisfies `0 in grad f(x~) + d phi(x~) + (x~ - D(x~))`, i.e. carries a systematic bias term. This part is not in dispute — it is exactly what the same numbers in Claim 1 demonstrate (loose-coupled residual floor vs. dual-coupled residual to zero).

**What is in dispute is the claim's own wording: "formalizing the asymptotic convergence guarantee."** Two problems, both traceable to the paper's own text, not to our interpretation:

1. **No convergence dynamics are proven, only a property of fixed points *if* one is reached.** The paper itself concedes, in its own words: "establishing global convergence guarantees for non-convex PnP algorithms with stochastic diffusion priors remains an open theoretical challenge." Theorem C.1 is a fixed-point characterization, not a convergence proof — it says nothing about whether, or how fast, the iteration in fact reaches a fixed point.

2. **Even granting a fixed point is reached, it is a stationary point of the tradeoff f + lambda*phi (data fidelity vs. prior), not literally a point on the data manifold {phi=0}.** It lands on the manifold only in the degenerate limit gamma to infinity (a hard projection) — precisely the regime the paper's own Assumption D.1 (built for a finite-gamma proximal denoiser) does not cover. This is directly visible in the Claim 1 numbers reused here: distance to manifold is O(s) and strictly nonzero for every non-degenerate (s>0) denoiser; exact manifold membership occurs only at the idealized s=0 limit, which is outside the finite-gamma proximal-denoiser regime the theorem's own assumption is stated for.

| s | dist(x*, M) at dual-coupled fixed point |
| --- | --- |
| 0.5 | 9.6e-03 |
| 0.1 | 2.3e-03 |
| 0.01 | 2.5e-04 |
| 0 (idealized, gamma to infinity) | 3.9e-16 |

**Verdict — PARTIALLY CONFIRMED / label overstated.** This is a legitimate nuanced finding, not a failure of the theorem or of the reproduction. The fixed-point optimality characterization itself is real, reproducible, and matches the numbers above exactly. But "formalizes the asymptotic convergence guarantee" claims two things the theorem does not deliver: (i) convergence *dynamics* (the paper's own text disclaims global convergence for this non-convex setting), and (ii) landing on the exact data manifold at finite gamma (only the gamma to infinity degenerate limit gets there, outside Assumption D.1's own stated regime). Honest status: the underlying math of Theorem C.1 is correct; the "asymptotic convergence to the exact data manifold" framing in the claim's own wording is not what the theorem as stated proves.


---
<!-- trackio-cell
{"type": "dashboard", "id": "cell_1af444063dc0", "created_at": "2026-07-20T14:06:09+00:00", "title": "Dashboard: repro-plug-and-play-diffusion-meets-admm-dual-variable-coupling-for-robust-medical", "dashboard_project": "repro-plug-and-play-diffusion-meets-admm-dual-variable-coupling-for-robust-medical"}
-->
**🎯 Trackio dashboard** `repro-plug-and-play-diffusion-meets-admm-dual-variable-coupling-for-robust-medical`

trackio-local-dashboard://repro-plug-and-play-diffusion-meets-admm-dual-variable-coupling-for-robust-medical
