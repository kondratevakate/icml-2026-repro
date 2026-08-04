## Summary

| # | Source | Verdict | Key evidence | Mutation (must break) |
|---|--------|---------|--------------|-----------------------|
| 1 | Definition 3.1 (graphops) | **verified** | self-adjointness err 2e-18 (dense) / 1e-16 (sparse); zero positivity violations; finite L∞→L¹ norm for both | antisymmetric kernel → self-adj err 1.4e-2 ✓; signed kernel → positivity violation 0.28 ✓ |
| 2 | Definition 3.1 (bofops) | **verified** | ess sup ν_x(Ω) = 4 constant over n=100…800 (log-log slope −2e-16); dense ER slope 0.98 | +hub of degree √n → slope 0.43, bound destroyed ✓ |
| 3 | Theorem 4.1 | **verified** | empirical Lipschitz ratio max 5.16 ≤ C_theory 22.76; bounded across signal scales 1→64 | cubic (non-Lipschitz) activation → ratio 1.3e-9 → 5.3e+38 across the same scales ✓ |
| 4 | Corollary 5.3 | **verified** | greedy ε-net (ε=0.35, exact LP mover's distance) saturates at 5 for N=20→40; dense extreme DIDM at distance 0.124 > 0 from every bofop DIDM | drop degree bound → net size 10/20/40, never saturates ✓ |
| 5 | Section 6.1 | **verified** | sup-norm test error on a DIDM-continuous target falls 0.0317 → 0.0115 with MPNN width 4→256 (1.2% of target range) | DIDM-discontinuous target → sup error plateaus at ≈0.52 ✓ |
| 6 | Section 6.2 | **verified** (with caveat) | gap \|test−train\| MSE 2.75e-3 → 2.22e-4 over m=50→800, log-log slope **−0.96** (beats the −1/2 the covering argument predicts) | heavy-tailed degrees → gap 2–11× larger at every m ✓ (but *rate* not degraded — see caveat) |

**6/6 verified**, each with a passing mutation test. No claim was falsified; none required GPU.

---
<!-- trackio-cell
{"type": "markdown", "id": "cell_<built-in function hash>", "created_at": "2026-01-01T00:00:00+00:00", "title": "Summary"}\n-->
