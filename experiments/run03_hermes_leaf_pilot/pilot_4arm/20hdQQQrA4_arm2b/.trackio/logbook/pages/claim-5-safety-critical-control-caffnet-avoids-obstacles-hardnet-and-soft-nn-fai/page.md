## Claim 5 — safety-critical control: CAffNet avoids obstacles, HardNet and soft NN fail
**Verdict: inconclusive** (CAffNet-vs-soft-NN direction strongly reproduced; HardNet arm not
reproducible; CAffNet not collision-free)
Source: Section 4.3, Appendix D.3, Figs 6–7 / Table 4. Script: `verify_claim5.py` ->
`results/claim5.json`. Command: `./.venv/bin/python verify_claim5.py --epochs 8 --ntrain 12 --seeds 3`
(paper: 300 initial states).

Full D.3 problem implemented: unicycle, dt = 0.1 s, 150 steps, 3 polytopic obstacles with the
exact A_j, b_j, per-edge CBF, smooth union (kappa = 10, eta_j = ln m_j), state box and control
box (A_u, b_u), alpha(h) = h, PID nominal + network correction; aggregated A(x) is 13 x 2
(`m_constraints: 13, n_out: 2`), |Gamma| = 91 (`n_gammas: 91`). Evaluation: 49 held-out initial states.

| method | collisions / 49 (mean of 3 seeds) | control violations | arrived (<0.5 m) | steps with empty S(x) |
|---|---|---|---|---|
| NN soft | **20.333333333333332** (24, 19, 18) | 24.0 | 47.666… | 0 |
| CAffNet-FF | **1.0** (1, 1, 1) | 0.3333333333333333 | 15.666… | 4–4–4 |
| CAffNet-FF a-posteriori projection | **1.0** | 0.3333333333333333 | 13.333… | 0 |

- The qualitative claim direction is reproduced: the soft-constrained baseline collides on
  ~41% of initial states (mean 20.33/49), CAffNet on 1/49 (2.04%).
- The residual CAffNet collision is **not** a contradiction of Theorem 3.4 but its precondition:
  every CAffNet-FF seed reports 4 simulation steps where the feasible set S(x) is **empty**
  (`infeasible_steps: 4` for all three CAffNet-FF seeds; Assumption 3.2 fails: discrete-time CBF +
  tight actuator box), and that is exactly where the single collision and the single control
  violation occur.
- **HardNet arm not reproduced** (honest refusal, not a toy substitute): HardNet's projection
  requires A(x) with full row rank; here A(x) is 13 x 2 (rank 2), so the formula is undefined
  (`hardnet_not_reproduced_reason` in the JSON).
- Also observed (paper's side remark): a-posteriori projection reaches the goal less often
  (13.333 vs 15.666 of 49) than the jointly trained CAffNet, consistent with the paper's
  "gets stuck near an obstacle" observation, though at this small training budget the effect is weak.
- Reduced training (12 initial states, 8 epochs vs the paper's 300 states) is the main reason the
  CAffNet arrival rate (15.666/49) is far below the paper's; hence inconclusive rather than verified.

---

---
<!-- trackio-cell
{"type": "markdown", "id": "cell_<built-in function hash>", "created_at": "2026-01-01T00:00:00+00:00", "title": "Claim 5 \u2014 safety-critical control: CAffNet avoids obstacles, HardNet and soft NN fail"}\n-->
