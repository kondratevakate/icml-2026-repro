## Claim 5 — safety-critical control: CAffNet avoids obstacles, HardNet and soft NN fail

**Source:** §4.3 / Table 4 (NN 2.60% violated, HardNet 2.60%, CAffNet-FF 0.00%) with App. D.3.

**Why the headline claim is not reproducible from the paper.** App. D.3 gives the three
obstacle polytopes, the state and input boxes, the PID gains and x0 = [−4.5, 0, 0.5], but it
does **not** give: the reference trajectory, the integration step dt, the episode horizon, the
training loss/dataset for the policy, or — decisively — the *affine* per-step constraint
A(x)u ≤ b(x) that is supposed to encode obstacle avoidance. Avoiding a polytopic obstacle is
the complement of a polytope, i.e. non-convex, so some hyperplane-selection scheme must be
supplied before any affine-constraint layer can be applied at all. Without it, Table 4's
numbers (cost 4.14e5 / 4.57e5 / 7.21e5, violation 2.60% / 2.60% / 0.00%) cannot be targeted.
Per the task rules I do not substitute a fabricated equivalent. **Verdict: inconclusive.**

**What was executed instead (labelled `toy`, `verify_claim5.py`, raw `results/claim5.json`).**
A well-posed mechanism-level sub-question: with the paper's obstacles, input box and PID
nominal controller, a *self-designed* discriminating-hyperplane constraint per obstacle
(keep the currently most-satisfied separating hyperplane satisfied at the next Euler step,
margin 0.05, dt 0.05, 1200 steps), giving m = 7 > n_out = 2 — exactly the regime the paper
says HardNet cannot handle — does the CAffNet projection keep the loop collision-free?

| Controller | seeds collided | max constraint violation |
|---|---|---|
| soft (box-clipped PID, no projection) | **5 / 5** | 7.314 |
| HardNet-style single pseudo-inverse | **5 / 5** | 1.167 |
| CAffNet (Eqs. 8/9/12) | **0 / 5** | 3.61e−15 |

This is consistent with the paper's qualitative statement, but it validates *my* constraint
construction, not the paper's, so it cannot upgrade the verdict beyond `inconclusive`.
Note also that the k = 1 truncation mutation gave identical results here (0/5 collided),
i.e. this particular scenario does not exercise the higher-cardinality combinations.

---

---
<!-- trackio-cell
{"type": "markdown", "id": "cell_<built-in function hash>", "created_at": "2026-01-01T00:00:00+00:00", "title": "Claim 5 \u2014 safety-critical control: CAffNet avoids obstacles, HardNet and soft NN fail"}\n-->
