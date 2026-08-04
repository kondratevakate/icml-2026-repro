## Claim 3 — Theorem 2, uniformly small misclassification error over all K tasks

**Source recorded:** `"Theorem 2.2 (Sec 2.2); restated as Theorem C.1 (App. C)"`
**Route:** `"simulation of Algorithm 1, KT iterations, uniform-over-tasks statistics"`, `K = 3`, `T_per_task = 40`, `total_iterations = "K*T"`, `reps_per_cell = 8`.

**Verdict: `"verified (qualitative statement; width scaled down -> see verdict_cap)"`**, `verdict_cap: toy`, reason `"m = 8d² substituted for the theorem's Ω̃(d⁸K⁴) width"`.

**Evidence (max-over-tasks error after KT steps):**

| d | n | m | mean max-task err | worst-seed max-task err | mean max-task hinge |
|---|---|---|---|---|---|
| 8 | 96 | 512 | 0.0013020833333333333 | 0.010416666666666666 | 0.00817123452631201 |
| 12 | 216 | 1152 | 0.0 | 0.0 | 0.0016392272111534453 |
| 16 | 384 | 2048 | 0.0 | 0.0 | 9.631748391573346e-05 |
| 20 | 600 | 3200 | 0.0 | 0.0 | 3.257195534838518e-05 |

`"uniform_error_small": true`, `"uniform_error_nonincreasing_in_d": true` — including the *worst* seed, i.e. the "with high probability, uniformly over tasks" shape of the statement.

**Mutation — noise violation (σ × 8, `mutation_noise_violation`):** `"property_breaks": true`; prediction `"sigma x8 violates sigma = Theta(1/(log^c d sqrt d)); uniform train error must rise"`. Errors collapse to chance: `[0.5104, 0.5081, 0.5013, 0.5023]`, i.e. `mean_max_task_err_ratio_vs_main = [392.0, 5.081e11, 5.013e11, 5.023e11]`.

---

---
<!-- trackio-cell
{"type": "markdown", "id": "cell_<built-in function hash>", "created_at": "2026-01-01T00:00:00+00:00", "title": "Claim 3 \u2014 Theorem 2, uniformly small misclassification error over all K tasks"}\n-->
