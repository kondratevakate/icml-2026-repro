## Claim 2 — parameter regime `n=Θ~(d²K)`, `m=Ω~(d⁸K⁴)`, `ηT=Θ(d²)`

**Source recorded:** `"Theorem 2.1 / Theorem C.1 (regime n=Θ̃(d²K), m=Ω̃(d⁸K⁴), ηT=Θ(d²))"`
**Route:** `"simulation of Algorithm 1 with the prescribed scalings; width scaled down"`, `K = 3`, `T = 40`, `reps_per_cell = 5`.

**Verdict: `toy`** — `verdict_cap_reason`:
> `"m = Ω̃(d⁸K⁴) unreachable on CPU; width substituted by m = 8d²"`
(the theorem's own requirement is logged as `"d^8 K^4 (e.g. 3.5e11 at d=16,K=3) — INFEASIBLE on CPU"`).

**Sub-verdicts, verbatim:**
- `"n and ηT scalings implementable and give decreasing o_d(1)-consistent forgetting": "verified"`
- `"full m = Ω̃(d⁸K⁴) regime": "inconclusive (computationally unreachable)"`

**Evidence (d-ladder, `n = 0.5d²K`, `ηT = 0.5d²`, `m = 8d²`):**

| d | n | ηT | m | \|F^tr(task 1)\| | stderr |
|---|---|---|---|---|---|
| 8 | 96 | 32.0 | 512 | 0.15652652737947723 | 0.03827 |
| 12 | 216 | 72.0 | 1152 | 0.029467489244489897 | 0.00716 |
| 16 | 384 | 128.0 | 2048 | 0.013955975321768755 | 0.00344 |
| 20 | 600 | 200.0 | 3200 | 0.007157140311013066 | 0.00178 |

`"forgetting_decreases_with_d": true`, `ratio_first_to_last = 21.869981665529412`. Metric note from the JSON: the linear surrogate `f(u)=1−u` was used because `"the hinge saturates to exactly 0 here and makes the quantity trivially unmeasurable"` (hinge values retained as `*_hinge`).

**Mutation — freeze n (`mutation_fixed_n`, n = 24):** `"property_breaks": false`; forgetting still decreased (`0.16556 → 0.03213`, `ratio_first_to_last = 5.153657378441504`). First attempt logged verbatim:
> `"PREDICTION WRONG on the first mutant, recorded verbatim: fixing n at its d=8 value (n=96) still gave decreasing forgetting, because n=96 already exceeds what these small d need. The mutant was hardened to n=24 (P58: remove the slack absorbing the mutation), not the verdict weakened."`

This failed mutation is the second reason the claim is capped below *verified*: at the reachable d-range the n-condition carries no measurable weight.

---

---
<!-- trackio-cell
{"type": "markdown", "id": "cell_<built-in function hash>", "created_at": "2026-01-01T00:00:00+00:00", "title": "Claim 2 \u2014 parameter regime `n=\u0398~(d\u00b2K)`, `m=\u03a9~(d\u2078K\u2074)`, `\u03b7T=\u0398(d\u00b2)`"}\n-->
