## Claim 2 — bofops: `ess sup_x ν_x(Ω) < ∞` (Definition 3.1)

Script `verify_claim2.py` → `results/claim2.json`. Verdict: **verified**. Mutation test: yes.

A boundedness condition cannot be checked at a single `n`. Sweep `n ∈ {100,200,400,800,1600}` and
fit the log-log growth exponent of `max_x ν_x(Ω)`:

| family | fiber sup across the sweep | exponent |
|---|---|---|
| bounded-degree (bofop) | 4, 4, 4, 4, 4 | **≈ 0** (|slope| < 0.05) |
| dense Erdős–Rényi `p=0.3` | 42 → 546 | 0.947 |
| single √n hub | 14 → 44 | 0.413 |

**Mutation.** Dropping the bounded-fiber hypothesis makes the fiber sup grow with `n` (exponent
0.947 dense, 0.413 for even a *single* √n hub). The definition genuinely discriminates a sparse
subclass inside the graphop class rather than being vacuous.

---
<!-- trackio-cell
{"type": "markdown", "id": "cell_<built-in function hash>", "created_at": "2026-01-01T00:00:00+00:00", "title": "Claim 2 \u2014 bofops: `ess sup_x \u03bd_x(\u03a9) < \u221e` (Definition 3.1)"}\n-->
