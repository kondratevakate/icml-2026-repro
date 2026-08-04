## Claim 3 — Corollary 1, Section 4.2 (`verify_claim3.py` → `results/claim3.json`)

The full breakpoint sequence is enumerated by divide-and-conquer over λ, counting min-cut
calls; runtime is fitted as `T/γ ~ (m+n)^α`.

| n | m | γ | nested chain | min-cut calls | s |
|---|---|---|---|---|---|
| 8 | 9 | 6 | yes | 148 | 0.23 |
| 12 | 16 | 4 | yes | 92 | 0.20 |
| 16 | 42 | 3 | yes | 67 | 0.37 |
| 20 | 66 | 3 | yes | 65 | 0.57 |
| 24 | 98 | 3 | yes | 62 | 0.79 |

* γ ≤ n+1 on every instance; every sequence is a nested chain.
* Calls per breakpoint bounded (≈21–25, the log-factor of the recursion) → `O(γ·polylog)`
  min-cut solves, matching the Õ(γ·…) shape.
* Empirical exponent **α = 1.02 ≤ 2**, i.e. observed cost per breakpoint is *below* the
  claimed (m+n)² bound (networkx's preflow-push on sparse graphs).
* **Mutation:** a naive uniform 200-point λ grid uses more cut solves than the D&C
  recursion on every instance and recovers no additional breakpoints.

Verdict: **verified** (the stated bound holds and is not violated; it is an upper bound,
observed cost is smaller).

---
<!-- trackio-cell
{"type": "markdown", "id": "cell_<built-in function hash>", "created_at": "2026-01-01T00:00:00+00:00", "title": "Claim 3 \u2014 Corollary 1, Section 4.2 (`verify_claim3.py` \u2192 `results/claim3.json`)"}\n-->
