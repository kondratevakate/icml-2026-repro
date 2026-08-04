## Claim 1 — Theorem 3.5: universal approximation with bound ||P* - f_t||_p < (3 + 3 sqrt(n_out)) K
**Verdict: verified**
Source: Theorem 3.5 (Sec 3.2) and its proof in Appendix C (K = eps / (3 + 3 sqrt(n_out))).
Script: `verify_claim1.py` -> `results/claim1.json` (command: `./.venv/bin/python verify_claim1.py`).

- Symbolic: `(3 + 3 sqrt(n_out)) * K - eps == 0` for K = eps/(3+3 sqrt(n_out)) — **true** (sympy,
  `symbolic_constant_closes: true`).
- Numeric stress: 8 (n_out, m) configs x 200 seeds, K = 1e-3, p = 2, targets placed on the
  polyhedron boundary so the projection branch of Eq (12) actually fires
  (110–200 projection-branch instances per config); `infeasible_instances: 0`.
  Max observed ratio ||P* - f_t||_2 / K vs the claimed bound:
  | (n_out, m) | max ratio | bound 3+3√n_out |
  |---|---|---|
  | (1,2) | 0.9971001586566164 | 6.0 |
  | (1,4) | 0.9923570959997907 | 6.0 |
  | (2,3) | 1.6424769212615287 | 7.242640687119286 |
  | (2,5) | 1.446907792078362 | 7.242640687119286 |
  | (3,4) | 1.6027273263822814 | 8.196152422706632 |
  | (3,7) | 1.2320821394200527 | 8.196152422706632 |
  | (4,6) | 1.68252632685596 | 9.0 |
  | (5,8) | 1.686931229412946 | 9.708203932499369 |
  Overall max ratio **1.686931229412946** (`max_ratio_overall`) — bound holds in 8/8 configs
  (`n_configs_bound_holds: 8 / n_configs_total: 8`, and even under the strictly tighter constant
  `1 + sqrt(n_out)`, `holds_under_tight_constant_1_plus_sqrt_n: true` for every config). The
  theorem's constant is valid but loose.
- **Mutation** (invert the selection rule of Eq 12: take the *farthest* feasible candidate instead
  of the nearest — `argmax instead of argmin in Eq 12`): the bound is **broken in 8/8 configs**
  (`holds: false` for all), max ratios 18.96320931916251, 7.846957574033642, 112.1564423072167,
  126.41046987139862, 71.4357431773435, 275.42964969616213, 202.0805111475756, 677.5939335688797.
  => the argmin in Eq (12) is the mechanism that yields the bound, not an accident.

---
<!-- trackio-cell
{"type": "markdown", "id": "cell_<built-in function hash>", "created_at": "2026-01-01T00:00:00+00:00", "title": "Claim 1 \u2014 Theorem 3.5: universal approximation with bound ||P* - f_t||_p < (3 + 3 sqrt(n_out)) K"}\n-->
