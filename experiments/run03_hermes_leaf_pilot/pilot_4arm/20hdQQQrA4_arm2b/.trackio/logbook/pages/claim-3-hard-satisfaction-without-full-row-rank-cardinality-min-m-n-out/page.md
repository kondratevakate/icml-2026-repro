## Claim 3 — hard satisfaction without full row rank; cardinality <= min(m, n_out)
**Verdict: verified**
Source: Section 3.1–3.2, Eq (2), Lemma 3.3, Theorem 3.4. Script: `verify_claim3.py` ->
`results/claim3.json`.

- 2100 instances (`n_instances_total: 2100`) = 7 configs (n_out, m) in
  {(1,4),(2,5),(2,8),(3,7),(3,10),(4,9),(5,12)} x 3 instance kinds
  {generic, duplicate_rows, rank_deficient} x 100 seeds.
- CAffNet: instances with **no feasible candidate = 0** (`caffnet_instances_with_no_feasible_candidate: 0`),
  instances with a constraint violation = **0** (`caffnet_instances_with_constraint_violation: 0`),
  max violation over all 2100 instances = **1.1075584893660562e-12**
  (`caffnet_max_violation_over_all_instances`, numerical zero). Lemma 3.3 / Theorem 3.4 hold on
  every instance including redundant and rank-deficient A.
- HardNet baseline (full-row-rank formula): **undefined on 1961 / 2100** instances
  (`hardnet_instances_undefined_due_to_rank_deficiency: 1961`) because A A^T is singular; on the
  139 where it is defined it produced 0 violations (`hardnet_instances_with_constraint_violation: 0`).
  This is exactly the limitation the claim asserts.
- Cardinality/count: max |gamma| over Gamma == min(m, n_out) (`T3_max_cardinality_equals_min_m_nout: true`)
  and |Gamma| == sum_{k=1..min(m,n_out)} C(m,k) <= 2^m - 1 (`T3_gamma_count_identity_and_2m_bound: true`),
  checked **exhaustively** for m = 1..10 and n_out = 1..5.
- **Mutation** (cap the combination cardinality at min(m, n_out) - 1, i.e. violate the stated
  decomposition size): feasibility is lost on **478 / 2100** instances
  (`MUTATION_kmax_minus_1_failure_count: 478 / MUTATION_kmax_minus_1_total: 2100`). The
  min(m, n_out) cardinality is necessary, not decorative.

---
<!-- trackio-cell
{"type": "markdown", "id": "cell_<built-in function hash>", "created_at": "2026-01-01T00:00:00+00:00", "title": "Claim 3 \u2014 hard satisfaction without full row rank; cardinality <= min(m, n_out)"}\n-->
