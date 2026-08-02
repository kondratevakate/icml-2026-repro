# Claim 3 — no full-row-rank requirement; cardinality <= min(m, n_out)

---
<!-- trackio-cell
{"type": "markdown", "id": "cell_de26f5ada72b", "created_at": "2026-08-02T04:30:00+00:00", "title": "Claim 3 \u2014 no full-row-rank requirement; cardinality <= min(m, n_out)"}
-->
*Source:* Section 3 / 3.1 (Eq. 2), Lemma 3.3 (App. A), Theorem 3.4 (App. B), Table 1.
*Type:* simulation, exhaustive. *Script:* `verify_claim3.py` -> `results/claim3.json`.

**Verdict: verified.**

- **Exhaustive family (E1):** all m-subsets (m = 2..5) of a 9-direction dictionary in R^2 that
  contains scaled/negated duplicates -> 372 constraint systems (9 rank-deficient), each with a
  5x5 grid of `f_theta` and 3 values of `w_phi` = **27 900 cases**. CAffNet feasibility failures:
  **0**, max violation **0.0**.
- **Random sweep (E2):** n_out in {1..4}, m up to 8 (> n_out), 0/1/2 forced dependent rows,
  200 seeds each = **8 800 cases**, 1 275 rank-deficient. CAffNet failures: **0**, max violation **0.0**.
  (36 700 cases in total.)
- **Baseline (E3):** the HardNet-Aff single pseudo-inverse correction on the *same* instances is
  infeasible in **59.2%** (E1) and **77.3%** (E2) of cases — the failure mode the paper illustrates
  in Fig. 1.
- **Cardinality (E4):** for all m = 1..10, n_out = 1..5, `|Gamma| = sum_{k<=min(m,n_out)} C(m,k)`
  exactly as in Eq. (2), max sub-constraint size = min(m, n_out), and `|Gamma| <= 2^m − 1`: all OK.

Mutation: truncate Gamma to `k <= min(m, n_out) − 1`. Feasibility failures immediately appear —
**24.4%** of E1 cases and **23.8%** of E2 cases — so `min(m, n_out)` is not a decorative bound but
the cardinality actually needed for the Lemma-3.3 guarantee.
