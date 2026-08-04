## Reproducibility & honesty notes

- **Determinism:** `check_reproducibility.py` re-runs every claim with the
  pinned seed 20260501 and asserts the verdicts and key numbers match
  `results/claim<N>.json`. (All randomness via `np.random.default_rng(SEED)`.)
- **Honest verdicts:** every claim is `verified` because each was reproduced
  from first principles to the paper's stated numbers (6.2%, 16.8%, 66%, and
  the general monotonicity inequalities over thousands of random instances).
  No claim was marked `falsified` or `inconclusive` because no discrepancy with
  the paper was found.
- **Caveats:** the non-monotonicity counterexamples (Claims 4, 5, and B.4)
  use the paper's *published canonical equilibrium multiplier profiles*; we
  verified their internal consistency (CPA binds, allocation matches, refinement
  holds) and recomputed the auction outcomes rather than merely copying the
  printed totals. Mutation tests confirm the results are sensitive to the
  setup, as expected for existence-style claims.
- **Environment:** Python 3.12 venv created in `.venv/` (numpy/scipy/sympy);
  no external data or network access needed for the computations.

---
<!-- trackio-cell
{"type": "markdown", "id": "cell_<built-in function hash>", "created_at": "2026-01-01T00:00:00+00:00", "title": "Reproducibility & honesty notes"}\n-->
