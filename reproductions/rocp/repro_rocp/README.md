# ROCP reproduction bundle

Independent audit bundle for *Optimal Decision-Making Based on Prediction Sets*
(`VAXW59dyfk`, arXiv `2602.00989v3`).

## Run

```bash
python -m pip install -r requirements.txt
python run_all.py --official-repo /path/to/official/repo
```

The default C1 run audits an exhaustive small loss family plus 120 seeded
larger cases. `run_all.py` exhausts all C3 exchangeability orbits through
sample size `n+1=6`. The official checkout is optional; without it the
paper-to-code comparison is skipped while C1 and C3 still run.

The cached empirical evaluator is resumable and parallelizes independent
seeds/splits while calling the official `evaluate_seed` unchanged:

```bash
python run_cached_empirics.py --official-repo ../official --dataset covid --variant lambda0 --workers 4
python run_cached_empirics.py --official-repo ../official --dataset covid --variant lambda1 --workers 4
python run_cached_empirics.py --official-repo ../official --dataset bdd --workers 4
python validate_cached_empirics.py
```

Each completed seed is written before aggregation, so an interrupted run can
resume without recomputing finished work.

## Scope

- C1 is a proof-backed finite-label audit of Lemma 2.1 and Theorem 2.2,
  independently checked against the primal linear program.
- C3 is a literal implementation of Algorithm 1's candidate-label `(n+1)`
  calibration and an exhaustive finite exchangeability audit.
- C4 is not claimed until the full cached 20-seed COVID and 20-split BDD
  outputs have completed and passed claim-level review.

The completed empirical scope and its stronger-statement limitation are
documented in `EMPIRICAL_CLAIM_AUDIT.md`.

See `CLAIM_AUDIT.md` for the derivations, exact claim boundary, and the two
paper-to-code divergences found during implementation.

## Sources

- Paper: https://arxiv.org/abs/2602.00989
- Official code: https://github.com/TaoWangPenn/Risk-Optimal-Conformal-Prediction
