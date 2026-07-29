# Tailored causal scoring-rule reproduction

This CPU-only bundle audits six claims from *Tailoring Strictly Proper Scoring
Rules for Downstream Tasks: An Application to Causal Inference* (arXiv
`2606.03332v1`, OpenReview `JTwryHNicJ`).

## Result

- C1 `PARTIALLY VERIFIED`: the IPW bound ledger closes conditional on a fixed
  propensity function. Standard K-fold cross-fitting alone does not justify
  the paper's unconditional `Var(mean)=Var(Z)/N` step because fold
  contributions may be dependent.
- C2 `VERIFIED`: SymPy derives the declared task curvature exactly; a changed
  boundary exponent fails a 998-point high-precision grid.
- C3 `VERIFIED`: both displayed partial losses satisfy the proper-loss
  differential equations, the weight is positive, and conditional risk is
  minimized at the truth on a 99 x 3993 exhaustive grid.
- C4 `PARTIALLY VERIFIED`: the integrated link, quartic, inverse mapping,
  symmetry, monotonicity, and `p-y` gradient all hold. It is nonzero at every
  finite logit, but tends to zero for confidently correct predictions, so the
  asymptotic no-vanishing interpretation is false.
- C5 `PARTIALLY VERIFIED`: a 30-seed, 10-fold independent Kang-Schafer panel
  gives a 7.10x log-loss/tailored IPW RMSE ratio on misspecified features and a
  93.3% tailored win rate. The advantage shrinks to 1.24x on latent features.
- C6 `INCONCLUSIVE`: the IHDP, Jobs, and ACIC headline was not rerun because
  author code, per-run data, seeds, splits, and tuning grids are unavailable.

Prepared score: `7/12`.

## Run

```bash
python prepare_official.py
python audit_claims.py \
  --paper-source ../official/source/main.tex \
  --output evidence/claims_audit.json
python -m unittest discover -s tests -v
```

The audit is deterministic under seed panel `20260729..20260758`; its SHA-256
is recorded in the evidence JSON.

## Integrity

The publishable bundle contains no author code, benchmark datasets, treatment
records, model weights, paper source, or secrets. No leaderboard entry or peer
reproduction is used as evidence.
