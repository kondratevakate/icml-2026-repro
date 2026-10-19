# Stable Localized Conformal Prediction independent reproduction

This CPU-only bundle audits claims from *Stable Localized Conformal Prediction
via Transduction* (arXiv `2605.01452v1`, OpenReview `lSMTccAN61`).

## Run

```bash
python audit_claims.py --output evidence/claims_audit.json
python -m unittest discover -s tests -v
```

If the pinned paper source is available locally:

```bash
python audit_claims.py \
  --source ../official/source/main.tex \
  --output evidence/claims_audit.json
```

The audit is deterministic and independent. It does not use leaderboard
entries or other participants' reproductions. The paper source is used only
to anchor exact statements; the numerical checks and counterexamples are
implemented from scratch.

The author implementation can be executed separately when it is checked out
under `../official/code`:

```bash
python run_official_synthetic.py \
  --output evidence/logabs_canonical.json
```

That command uses the paper's full 50-repeat LogAbs protocol and can take
hours on CPU. A reduced run must be labeled directional:

```bash
python run_official_synthetic.py \
  --repeats 10 \
  --test-points 200 \
  --epochs 50 \
  --output evidence/logabs_reduced.json
```

## Current verdicts

- C1 `VERIFIED`: the law-of-total-variance experiment distinguishes the
  paper's construction-level stability from raw per-test set-size variance.
- C2 `NOT ESTABLISHED AS WRITTEN`: the proof uses a realized random
  `delta_S` to bound an unconditional expectation. A two-point quantile
  counterexample breaks that step; using `E[delta_S]` repairs it.
- C3 `NOT ESTABLISHED UNDER STATED ASSUMPTIONS`: the rate algebra passes an
  independent surrogate test, but the proof explicitly relies on unstated
  local second-order smoothness and does not justify its `O_p` to
  second-moment transition.
- C4 `FALSIFIED AS WRITTEN`: the proof's
  `ceil(n p)/(n+1) >= p` assertion is false. For the paper's
  `n=30`, `alpha=0.1`, `alpha_tol=0.02` setting, the exact continuous-score
  coverage at `q_L` is `27/31 = 0.87097`, below the claimed `0.88`.
- C5 is pending a fresh DermaMNIST or TissueMNIST run.
- C6 is pending a fresh official LogAbs synthetic run.

The paper defines Std with the sample denominator `R-1`, while the released
synthetic aggregator uses NumPy's default population denominator `R`. The
runner reports both the released value and the sample-corrected value.

`requirements.txt` is sufficient for the independent audit.
`requirements-official.txt` records the local environment used to execute
the pinned author pipeline; the tested Torch build was `2.10.0+cpu`.

No official source, author code, model weights, datasets, or third-party
results are included in this publishable bundle.
