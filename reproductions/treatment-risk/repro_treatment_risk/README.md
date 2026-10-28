# Treatment-Risk Reproduction Bundle

Independent reproduction bundle for *Learning Treatment Allocations with Risk
Control Under Partial Identifiability* (arXiv `2505.08378`, OpenReview
`9kWVQtpvJi`).

## Environment

- Python 3.11
- NumPy 1.26
- SciPy 1.15
- Matplotlib 3.x
- CPU only

Install the numerical dependencies in a clean environment:

```bash
python -m pip install "numpy>=1.26,<2" "scipy>=1.15,<2" "matplotlib>=3.8,<4"
```

## Commands

Run from this directory:

```bash
python audit_lemma41.py --output evidence/lemma41_finite_audit.json
python audit_theorem43.py --maximum-grid-size 10 --output evidence/theorem43_audit.json
python reproduce_synthetic.py --runs 1000 \
  --output evidence/synthetic_reproduction.json \
  --raw-output evidence/synthetic_raw.npz \
  --figure evidence/synthetic_coverage.png
```

Expected headline results:

- Lemma 4.1: all `268800` exact checks pass; underestimated-`Gamma`
  controls violate both bounds.
- Theorem 4.3: Eq. (18) counterexample reproduced; finite-grid repair has
  zero counterexamples through grid size 10; Bentkus coverage passes `165/165`.
- Synthetic experiment: valid arms control `8/8` tolerances; the invalid
  confounded `Gamma=1` arm controls `5/8` with minimum coverage near `0.26`.

The full synthetic run uses seed `20260728` and takes about four minutes on the
reference Windows CPU. The JSON records protocol and environment metadata; the
NPZ contains all per-run population risks, treatment risks, and treated
fractions.

## Scope

Tennessee STAR and International Stroke Trial results are not included because
the exact third-party analysis cohorts were unavailable. No substitute dataset
is used.
