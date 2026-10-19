# KM-ARL / KM-ADD reproduction bundle

CPU-only NumPy reproduction for arXiv `2605.18798`, OpenReview
`LhGxRnGmGJ`.

## Rerun

```powershell
python kmarl_repro.py
python sweep_thresh.py
python audit_finite_sample_bounds.py --output evidence/finite_sample_bounds.json
python poisson_reproduction.py --output evidence/poisson_figure13.json
```

The publication evidence was produced with Python 3.11 and NumPy. The two
repair scripts have fixed deterministic seeds and print a protocol SHA-256.

## Claim map

| Anchored claim | Evidence |
| --- | --- |
| C1, Theorem 4.1 | `audit_finite_sample_bounds.py`, ARL context |
| C2, Theorem 4.2 | `audit_finite_sample_bounds.py`, ADD context |
| C3, Theorems 4.3/4.4 | `kmarl_repro.py` |
| C4, Eq. 11 | `kmarl_repro.py` |
| C5, WISDM | Not attempted |
| C6, Gaussian and Poisson simulations | `kmarl_repro.py`, `sweep_thresh.py`, `poisson_reproduction.py` |

The theorem audit does not re-prove the external Stute (1994) lemma used by
the paper. It checks the paper's reduction from that lemma, independently
integrates the bounds, validates them by Monte Carlo, and includes a dependent
censoring destructive control.
