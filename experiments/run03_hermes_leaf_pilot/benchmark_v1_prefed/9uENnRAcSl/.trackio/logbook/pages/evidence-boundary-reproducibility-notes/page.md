## Evidence boundary & reproducibility notes

- **CPU-only, reproducible.** Every `results/claim<N>.json` is produced by `verify_claim<N>.py`
  with a fixed, recorded `seed` (20260320–20260326). Re-running any `verify_claim<N>.py` under the
  provided `.venv` regenerates identical numbers (the verify scripts use `numpy` only).
- **Claims 1–5 (verified)** are theory-driven: analytic rate-matching, bound-formula evaluation,
  and faithful dynamical/instantiation constructions (the paper's exact trap-loop for Claim 3).
  Each verified claim has a mutation test that *flips* the claimed property, confirming mechanism
  attribution (log-loss necessity, RTVC necessity, non-smoothness, augmentation necessity, tightness).
- **Claim 6 (toy)** is an empirical phenomenon with no released data; our synthetic stand-in
  reproduces the qualitative direction only. Marked honestly as `toy`, not `verified`.
- **No claim falsified.** `check_reproducibility.py` re-asserts every number above against
  `results/claim<N>.json` (bounds/relations, slopes within tolerance, mutation `passed`) and exits
  0 only when all checks pass.

---
<!-- trackio-cell
{"type": "markdown", "id": "cell_<built-in function hash>", "created_at": "2026-01-01T00:00:00+00:00", "title": "Evidence boundary & reproducibility notes"}\n-->
