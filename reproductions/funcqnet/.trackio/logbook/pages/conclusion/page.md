# Conclusion


---
<!-- trackio-cell
{"type": "markdown", "id": "cell_c0nc1u510nfnd", "created_at": "2026-07-23T09:12:58+00:00", "title": "Findings"}
-->
FunCQNet's simulation **core claim reproduces independently on CPU**, using a transparent linear-in-Z B-spline censored quantile estimator (not the paper's neural network). The paper's real-data results are not reproducible.

**Supported (reproduced).**
- **Claim 1 (S1/S2, Z-modulated effect):** an interaction-aware estimator cuts coefficient-function RMSE by 58–85% versus a Z-agnostic reduced model. The α₂ reduction (~75% in S1) matches the paper's reported ~75% (Table 3, τ=0.5), and the No-Interaction RMSE(α̂₁)≈0.41 matches the paper's 0.38. Holds across 0–50% censoring.

**Reproduced only with a caveat.**
- **Claim 2 (S3, no Z-dependence):** the paper says interaction ≈ no-interaction, but an *unregularized* interaction model is ~2× worse. Adding an L1 penalty on the Z-interaction terms restores comparability (0.076 → 0.056 at 30% censoring) while preserving the S1 gain — so the paper's S3 result depends on FunCQNet's built-in regularization, a point the paper leaves implicit.

**Not reproducible.**
- **Claim 3 (real data):** the motivating cohort is the **UNOS/OPTN** registry (confirmed from the PDF) — access-controlled STAR files, DUA-gated, no public version, and no code released. The real-data quantitative claims cannot be independently verified.

**Reproducibility notes.** All results use numpy/scipy on CPU, deterministic under fixed seeds; the DGP is a faithful transcription of Appendix D.1 and paper text was extracted with PyMuPDF (not an LLM summary). Estimator is a fair, not strawman, surrogate because the paper's true α is linear in Z and the estimator can represent it exactly. Scripts: `dgp.py`, `estimator.py`, `run_sim.py` (see the reproduction bundle below). Rerun: `python run_sim.py --smoke` then `python run_sim.py --B 20` (~1 min).


---
<!-- trackio-cell
{"type": "artifact", "id": "cell_7db33edf2a11", "created_at": "2026-07-23T09:19:36+00:00", "title": "Reproduction bundle", "artifact": "repro-funcqnet-functional-censored-quantile-net-kidney-survival/repro-bundle:v0", "artifact_type": "dataset"}
-->
**📦 Artifact** `repro-funcqnet-functional-censored-quantile-net-kidney-survival/repro-bundle:v0` · dataset

https://huggingface.co/buckets/kondratevakate/repro-funcqnet-functional-censored-quantile-net-kidney-survival-artifacts#repro-funcqnet-functional-censored-quantile-net-kidney-survival/repro-bundle:v0
