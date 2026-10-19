# MedCRP-CL release audit

Independent reproduction package for:

**MedCRP-CL: Continual Medical Image Segmentation via Bayesian Nonparametric
Semantic Modality Discovery** (`v0DWbfP3b9`, arXiv `2605.20297v1`).

Prepared result: **4/10** against the five prespecified two-point claims.

The package deliberately distinguishes release mechanics from empirical
reproduction. It safely audits the official modality and EWC states, exercises
CRP-supporting statistics and modality-specific LoRA behavior, checks the
clustering proposition, inventories missing empirical artifacts, and builds a
publication-ready Trackio logbook.

Run:

```powershell
python audit_medcrp_cl.py
python -m unittest discover -s tests -v
python repro_medcrp_cl/build_logbook.py
python ..\validate_icml_logbook.py
```

See `FROZEN_FORECAST.md`, `INDEPENDENT_CHECKS.md`, and
`PUBLICATION_PREFLIGHT.md` for the evidence boundary.
