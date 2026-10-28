# Conclusion


---
<!-- trackio-cell
{"type": "markdown", "id": "cell_32a073905bc6", "created_at": "2026-07-20T08:31:34+00:00", "title": "Reproduction bundle"}
-->
**Summary.** Claim 1 (dual-variable coupling removes steady-state bias) reproduces cleanly across 12 seeds with zero overlap between the dual-coupled and loose-coupled schemes at every softness level. Claim 2's fixed-point optimality theorem is mathematically correct and reproducible, but the claim's "formalizes the asymptotic convergence guarantee" framing overstates what the theorem delivers: no convergence dynamics are proven (the paper's own text disclaims global convergence for non-convex PnP), and the fixed point reaches the exact data manifold only in the degenerate gamma to infinity limit, outside the finite-gamma regime the paper's own Assumption D.1 is built for.

**Reproduction bundle** (`claim1_manifold.py`, numpy, CPU only, no GPU, no external/medical data) is attached as the artifact cell below.

**Rerun:**

    python claim1_manifold.py

Requires numpy only. Synthetic subspace manifold generated in-script; no downloads.


---
<!-- trackio-cell
{"type": "artifact", "id": "cell_d22a2ec4ce70", "created_at": "2026-07-20T14:06:40+00:00", "title": "Reproduction bundle", "artifact": "repro-plug-and-play-diffusion-meets-admm-dual-variable-coupling-for-robust-medical/repro-bundle:v0", "artifact_type": "dataset"}
-->
**📦 Artifact** `repro-plug-and-play-diffusion-meets-admm-dual-variable-coupling-for-robust-medical/repro-bundle:v0` · dataset

https://huggingface.co/buckets/kondratevakate/repro-plug-and-play-diffusion-meets-admm-dual-variable-coupling-for-robust-medical-artifacts#repro-plug-and-play-diffusion-meets-admm-dual-variable-coupling-for-robust-medical/repro-bundle:v0
