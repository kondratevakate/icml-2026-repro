# Conclusion


---
<!-- trackio-cell
{"type": "markdown", "id": "cell_fd3ebd87917e", "created_at": "2026-07-22T12:46:43+00:00", "title": "Findings"}
-->
Two of LERD's six claims reproduce independently on CPU, and both hold.

**Supported (reproduced).**
- **Claim 3 (toy):** reproduced latent event-rate estimate matches LERD's reported Median Rate within 2-3.4%; a trajectory-only neural-ODE baseline scores boundary IoU ~0 at CS ~0.99, reproducing Table 1's high-CS/zero-IoU contrast. LERD's own IoU magnitudes (0.202-0.473) require the trained model and were not regenerated.
- **Claim 6 (real data, ds004504):** the AD spectral-slowing signature the claim rests on holds on all 88 subjects - AD central frequency below HC in 19/19 channels, theta elevated, alpha reduced, FTD intermediate.

**Not evaluable / not attempted.**
- **Claim 4** (75.03% accuracy): faithful pipeline built and smoke-tested (loss falls, 63% at 3/30 epochs) but CPU-infeasible (~19 h for the No-prior variant alone); needs a GPU.
- **Claim 5** (Cohort B): data not public (paper Data Availability 'Not applicable'; underlying EEG private). Cannot be reproduced or refuted.
- **Claims 1-2** (architecture, Theorem 4.1): descriptive / analytic; transcribed and found internally consistent; the numerical audit of Theorem 4.1 is deferred to the GPU follow-up.

**Reproducibility notes.** ds004504 was obtained via an anonymous-S3 MCP server (mcp_openneuro/, in the bundle). All reproduced results use numpy / mne / scipy on CPU and are deterministic (fixed seeds). Scripts: toy_repro.py, iou_toy.py, spectral_slowing.py, lerd_eeg/. Paper text was extracted from the PDF with PyMuPDF (not an LLM summary).
