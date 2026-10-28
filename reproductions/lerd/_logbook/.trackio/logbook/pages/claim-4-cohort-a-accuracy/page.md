# Claim 4: Cohort A accuracy


---
<!-- trackio-cell
{"type": "markdown", "id": "cell_2b2ada4f7dad", "created_at": "2026-07-22T12:42:40+00:00", "title": "Claim 4: Cohort A accuracy"}
-->
**Paper claim (Table 2).** On AD Cohort A (ds004504, 88 participants: 36 AD,
23 FTD, 29 HC), LERD attains 75.03 +/- 8.29% accuracy and 72.69 +/- 8.16% F1,
outperforming LEAD (72.68%), ADFormer (69.35%), and other baselines.

**Verdict: not evaluated (compute-bound).**

A faithful pipeline is implemented (`lerd_eeg/`): 2 s non-overlapping windows
(T=1000) from the 500 Hz recordings, channel-wise z-score per window, C=19
electrodes, 5-fold cross-subject splits, subject-level majority vote, and the
EEGNet-style temporal-spatial encoder + classifier (the Table 4 "No prior"
variant), per Appendix F.

**Smoke test (1 fold, 3/30 epochs, CPU).** Pipeline correct: loss falls
1.06 -> 0.92, subject-level accuracy 63.16% / F1 50% (below the 30-epoch target,
as expected). Labels verified against `participants.tsv`: A=36, F=23, C=29.

**Blocker.** ~21 s per 1024-batch on an 8-core CPU (PyTorch depthwise/grouped
convolutions are unoptimized on CPU) => ~9 min/epoch => ~19 h for the full
5-fold x 30-epoch No-prior run alone; the dLIF/ERG priors (per-channel ODE Euler
substeps per window) add far more. Claim 4 therefore needs a GPU; the code runs
there unchanged (estimated ~20-40 min for No-prior, ~2-8 h for the full model on
an L4).

Dataset: [OpenNeuro ds004504](https://openneuro.org/datasets/ds004504) (downloaded
via the `mcp_openneuro` anonymous-S3 server in this bundle).
