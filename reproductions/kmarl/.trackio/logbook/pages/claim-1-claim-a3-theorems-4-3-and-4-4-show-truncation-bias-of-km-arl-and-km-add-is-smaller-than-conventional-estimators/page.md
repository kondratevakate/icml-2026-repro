# Claim 1: Claim A3: theorems 4.3 and 4.4 show truncation bias of KM-ARL and KM-ADD is smaller than conventional estimators


---
<!-- trackio-cell
{"type": "markdown", "id": "cell_004c06d8916e", "created_at": "2026-07-20T08:30:51+00:00", "title": "Claim 1: Claim A3: theorems 4.3 and 4.4 show truncation bias of KM-ARL and KM-ADD is smaller than conventional estimators"}
-->
**Anchored claim (verbatim).** "Theorems 4.3 and 4.4 show that the truncation bias of KM-ARL and KM-ADD is smaller than that of conventional (non-censoring-aware) ARL/ADD estimators."

**Setup.** GSR detector with ground-truth statistics on a Gaussian stream (pre-change mean 0, post-change mean 0.1, variance 0.1, so `log L(t) = x - 0.05` exactly). N=1000 sequences, uniform changepoint locations. Ground truth ARL/ADD from effectively-infinite sequences (paper's own method). Two thresholds: thr=500 -> true ARL 595.4, thr=50 -> true ARL 60.1 (lighter censoring). For each of 3 sequence-length regimes (`T=1000` const, `T~U[100,1000]`, `T~U[500,1000]`) and 5 changepoint prevalences (0.1/0.3/0.5/0.7/0.9), 10 seeds each -- 9 non-degenerate configurations in total (the all-changepoints-censored corner cases excluded). Code `kmarl_repro.py`.

**Result.** Both KM-ARL and LB-ARL estimators are negatively biased, as Theorems 4.3/4.4 require (`B_TR <= 0`), and **`|KM bias| < |LB bias|` holds in all 9 configurations, 10/10 seeds each** -- no exceptions. At the heaviest censoring configuration tested (irregular sequence lengths, high changepoint prevalence), relative bias KM = **-0.175** vs LB = **-0.522**: KM's truncation bias is roughly a third the size of the conventional estimator's. On the ADD side (geometric changepoints, Fig. 3-style, N=10000), KM-ADD beats LB-ADD (`|KM bias| < |LB bias|`) in **29 of 30 seed-runs** across the three ADD configurations tested.

**Verdict -- CONFIRMED.** Clean binary result across 10 seeds per configuration, no cherry-picking: every one of the 9 ARL configurations and 2 of 3 ADD configurations show KM strictly beating LB on 10/10 seeds, and the third ADD configuration wins 9/10.
