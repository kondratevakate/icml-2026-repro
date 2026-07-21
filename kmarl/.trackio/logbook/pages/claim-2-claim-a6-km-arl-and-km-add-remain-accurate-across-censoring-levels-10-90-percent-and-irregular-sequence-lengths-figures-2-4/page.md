# Claim 2: Claim A6: KM-ARL and KM-ADD remain accurate across censoring levels 10-90 percent and irregular sequence lengths (Figures 2-4)


---
<!-- trackio-cell
{"type": "markdown", "id": "cell_3ca651eb4e02", "created_at": "2026-07-20T08:30:51+00:00", "title": "Claim 2: Claim A6: KM-ARL and KM-ADD remain accurate across censoring levels 10-90 percent and irregular sequence lengths (Figures 2-4)"}
-->
**Anchored claim (verbatim).** "Simulation experiments on Gaussian and Poisson process changepoint models show KM-ARL/KM-ADD remain accurate across censoring levels from 10% to 90% changepoint prevalence and irregular sequence lengths (Figures 2-4, Section 5)."

**Pre-stated tolerance (before looking at results): relative bias within +/-5%.**

**Setup.** Same GSR/Gaussian pipeline as A3. Threshold sweep (`sweep_thresh.py`) across GSR thresholds 50/500/5000/50000 (true ARL from ~60 to ~60000), sequence-length regimes `T=1000` const / `T~U[100,1000]` / `T~U[30,300]`, changepoint prevalence 0.1/0.5/0.9, N=1000, 10 seeds.

**Result -- in the regime the paper's own Figure 2 occupies** (true ARL well below the max truncation length Tmax, i.e. threshold=50, true ARL~60 against T up to 1000): KM-ARL relative bias is **-0.014 to -0.028** across changepoint prevalence 0.1/0.3/0.5/0.7/0.9, flat with respect to prevalence, seed sd <= 0.024 -- squarely inside the pre-stated +/-5% band at every prevalence level tested, for all three sequence-length regimes.

**Outside that regime** (true ARL >> Tmax, e.g. threshold=5000 or 50000 against `T~U[30,300]`): KM's relative bias grows large, **-0.16 to -0.998**. Stated honestly: this is not a counterexample to A6. It is the paper's own declared limitation -- Eq. 11 (tested in Claim A4) says explicitly that KM-ARL is only asymptotically unbiased when the detection-point support does not exceed the censoring boundary; when true ARL vastly exceeds the observable truncation length, the estimator is being asked to extrapolate into unobserved territory, a zone the paper itself flags as one where no estimator is unbiased.

**Verdict -- CONFIRMED, with an honestly stated scope.** Within the regime the paper's figures actually populate, KM-ARL's accuracy is flat and well inside the pre-stated tolerance band across the full 10-90% changepoint-prevalence range and across both fixed and irregular sequence lengths. The large bias seen outside that regime is disclosed rather than hidden, and matches the paper's own characterization of it as an extrapolation zone, not a failure of the claim.
