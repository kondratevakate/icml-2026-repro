# DPsurv claim decomposition

Paper: **DPsurv: Dual-Prototype Evidential Fusion for Uncertainty-Aware and
Interpretable Whole-Slide Image Survival Prediction**

OpenReview ID: `RKqL4GYXz3`

arXiv: `2510.00053`

Each claim is worth two points.

1. **Five-cohort data and evaluation protocol.** The release implements the
   stated five-fold TCGA BRCA, BLCA, LUAD, UCEC, and KIRC disease-specific
   survival protocol with UNI2-h/PANTHER inputs and train-only preprocessing.
2. **Dual-prototype evidential fusion.** Patch-prototype GMM components,
   component-prototype experts, and prevalence-weighted evidence mixture form
   an executable DPsurv prediction path.
3. **GRFN survival bounds and mixture.** Belief/plausibility functions bound
   survival probabilities, mixture belief/plausibility are component-weighted
   sums, and the released implementation matches the paper interpolation
   controlled by lambda.
4. **Headline discrimination.** Released artifacts independently support the
   five-cohort mean C-index 0.704 and cohort values 0.720, 0.625, 0.667, 0.766,
   and 0.741.
5. **Calibration and uncertainty.** Released artifacts independently support
   mean IBS 0.310, mean IBLL 0.824, and the claim that BPIs improve coverage
   calibration over PPIs.
6. **Interpretability, ablations, and robustness.** Released artifacts support
   the component attribution mechanism and the reported ablation, sensitivity,
   alternate-feature-extractor, pathologist-coherence, and runtime findings.
