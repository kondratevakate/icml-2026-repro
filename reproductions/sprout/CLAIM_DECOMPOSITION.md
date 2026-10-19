# SPROUT claim decomposition

Paper: **Supervise Less, See More: Training-free Nuclear Instance Segmentation with Prototype-Guided Prompting**

OpenReview ID: `qYfNhYenuu`

arXiv: `2511.19953`

Each claim is worth two points.

1. **Fully training-free automatic prompting.** The release executes the complete image-to-prompts-to-instance-masks pipeline without annotations, fine-tuning, or parameter updates.
2. **Self-reference feature calibration.** H&E stain priors, per-image prototypes, and feature similarity maps provide an executable self-reference mechanism for foreground/background prompt generation.
3. **POT-Scan formulation and guarantee.** The progressive partial optimal transport implementation matches the paper formulation, and the slack-column reformulation is mathematically equivalent to the stated partial-OT problem.
4. **Containment-aware refinement.** The released NMS and point-sampling components independently support the claimed containment-aware mask refinement behavior.
5. **Benchmark performance.** Released artifacts independently support the reported MoNuSeg, CPM17, TNBC, and PanNuke instance-segmentation metrics, including the stated +8.2% AJI gain on MoNuSeg.
6. **Robustness and efficiency.** Released artifacts independently support backbone/SAM/hyperparameter robustness, runtime, and three-repeat empirical findings.
