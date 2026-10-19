# MedCRP-CL claim decomposition

Paper: **MedCRP-CL: Continual Medical Image Segmentation via Bayesian Nonparametric Semantic Modality Discovery**

OpenReview ID: `v0DWbfP3b9`  
arXiv: `2605.20297v1`

Each claim is worth two points.

1. **Online semantic-modality discovery.** The released Adaptive CRP assigns the 16 sequential tasks to five semantically coherent modalities without a predefined cluster count.
2. **Replay-free structure-aware continual learning.** The release implements modality-specific LoRA parameter isolation and intra-modality EWC while retaining aggregate state rather than raw patient examples.
3. **Headline continual-segmentation performance.** Released artifacts independently support 73.3% mean Dice and 4.1% forgetting across 16 tasks, including the reported uncertainty.
4. **Robustness and ablation evidence.** Released artifacts independently support task-order, prompt-perturbation, component-ablation, and Fisher-weighted merge findings.
5. **Efficiency and clinical breadth.** Released artifacts independently support the 8.6M trainable-parameter count, six-fold reduction versus MoE-Adapters, and evaluation across the stated medical datasets and modalities.
