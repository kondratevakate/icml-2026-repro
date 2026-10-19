# CAME-Grad frozen reproducibility forecast

Frozen on: 2026-07-29  
Paper: *The Double Dilemma in Multi-Task Radiology Report Generation: A
Gradient Dynamics Analysis and Solution*  
OpenReview: `T7y2wavrFM`  
arXiv: `2605.22635v2`

This forecast was written after primary-artifact inventory and claim anchoring,
but before any independent numerical implementation or experiment.
Leaderboards and third-party reproduction verdicts are excluded.

## Claim-level forecast

| Claim | Forecast | Frozen reason |
|---|---:|---|
| C1. CAME-Grad is a three-stage optimizer comprising trust-region direction rectification, magnitude enhancement, and adaptive fusion. | 2/2 possible for equation-level mechanics only. | Algorithm 1 and the equations are published, so an independent implementation can test trust-region, norm, fusion, limiting-case, and finite-gradient invariants. Backbone-agnostic clinical efficacy is not implied by a toy check. |
| C2. The method resolves the SDE “Double Dilemma,” guarantees geometric validity/global convergence stability, and injects escape energy toward flatter minima. | 0/2 expected. | No formal convergence theorem connects the deterministic update to flatter minima, and the released optimizer implementation is withheld. A local audit can assess equation consistency but not establish the causal optimization claim. |
| C3. Across eight RRG methods, average CE improves by 2.3% on MIMIC-CXR and 1.9% on IU X-Ray. | 0/2 expected. | Eight complete training pipelines, processed splits, checkpoints, seeds, and the optimizer source are absent. MIMIC-CXR additionally requires credentialed access and the paper used an A40 GPU. |
| C4. CAME-Grad outperforms seven alternative multi-task optimizers on the REVTAF backbone. | 0/2 expected. | Comparator implementations/configurations and the matching trained artifacts are not released in this repository. |
| C5. Stage ablations establish the contribution of direction rectification, energy injection, and adaptive fusion on PromptMRG/DDaTR and broader backbones. | 0/2 expected. | The optimizer source and ablation runner are absent; paper tables cannot substitute for fresh runs. |

## Frozen total

- Conservative expectation: **2/10**.
- Plausible range: **0–2/10**.
- Stretch ceiling without a new official release: **2/10**.
- Expected local effort: **1–3 hours** for deterministic equation and release
  audits; canonical training is out of local scope.

## Stop conditions

Do not claim numerical reproduction of C3–C5 if the core optimizer, matching
data pipeline, checkpoints, or run configuration remains absent. Do not count
TeX-table arithmetic as experimental reproduction. Do not infer clinical
generalization from synthetic gradients.
