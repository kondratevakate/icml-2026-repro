# Frozen forecast

Frozen before installing dependencies, loading released checkpoints, or executing released code.

- Expected score: **5 / 10**
- Plausible range: **3-7 / 10**

## Rationale

The official repository contains the complete training, inference, Adaptive
CRP, LoRA, and EWC paths, plus an official Hugging Face checkpoint release.
The small modality-state checkpoint should permit an independent audit of the
reported task partition, and deterministic unit/smoke checks should cover the
core state-update mechanics without training CLIPSeg.

The 16 medical datasets, exact processed splits, metric outputs, run-level
seeds, and baseline artifacts are not bundled. The full reproduction requires
large external datasets, a CLIPSeg base model, and GPU training. The repository
also publishes a raw Conda environment export as `requirements.txt`, which is
unlikely to be directly installable with pip. Therefore the headline
performance, uncertainty, and ablation tables are unlikely to be independently
recoverable from the release alone.

## Expected claim scores

| Claim | Forecast |
|---|---:|
| Online semantic-modality discovery | 2 |
| Replay-free structure-aware continual learning | 2 |
| Headline continual-segmentation performance | 0 |
| Robustness and ablation evidence | 0 |
| Efficiency and clinical breadth | 1 |

