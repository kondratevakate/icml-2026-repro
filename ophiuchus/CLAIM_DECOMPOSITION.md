# Ophiuchus claim decomposition

- **C1 agent/tool loop:** the released system should parse interleaved thoughts,
  tool calls, observations, and answers; maintain multi-turn image state; and
  expose the declared SAM2, BiomedParse, and zoom-in tools.
- **C2 three-stage training:** the release should expose cold-start SFT,
  self-reflection fine-tuning, and ATRL with the paper's decomposed format,
  answer, and strategic tool-use rewards.
- **C3 64k trajectory corpus:** requires the complete tool-integrated dataset,
  construction provenance, image/mask references, and a frozen schema/count
  manifest.
- **C4 benchmark superiority:** requires the trained checkpoint, all eight
  benchmark inputs/splits, canonical evaluation code, baseline predictions,
  and repeated-run outputs.
- **C5 grounded segmentation and reflective tool behavior:** requires fresh
  detection/segmentation metrics, tool-transfer and ablation runs, and
  trajectory-level checks against held-out annotations.
