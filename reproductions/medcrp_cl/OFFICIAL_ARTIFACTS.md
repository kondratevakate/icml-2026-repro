# Official artifacts

Pinned before execution on 2026-07-29.

## Paper

- arXiv abstract: `https://arxiv.org/abs/2605.20297`
- arXiv PDF: `https://arxiv.org/pdf/2605.20297`
- arXiv source: `https://arxiv.org/e-print/2605.20297`
- PDF SHA-256: `028c03b341240fe34a760637e9b3ab887786182606194c3c0e0b1604453f91cd`
- source archive SHA-256: `690d738f26bf49810805953fada93fea27fe7906677fc32f10a21d8e236c44d2`

## Code

- repository: `https://github.com/zygao930/MedCRP-CL`
- pinned commit: `12d194632af8c2f0d7b51d1d0c5864e1e5ce93a5`

The audit uses the commit above, not the moving default branch.

## Checkpoint

- Hugging Face model: `clg-g/MedCRP-CL`
- pinned revision: `84ba00455749229c2cd478af4560c862fd832e48`
- `modality_state.pth`: 13,323 bytes, SHA-256 `d98e04242a7623c4530db76d1059cdafffa454380c5ae0b7d728bf8a632f5ed4`
- `ewc_state.pth`: 69,470,834 bytes, LFS SHA-256 `76fc4e7223718c47a5bb923174950ec1ff41e9d73474062eb77338b15b804e4c`
- `peft_model.pth`: 628,463,804 bytes, LFS SHA-256 `39017e2aa0684b9ab896e030e55a1149293de77df68cfd3c84813469b1e7c2da`

Only the small modality state is needed for the frozen local checks. The large
weight files are inventoried from the immutable Hub revision but are not
downloaded because the missing datasets and base-model environment already
block an end-to-end metric rerun.

## Scope

Official artifacts are evidence inputs, not ground truth. Paper claims are
scored only when the released artifacts permit an independent check. Public
leaderboards, third-party verdicts, and prior logbooks are excluded.

