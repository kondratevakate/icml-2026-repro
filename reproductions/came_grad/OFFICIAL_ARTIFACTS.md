# CAME-Grad official artifact inventory

Frozen on: 2026-07-29

| Artifact | Identity | Status |
|---|---|---|
| Paper PDF | `official/paper.pdf` | SHA-256 `03D9097B1F9B5E39DE89703385586123A87DCBFBF574207A89ECE58A01F28295` |
| arXiv source | `official/source.tar` | SHA-256 `9F5109704136930A9453E0995AAD73EC6F794F51FF15C55E54A3F031369199D1` |
| Extracted source | `official/arxiv-source/` | `example_paper.tex` plus figures/style files |
| Official code | `official/code/` | commit `79059e39060d13ef6b6cb2ea9ad7a5d8e2519c83` |

## Release gaps observed before execution

- `modules/trainer.py` imports `modules.CAME_Grad`, but that module is absent.
- `main_train.py` and `main_test.py` import a top-level `dataset` package that
  is absent.
- README instructs `pip install -r requirements.txt`, but no requirements file
  exists.
- The CheXbert checkpoint path is the literal placeholder `xxxxxxxx`.
- No processed annotations, splits, weights, metrics outputs, or run logs are
  included.
- Git history contains no version of `modules/CAME_Grad.py`, the dataset
  package, or `requirements.txt`.
- The official README explicitly states that core optimizer source is
  temporarily withheld and lists weights/full release as TODO work.

These are release facts from the primary repository, not third-party
reproduction findings.
