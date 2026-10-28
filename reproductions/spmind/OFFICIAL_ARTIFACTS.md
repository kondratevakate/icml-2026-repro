# Official artifact inventory

Frozen on: 2026-07-29

| Artifact | Local path / identity | Status |
|---|---|---|
| Paper PDF | `official/paper.pdf` | Downloaded; SHA-256 `4CB694513EB6F2B3DC1E47E97E5A9BC53B013FEBE3E963455680802244E10F50` |
| arXiv source | `official/source.tar`, extracted to `official/arxiv-source/` | Downloaded; SHA-256 `2B38444FEA77FE5B03F74FC682771785294F63CE20909DFD58ADEAEE9E8CD041` |
| Official code | `official/code/` | Git commit `d5b889c3649fbdfd1489e5b2f60fc52a0d3ddbc6` |
| SP-Bench manifest | `official/code/benchmark/sp_bench.jsonl` | Present |
| Experiment evaluators | `official/code/experiments/` | Present |
| Official dataset | Hugging Face `tomyuanyucheng/spmind` | 79 files, 16,132,748,883 bytes from the repository tree; not downloaded in full |

## Pre-execution limitations found in official artifacts

- The paper experiments used `claude-sonnet-4-20250514`; the repository states
  that this model is retired and defaults to Claude Sonnet 4.5.
- SP-Bench inputs and ground truth are published, but the original per-query
  agent outputs and complete run traces are not.
- The repository describes SP-Bench execution one task at a time and does not
  include an automatic implementation of the paper’s success judge.
- CRC-CODEX quantification requires external source images plus generated
  quantification outputs; this bundle is not part of the published Hugging Face
  dataset.
- Cell-annotation scoring requires Qwen3-Embedding-8B and LLM-generated label
  descriptions in addition to prediction files.
