# GlobalHealthAtlas independent checks

Final prepared score: **6/12**, equal to the frozen forecast.

| Claim | Score | Independent result |
| --- | ---: | --- |
| C1 corpus scale and coverage | 1/2 | Detailed test-result exports exercise the paper's 15-domain/17-language schema, and most complete model exports sum to the reported 27,511 test instances. The linked artifacts do not contain the 280,210-row corpus or the more than 35,500-source inventory, so total scale and source grounding cannot be checked. |
| C2 composition and quality control | 1/2 | The paper's split counts and QA/SC totals are arithmetically self-consistent, and aggregate exports preserve task counts and difficulty breakdowns. The row-level corpus, 14,010 expert-audit rows, deduplication evidence, and quality-gate decisions are absent. |
| C3 evaluator validity and stability | 0/2 | The 61,380,432-byte evaluator adapter is a valid 288-tensor safetensors payload whose SHA-256 matches Hub LFS. This establishes artifact integrity, not the reported ICC 0.9735 or ten-run stability: the 100 expert labels, evaluator predictions, and repeated-run matrices are absent. |
| C4 multilingual/multi-domain benchmark | 2/2 | Independent weighted aggregation of 16 detailed per-model CSVs reconstructs all 16 corresponding paper Table 5 overall/QA/SC values within displayed rounding (maximum absolute error `0.000500`). |
| C5 SFT and transfer | 1/2 | All 15 incremental-SFT aggregate rows recompute with maximum error `4.71e-7`; every sampled SFT fraction beats base for 4B and 14B, but not for 8B. Eleven of 12 transfer cells match paper rounding. Qwen-8B base GPQA recomputes to `5.2034`, not paper Table 9's `5.342`, and the final two Qwen-14B rows are both labeled base in the CSV. |
| C6 robustness and leakage | 1/2 | All 24 robustness values reproduce Table 11 exactly; valid sample counts range from 694 to 1,000. All 10-gram leakage rates are zero and the largest 5-gram rate is 0.97%. Raw questions, perturbations, predictions, and seeds are not released, so these are cached-output checks. |

Additional findings:

- Both public LoRA payloads match their Hub LFS SHA-256 and have internally
  complete safetensors extents. The evaluator has 15,335,424 parameters; the
  Public-Model adapter has 21,823,488.
- Public-Model Hub configuration matches the paper's rank 8, alpha 16,
  zero-dropout, seven-target-module, two-epoch description.
- The only released `training/train_lora.sh` is not a faithful paper recipe:
  it names `distill_psychology-10k-r1`, rank 16, four attention targets,
  5,400 samples, and one epoch. It cannot regenerate the paper's Public-Model.
- Prompt construction and atomic JSON save/load pass a CPU-only smoke test.
  End-to-end inference was not attempted because it requires Qwen3-8B and
  CUDA/vLLM; the repository documents at least 80 GB VRAM for full loading.
- PDF pages 1, 6, 7, 8, 9, and 38 were rendered and visually inspected. The
  key tables are legible and the checked values agree with the TeX source.

