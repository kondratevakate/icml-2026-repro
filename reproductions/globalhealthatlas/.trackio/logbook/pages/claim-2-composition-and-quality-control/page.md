# Claim 2: composition and quality control


---
<!-- trackio-cell
{"type": "markdown", "id": "cell_ad91fd9f3f71", "created_at": "2026-07-30T06:50:26+00:00", "title": "Claim 2: composition and quality control"}
-->
**PARTIAL — 1/2.** Aggregate exports preserve task and difficulty
breakdowns, and the paper's totals are internally consistent. Row-level corpus
data, 14,010 expert quality-audit rows, deduplication results, and quality-gate
decisions are absent.


---
<!-- trackio-cell
{"type": "code", "id": "cell_fd3d24eb33cc", "created_at": "2026-07-30T06:50:26+00:00", "title": "Claim 2: composition and quality control evidence", "language": "python"}
-->
````output
{
  "availability": {
    "github_revision": "23edda8517ed95e3a3db4fda0fc0fc53546532cb",
    "result_csv_count": 26,
    "result_csv_bytes": 354788,
    "exact_name_hf_dataset_search_result_count": 0,
    "raw_corpus_present_in_repo": false,
    "expert_gold_or_repeated_run_matrices_present": false
  },
  "limits": [
    "The 280,210 row corpus and 35,500 source-document inventory were not released in the linked artifacts.",
    "The 14,010 expert quality-audit rows were not released.",
    "The 100 expert/evaluator agreement rows and ten-run stability matrices were not released.",
    "Cached result tables lack raw prompts, predictions, perturbations, and generation seeds.",
    "Full inference needs Qwen3-8B plus CUDA/vLLM and was not attempted on this CPU host."
  ]
}
````
