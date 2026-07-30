# Claim 6: robustness and leakage


---
<!-- trackio-cell
{"type": "markdown", "id": "cell_127faf8ffce6", "created_at": "2026-07-30T06:50:26+00:00", "title": "Claim 6: robustness and leakage"}
-->
**PARTIAL — 1/2.** All
24 robustness cells exactly match
Table 11; all 10-gram leakage rates are zero. The exports omit raw questions,
perturbations, predictions, and seeds, so end-to-end regeneration is not
possible from the release.


---
<!-- trackio-cell
{"type": "code", "id": "cell_2d63a18865ab", "created_at": "2026-07-30T06:50:26+00:00", "title": "Claim 6: robustness and leakage evidence", "language": "python"}
-->
````output
{
  "robustness": {
    "rows": 24,
    "max_abs_error": 0.0,
    "valid_count_range": [
      694,
      1000
    ],
    "comparisons": [
      {
        "model": "qwen3-8b 原始",
        "setting": "cross_lingual_score",
        "released": 0.8168,
        "paper": 0.8168,
        "absolute_error": 0.0,
        "valid_count": 928
      },
      {
        "model": "qwen3-8b 原始",
        "setting": "paraphrase_score",
        "released": 0.8391,
        "paper": 0.8391,
        "absolute_error": 0.0,
        "valid_count": 889
      },
      {
        "model": "qwen3-8b 原始",
        "setting": "Public_Benchmark1000",
        "released": 0.8541,
        "paper": 0.8541,
        "absolute_error": 0.0,
        "valid_count": 891
      },
      {
        "model": "qwen3-8b 原始",
        "setting": "Public_Benchmark_Noisy_score",
        "released": 0.7726,
        "paper": 0.7726,
        "absolute_error": 0.0,
        "valid_count": 950
      },
      {
        "model": "qwen3-8b 100%微调",
        "setting": "cross_lingual_score",
        "released": 0.844,
        "paper": 0.844,
        "absolute_error": 0.0,
        "valid_count": 1000
      },
      {
        "model": "qwen3-8b 100%微调",
        "setting": "paraphrase_score",
        "released": 0.86,
        "paper": 0.86,
        "absolute_error": 0.0,
        "valid_count": 1000
      },
      {
        "model": "qwen3-8b 100%微调",
        "setting": "Public_Benchmark1000",
        "released": 0.873,
        "paper": 0.873,
        "absolute_error": 0.0,
        "valid_count": 1000
      },
      {
        "model": "qwen3-8b 100%微调",
        "setting": "Public_Benchmark_Noisy_score",
        "released": 0.7734,
        "paper": 0.7734,
        "absolute_error": 0.0,
        "valid_count": 993
      },
      {
        "model": "ds-r1-distill-qwen",
        "setting": "cross_lingual_score",
        "released": 0.6152,
        "paper": 0.6152,
        "absolute_error": 0.0,
        "valid_count": 933
      },
      {
        "model": "ds-r1-distill-qwen",
        "setting": "paraphrase_score",
        "released": 0.494,
        "paper": 0.494,
        "absolute_error": 0.0,
        "valid_count": 923
      },
      {
        "model": "ds-r1-distill-qwen",
        "setting": "Public_Benchmark1000",
        "released": 0.4555,
        "paper": 0.4555,
        "absolute_error": 0.0,
        "valid_count": 898
      },
      {
        "model": "ds-r1-distill-qwen",
        "setting": "Public_Benchmark_Noisy_score",
        "released": 0.4,
        "paper": 0.4,
        "absolute_error": 0.0,
        "valid_count": 870
      },
      {
        "model": "claude",
        "setting": "cross_lingual_score",
        "released": 0.6589,
        "paper": 0.6589,
        "absolute_error": 0.0,
        "valid_count": 985
      },
      {
        "model": "claude",
        "setting": "paraphrase_score",
        "released": 0.58,
        "paper": 0.58,
        "absolute_error": 0.0,
        "valid_count": 988
      },
      {
        "model": "claude",
        "setting": "Public_Benchmark1000",
        "released": 0.6469,
        "paper": 0.6469,
        "absolute_error": 0.0,
        "valid_count": 994
      },
      {
        "model": "claude",
        "setting": "Public_Benchmark_Noisy_score",
        "released": 0.6458,
        "paper": 0.6458,
        "absolute_error": 0.0,
        "valid_count": 960
      },
      {
        "model": "Qwen3-14B",
        "setting": "cross_lingual_score",
        "released": 0.819,
        "paper": 0.819,
        "absolute_error": 0.0,
        "valid_count": 1000
      },
      {
        "model": "Qwen3-14B",
        "setting": "paraphrase_score",
        "released": 0.827,
        "paper": 0.827,
        "absolute_error": 0.0,
        "valid_count": 1000
      },
      {
        "model": "Qwen3-14B",
        "setting": "Public_Benchmark1000",
        "released": 0.833,
        "paper": 0.833,
        "absolute_error": 0.0,
        "valid_count": 1000
      },
      {
        "model": "Qwen3-14B",
        "setting": "Public_Benchmark_Noisy_score",
        "released": 0.771,
        "paper": 0.771,
        "absolute_error": 0.0,
        "valid_count": 1000
      },
      {
        "model": "llama3.1-8B",
        "setting": "cross_lingual_score",
        "released": 0.7309,
        "paper": 0.7309,
        "absolute_error": 0.0,
        "valid_count": 799
      },
      {
        "model": "llama3.1-8B",
        "setting": "paraphrase_score",
        "released": 0.7053,
        "paper": 0.7053,
        "absolute_error": 0.0,
        "valid_count": 733
      },
      {
        "model": "llama3.1-8B",
        "setting": "Public_Benchmark1000",
        "released": 0.7089,
        "paper": 0.7089,
        "absolute_error": 0.0,
        "valid_count": 694
      },
      {
        "model": "llama3.1-8B",
        "setting": "Public_Benchmark_Noisy_score",
        "released": 0.5,
        "paper": 0.5,
        "absolute_error": 0.0,
        "valid_count": 872
      }
    ]
  },
  "leakage": {
    "rows": 12,
    "nonzero_rows": [
      {
        "model": "grok3-mini-5",
        "prop1": 0.0005,
        "prop2": 1.0
      },
      {
        "model": "deepseekv3-5",
        "prop1": 0.0005,
        "prop2": 0.0
      },
      {
        "model": "qwen3-max-5",
        "prop1": 0.0097,
        "prop2": 0.5
      }
    ],
    "max_prop1": 0.0097,
    "all_10gram_zero": true
  }
}
````
