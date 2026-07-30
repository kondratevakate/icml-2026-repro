# Claim 5: SFT and transfer


---
<!-- trackio-cell
{"type": "markdown", "id": "cell_f7cf51992946", "created_at": "2026-07-30T06:50:26+00:00", "title": "Claim 5: SFT and transfer"}
-->
**PARTIAL — 1/2.** All 15 incremental-SFT rows recompute to
`4.71e-07` maximum error.
Eleven of 12 transfer cells match; Qwen-8B base GPQA recomputes to 5.2034
rather than paper 5.342, and the final Qwen-14B SFT rows are mislabeled as
base in the CSV.


---
<!-- trackio-cell
{"type": "code", "id": "cell_f95dd5f2c9b2", "created_at": "2026-07-30T06:50:26+00:00", "title": "Claim 5: SFT and transfer evidence", "language": "python"}
-->
````output
{
  "incremental": {
    "rows": 15,
    "max_abs_error": 4.707499998701792e-07,
    "all_sft_above_base": {
      "Qwen4": true,
      "Qwen8": false,
      "Qwen14": true
    },
    "rows_detail": [
      {
        "model": "Qwen4b00",
        "recomputed": 5.772659048072108,
        "released": 5.772659,
        "absolute_error": 4.8072108072005904e-08
      },
      {
        "model": "Qwen4b10",
        "recomputed": 6.2446666696666675,
        "released": 6.244667,
        "absolute_error": 3.3033333224352646e-07
      },
      {
        "model": "Qwen4b30",
        "recomputed": 6.312719147971958,
        "released": 6.312719,
        "absolute_error": 1.4797195735383184e-07
      },
      {
        "model": "Qwen4b60",
        "recomputed": 6.369416704333333,
        "released": 6.369417,
        "absolute_error": 2.9566666714941903e-07
      },
      {
        "model": "Qwen4b100",
        "recomputed": 6.38183347075,
        "released": 6.381833,
        "absolute_error": 4.707499998701792e-07
      },
      {
        "model": "Qwen8b00",
        "recomputed": 6.489567656651644,
        "released": 6.489568,
        "absolute_error": 3.433483559689421e-07
      },
      {
        "model": "Qwen8b10",
        "recomputed": 6.418459412706353,
        "released": 6.418459,
        "absolute_error": 4.1270635264822886e-07
      },
      {
        "model": "Qwen8b30",
        "recomputed": 6.49683172102718,
        "released": 6.496832,
        "absolute_error": 2.7897281995592493e-07
      },
      {
        "model": "Qwen8b60",
        "recomputed": 6.527861214714715,
        "released": 6.527861,
        "absolute_error": 2.1471471534084685e-07
      },
      {
        "model": "Qwen8b100",
        "recomputed": 6.531614918084751,
        "released": 6.531615,
        "absolute_error": 8.191524969447528e-08
      },
      {
        "model": "Qwen14b00",
        "recomputed": 6.133450185444834,
        "released": 6.13345,
        "absolute_error": 1.8544483371130127e-07
      },
      {
        "model": "Qwen14b10",
        "recomputed": 6.4423300494074445,
        "released": 6.44233,
        "absolute_error": 4.9407444357996155e-08
      },
      {
        "model": "Qwen14b30",
        "recomputed": 6.5520206122244495,
        "released": 6.552021,
        "absolute_error": 3.8777555033675526e-07
      },
      {
        "model": "Qwen14b60",
        "recomputed": 6.574787334250458,
        "released": 6.574787,
        "absolute_error": 3.3425045842250256e-07
      },
      {
        "model": "Qwen14b100",
        "recomputed": 6.5647824699849915,
        "released": 6.564782,
        "absolute_error": 4.699849913691878e-07
      }
    ]
  },
  "transfer": {
    "rows": 12,
    "max_abs_error_after_paper_rounding": 0.13862933333333238,
    "rows_matching_paper_rounding": 11,
    "duplicate_qwen14_base_label": true,
    "comparisons": [
      {
        "released_model_label": "Qwen-4b 原始",
        "paper_role": "Qwen-4b base",
        "dataset": "MMLU-Pro",
        "recomputed": 6.762477833333333,
        "paper": 6.762,
        "absolute_error": 0.00047783333333306643,
        "valid_count": 12008
      },
      {
        "released_model_label": "Qwen-4b 原始",
        "paper_role": "Qwen-4b base",
        "dataset": "GPQA",
        "recomputed": 4.948056833333333,
        "paper": 4.948,
        "absolute_error": 5.6833333332839686e-05,
        "valid_count": 446
      },
      {
        "released_model_label": "Qwen-4b 100%微调",
        "paper_role": "Qwen-4b SFT",
        "dataset": "MMLU-Pro",
        "recomputed": 7.042782166666666,
        "paper": 7.043,
        "absolute_error": 0.00021783333333402766,
        "valid_count": 11804
      },
      {
        "released_model_label": "Qwen-4b 100%微调",
        "paper_role": "Qwen-4b SFT",
        "dataset": "GPQA",
        "recomputed": 5.3419118333333335,
        "paper": 5.342,
        "absolute_error": 8.816666666611184e-05,
        "valid_count": 408
      },
      {
        "released_model_label": "Qwen-8b 原始",
        "paper_role": "Qwen-8b base",
        "dataset": "MMLU-Pro",
        "recomputed": 7.138956,
        "paper": 7.139,
        "absolute_error": 4.399999999993298e-05,
        "valid_count": 11999
      },
      {
        "released_model_label": "Qwen-8b 原始",
        "paper_role": "Qwen-8b base",
        "dataset": "GPQA",
        "recomputed": 5.203370666666667,
        "paper": 5.342,
        "absolute_error": 0.13862933333333238,
        "valid_count": 445
      },
      {
        "released_model_label": "Qwen-8b 100%微调",
        "paper_role": "Qwen-8b SFT",
        "dataset": "MMLU-Pro",
        "recomputed": 7.405004166666667,
        "paper": 7.405,
        "absolute_error": 4.166666666804986e-06,
        "valid_count": 11404
      },
      {
        "released_model_label": "Qwen-8b 100%微调",
        "paper_role": "Qwen-8b SFT",
        "dataset": "GPQA",
        "recomputed": 5.903557999999999,
        "paper": 5.904,
        "absolute_error": 0.00044200000000049755,
        "valid_count": 356
      },
      {
        "released_model_label": "Qwen-14b 原始",
        "paper_role": "Qwen-14b base",
        "dataset": "MMLU-Pro",
        "recomputed": 7.426010666666666,
        "paper": 7.426,
        "absolute_error": 1.0666666665493096e-05,
        "valid_count": 12013
      },
      {
        "released_model_label": "Qwen-14b 原始",
        "paper_role": "Qwen-14b base",
        "dataset": "GPQA",
        "recomputed": 5.277031833333333,
        "paper": 5.277,
        "absolute_error": 3.183333333289795e-05,
        "valid_count": 447
      },
      {
        "released_model_label": "Qwen-14b 原始",
        "paper_role": "Qwen-14b SFT",
        "dataset": "MMLU-Pro",
        "recomputed": 7.308666333333332,
        "paper": 7.309,
        "absolute_error": 0.0003336666666680088,
        "valid_count": 11412
      },
      {
        "released_model_label": "Qwen-14b 原始",
        "paper_role": "Qwen-14b SFT",
        "dataset": "GPQA",
        "recomputed": 5.722713833333334,
        "paper": 5.723,
        "absolute_error": 0.00028616666666625434,
        "valid_count": 339
      }
    ]
  }
}
````
