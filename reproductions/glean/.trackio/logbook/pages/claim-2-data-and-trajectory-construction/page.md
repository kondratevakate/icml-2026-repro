# Claim 2: data and trajectory construction


---
<!-- trackio-cell
{"type": "markdown", "id": "cell_7364181fac44", "created_at": "2026-07-30T07:15:06+00:00", "title": "Claim 2: data and trajectory construction"}
-->
**PARTIAL - 1/2.** All official CDM v1.1 payload checksums match
and the pathology map has 2400 cases, including all
three GLEAN diseases. The release has
5960 radiology rows, one more than the official-page
summary. The paper-specific 4,000 trajectories, balancing decisions, labels,
and retrieved-guideline assignments are absent.


---
<!-- trackio-cell
{"type": "code", "id": "cell_f531195fc02a", "created_at": "2026-07-30T07:15:07+00:00", "title": "Claim 2: data and trajectory construction evidence", "language": "python"}
-->
````output
{
  "clinical_dataset": {
    "official_checksums": {
      "files_checked": 12,
      "all_match": true,
      "files": [
        {
          "file": "LICENSE.txt",
          "expected": "26794be6ea7916cd4b4017b45b9c34c3c31f645a03a9cd6c591c529b92435042",
          "actual": "26794be6ea7916cd4b4017b45b9c34c3c31f645a03a9cd6c591c529b92435042",
          "matches": true
        },
        {
          "file": "discharge_diagnosis.csv",
          "expected": "bf7f1b251e9a6142e75ff262d3876912dd1cc50a075dfff3db0feea852addb20",
          "actual": "bf7f1b251e9a6142e75ff262d3876912dd1cc50a075dfff3db0feea852addb20",
          "matches": true
        },
        {
          "file": "discharge_procedures.csv",
          "expected": "69093a7ac9b52885aba9ef0d4394a967034dd01f976e4a7f20968a5d9819261d",
          "actual": "69093a7ac9b52885aba9ef0d4394a967034dd01f976e4a7f20968a5d9819261d",
          "matches": true
        },
        {
          "file": "history_of_present_illness.csv",
          "expected": "ea66220f7fb184a6890a73b33d1b78406efaeca843ab83b2aa11071bc1e14a46",
          "actual": "ea66220f7fb184a6890a73b33d1b78406efaeca843ab83b2aa11071bc1e14a46",
          "matches": true
        },
        {
          "file": "icd_diagnosis.csv",
          "expected": "8eb3009bc9bf61040f11d4359f16e120bf82261020a6a66266cf1555aa3bc825",
          "actual": "8eb3009bc9bf61040f11d4359f16e120bf82261020a6a66266cf1555aa3bc825",
          "matches": true
        },
        {
          "file": "icd_procedures.csv",
          "expected": "431e80b46443d11b2c94ee6650c65eddecc52f5399fb51f69ac90d7ed604fcaa",
          "actual": "431e80b46443d11b2c94ee6650c65eddecc52f5399fb51f69ac90d7ed604fcaa",
          "matches": true
        },
        {
          "file": "lab_test_mapping.csv",
          "expected": "dcf3b693b86f4f3a19af89f2724d9e8bca4753df1a8bfdf74977e9db35a52b65",
          "actual": "dcf3b693b86f4f3a19af89f2724d9e8bca4753df1a8bfdf74977e9db35a52b65",
          "matches": true
        },
        {
          "file": "laboratory_tests.csv",
          "expected": "c1e23ee4ae87ae1031970b943cb9bb3bf5857f93208f32efddc8799f6728d7aa",
          "actual": "c1e23ee4ae87ae1031970b943cb9bb3bf5857f93208f32efddc8799f6728d7aa",
          "matches": true
        },
        {
          "file": "microbiology.csv",
          "expected": "34d02fbba574f67b327c685d8d9ae866b3478b58a4d68ea930f8e21671fe4c92",
          "actual": "34d02fbba574f67b327c685d8d9ae866b3478b58a4d68ea930f8e21671fe4c92",
          "matches": true
        },
        {
          "file": "pathology_ids.json",
          "expected": "3a1b91912e551c5538905ce3a7eacbbcd498b0343abe004ec80d739828c668fb",
          "actual": "3a1b91912e551c5538905ce3a7eacbbcd498b0343abe004ec80d739828c668fb",
          "matches": true
        },
        {
          "file": "physical_examination.csv",
          "expected": "f8285b4c8b658efc850ca8e6248e468912a2a330ea4e6cda2fbc9575b46c47a9",
          "actual": "f8285b4c8b658efc850ca8e6248e468912a2a330ea4e6cda2fbc9575b46c47a9",
          "matches": true
        },
        {
          "file": "radiology_reports.csv",
          "expected": "a535c7efb332252bec29d22b8ee56968a40319dad9fee87e9ef4ffa369ee3ea9",
          "actual": "a535c7efb332252bec29d22b8ee56968a40319dad9fee87e9ef4ffa369ee3ea9",
          "matches": true
        }
      ]
    },
    "pathology_counts": {
      "appendicitis": 957,
      "cholecystitis": 648,
      "diverticulitis": 257,
      "pancreatitis": 538
    },
    "pathology_total": 2400,
    "glean_diseases_present": true,
    "csv_inventory": {
      "discharge_diagnosis.csv": {
        "rows": 2400,
        "columns": [
          "hadm_id",
          "discharge_diagnosis"
        ],
        "bytes": 137874
      },
      "discharge_procedures.csv": {
        "rows": 2122,
        "columns": [
          "hadm_id",
          "discharge_procedure"
        ],
        "bytes": 92166
      },
      "history_of_present_illness.csv": {
        "rows": 2400,
        "columns": [
          "hadm_id",
          "hpi"
        ],
        "bytes": 2622847
      },
      "icd_diagnosis.csv": {
        "rows": 17357,
        "columns": [
          "hadm_id",
          "icd_diagnosis"
        ],
        "bytes": 941628
      },
      "icd_procedures.csv": {
        "rows": 2917,
        "columns": [
          "hadm_id",
          "icd_code",
          "icd_title",
          "icd_version"
        ],
        "bytes": 182558
      },
      "lab_test_mapping.csv": {
        "rows": 1209,
        "columns": [
          "itemid",
          "label",
          "fluid",
          "category",
          "count",
          "corresponding_ids"
        ],
        "bytes": 76322
      },
      "laboratory_tests.csv": {
        "rows": 138788,
        "columns": [
          "hadm_id",
          "itemid",
          "valuestr",
          "ref_range_lower",
          "ref_range_upper"
        ],
        "bytes": 5499890
      },
      "microbiology.csv": {
        "rows": 4403,
        "columns": [
          "hadm_id",
          "test_itemid",
          "valuestr",
          "spec_itemid"
        ],
        "bytes": 275253
      },
      "physical_examination.csv": {
        "rows": 2400,
        "columns": [
          "hadm_id",
          "pe"
        ],
        "bytes": 999636
      },
      "radiology_reports.csv": {
        "rows": 5960,
        "columns": [
          "hadm_id",
          "note_id",
          "modality",
          "region",
          "exam_name",
          "text"
        ],
        "bytes": 7193002
      }
    },
    "radiology": {
      "rows": 5960,
      "unique_note_ids": 5960,
      "empty_text_rows": 0,
      "official_page_claim": 5959,
      "difference_from_official_page": 1
    }
  },
  "guidelines": {
    "size": {
      "size": {
        "dataset": {
          "dataset": "epfl-llm/guidelines",
          "num_bytes_original_files": 877724423,
          "num_bytes_parquet_files": 424634397,
          "num_bytes_memory": 865223621,
          "num_rows": 37970,
          "estimated_num_rows": null
        },
        "configs": [
          {
            "dataset": "epfl-llm/guidelines",
            "config": "default",
            "num_bytes_original_files": 877724423,
            "num_bytes_parquet_files": 424634397,
            "num_bytes_memory": 865223621,
            "num_rows": 37970,
            "num_columns": 7,
            "estimated_num_rows": null
          }
        ],
        "splits": [
          {
            "dataset": "epfl-llm/guidelines",
            "config": "default",
            "split": "train",
            "num_bytes_parquet_files": 424634397,
            "num_bytes_memory": 865223621,
            "num_rows": 37970,
            "num_columns": 7,
            "estimated_num_rows": null
          }
        ]
      },
      "pending": [],
      "failed": [],
      "partial": false
    },
    "splits": {
      "splits": [
        {
          "dataset": "epfl-llm/guidelines",
          "config": "default",
          "split": "train"
        }
      ],
      "pending": [],
      "failed": []
    },
    "validity": {
      "preview": true,
      "viewer": true,
      "search": true,
      "filter": true,
      "statistics": true
    }
  },
  "notes_availability": {
    "available": true,
    "tables": {
      "discharge.csv.gz": {
        "available": true,
        "bytes": 1139203149,
        "columns": [
          "note_id",
          "subject_id",
          "hadm_id",
          "note_type",
          "note_seq",
          "charttime",
          "storetime",
          "text"
        ]
      },
      "radiology.csv.gz": {
        "available": true,
        "bytes": 781830996,
        "columns": [
          "note_id",
          "subject_id",
          "hadm_id",
          "note_type",
          "note_seq",
          "charttime",
          "storetime",
          "text"
        ]
      }
    }
  }
}
````
