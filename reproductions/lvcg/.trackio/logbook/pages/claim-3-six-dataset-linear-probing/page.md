# Claim 3: six-dataset linear probing


---
<!-- trackio-cell
{"type": "markdown", "id": "cell_bb42efacf11f", "created_at": "2026-07-30T07:29:41+00:00", "title": "Claim 3: six-dataset linear probing"}
-->
**PARTIAL - 1/2.** Released record and patient-ID splits are
disjoint and Table-1 averages are arithmetically consistent. No checkpoint or
results are released; the checkpoint override is ineffective; and PTB-XL
Sub-Class label order differs between train and validation/test.


---
<!-- trackio-cell
{"type": "code", "id": "cell_4779cbd37523", "created_at": "2026-07-30T07:29:41+00:00", "title": "Claim 3: six-dataset linear probing evidence", "language": "python"}
-->
````output
{
  "splits": {
    "tasks": {
      "chapman": {
        "rows": {
          "train": 16546,
          "val": 1860,
          "test": 4620
        },
        "record_key": "ecg_path",
        "record_overlap": {
          "train_val": 0,
          "train_test": 0,
          "val_test": 0
        },
        "patient_overlap": {},
        "duplicate_records": {
          "train": 0,
          "val": 0,
          "test": 0
        },
        "column_schema_equal": true,
        "label_columns": {
          "train": [
            "ecg_path",
            "age",
            "diagnose",
            "AQW",
            "UW",
            "SR",
            "WPW",
            "2AVB",
            "AT",
            "VB",
            "ARS",
            "STTC",
            "SA",
            "STE",
            "VPB",
            "TWO",
            "STTU",
            "ALS",
            "APB",
            "2AVB1",
            "PRIE",
            "CCR",
            "CR",
            "AF",
            "AVB",
            "QTIE",
            "LBBB",
            "VEB",
            "SVT",
            "RBBB",
            "1AVB",
            "STDD",
            "MI",
            "AFIB",
            "TWC",
            "PWC",
            "ERV",
            "RVH",
            "LVH",
            "ST",
            "JEB"
          ],
          "val": [
            "ecg_path",
            "age",
            "diagnose",
            "AQW",
            "UW",
            "SR",
            "WPW",
            "2AVB",
            "AT",
            "VB",
            "ARS",
            "STTC",
            "SA",
            "STE",
            "VPB",
            "TWO",
            "STTU",
            "ALS",
            "APB",
            "2AVB1",
            "PRIE",
            "CCR",
            "CR",
            "AF",
            "AVB",
            "QTIE",
            "LBBB",
            "VEB",
            "SVT",
            "RBBB",
            "1AVB",
            "STDD",
            "MI",
            "AFIB",
            "TWC",
            "PWC",
            "ERV",
            "RVH",
            "LVH",
            "ST",
            "JEB"
          ],
          "test": [
            "ecg_path",
            "age",
            "diagnose",
            "AQW",
            "UW",
            "SR",
            "WPW",
            "2AVB",
            "AT",
            "VB",
            "ARS",
            "STTC",
            "SA",
            "STE",
            "VPB",
            "TWO",
            "STTU",
            "ALS",
            "APB",
            "2AVB1",
            "PRIE",
            "CCR",
            "CR",
            "AF",
            "AVB",
            "QTIE",
            "LBBB",
            "VEB",
            "SVT",
            "RBBB",
            "1AVB",
            "STDD",
            "MI",
            "AFIB",
            "TWC",
            "PWC",
            "ERV",
            "RVH",
            "LVH",
            "ST",
            "JEB"
          ]
        },
        "leading_blank_line": {
          "train": false,
          "val": false,
          "test": true
        }
      },
      "icbeb": {
        "rows": {
          "train": 4950,
          "val": 551,
          "test": 1376
        },
        "record_key": "ecg_id",
        "record_overlap": {
          "train_val": 0,
          "train_test": 0,
          "val_test": 0
        },
        "patient_overlap": {
          "train_val": 0,
          "train_test": 0,
          "val_test": 0
        },
        "duplicate_records": {
          "train": 0,
          "val": 0,
          "test": 0
        },
        "column_schema_equal": true,
        "label_columns": {
          "train": [
            "patient_id",
            "ecg_id",
            "filename",
            "validation",
            "age",
            "sex",
            "scp_codes",
            "AFIB",
            "VPC",
            "NORM",
            "1AVB",
            "CRBBB",
            "STE",
            "PAC",
            "CLBBB",
            "STD"
          ],
          "val": [
            "patient_id",
            "ecg_id",
            "filename",
            "validation",
            "age",
            "sex",
            "scp_codes",
            "AFIB",
            "VPC",
            "NORM",
            "1AVB",
            "CRBBB",
            "STE",
            "PAC",
            "CLBBB",
            "STD"
          ],
          "test": [
            "patient_id",
            "ecg_id",
            "filename",
            "validation",
            "age",
            "sex",
            "scp_codes",
            "AFIB",
            "VPC",
            "NORM",
            "1AVB",
            "CRBBB",
            "STE",
            "PAC",
            "CLBBB",
            "STD"
          ]
        },
        "leading_blank_line": {
          "train": false,
          "val": false,
          "test": false
        }
      },
      "ptbxl_form": {
        "rows": {
          "train": 7197,
          "val": 901,
          "test": 880
        },
        "record_key": "ecg_id",
        "record_overlap": {
          "train_val": 0,
          "train_test": 0,
          "val_test": 0
        },
        "patient_overlap": {
          "train_val": 0,
          "train_test": 0,
          "val_test": 0
        },
        "duplicate_records": {
          "train": 0,
          "val": 0,
          "test": 0
        },
        "column_schema_equal": true,
        "label_columns": {
          "train": [
            "PAC",
            "DIG",
            "HVOLT",
            "STD",
            "LPR",
            "QWAVE",
            "VCLVH",
            "NDT",
            "TAB",
            "LOWT",
            "ABQRS",
            "LNGQT",
            "PRC(S)",
            "NT",
            "STE",
            "NST",
            "PVC",
            "INVT",
            "LVOLT"
          ],
          "val": [
            "PAC",
            "DIG",
            "HVOLT",
            "STD",
            "LPR",
            "QWAVE",
            "VCLVH",
            "NDT",
            "TAB",
            "LOWT",
            "ABQRS",
            "LNGQT",
            "PRC(S)",
            "NT",
            "STE",
            "NST",
            "PVC",
            "INVT",
            "LVOLT"
          ],
          "test": [
            "PAC",
            "DIG",
            "HVOLT",
            "STD",
            "LPR",
            "QWAVE",
            "VCLVH",
            "NDT",
            "TAB",
            "LOWT",
            "ABQRS",
            "LNGQT",
            "PRC(S)",
            "NT",
            "STE",
            "NST",
            "PVC",
            "INVT",
            "LVOLT"
          ]
        },
        "leading_blank_line": {
          "train": false,
          "val": false,
          "test": false
        }
      },
      "ptbxl_rhythm": {
        "rows": {
          "train": 16832,
          "val": 2100,
          "test": 2098
        },
        "record_key": "ecg_id",
        "record_overlap": {
          "train_val": 0,
          "train_test": 0,
          "val_test": 0
        },
        "patient_overlap": {
          "train_val": 0,
          "train_test": 0,
          "val_test": 0
        },
        "duplicate_records": {
          "train": 0,
          "val": 0,
          "test": 0
        },
        "column_schema_equal": true,
        "label_columns": {
          "train": [
            "SVARR",
            "BIGU",
            "STACH",
            "SARRH",
            "SBRAD",
            "TRIGU",
            "AFIB",
            "SR",
            "SVTAC",
            "PSVT",
            "AFLT",
            "PACE"
          ],
          "val": [
            "SVARR",
            "BIGU",
            "STACH",
            "SARRH",
            "SBRAD",
            "TRIGU",
            "AFIB",
            "SR",
            "SVTAC",
            "PSVT",
            "AFLT",
            "PACE"
          ],
          "test": [
            "SVARR",
            "BIGU",
            "STACH",
            "SARRH",
            "SBRAD",
            "TRIGU",
            "AFIB",
            "SR",
            "SVTAC",
            "PSVT",
            "AFLT",
            "PACE"
          ]
        },
        "leading_blank_line": {
          "train": false,
          "val": false,
          "test": false
        }
      },
      "ptbxl_sub_class": {
        "rows": {
          "train": 17084,
          "val": 2146,
          "test": 2158
        },
        "record_key": "ecg_id",
        "record_overlap": {
          "train_val": 0,
          "train_test": 0,
          "val_test": 0
        },
        "patient_overlap": {
          "train_val": 0,
          "train_test": 0,
          "val_test": 0
        },
        "duplicate_records": {
          "train": 0,
          "val": 0,
          "test": 0
        },
        "column_schema_equal": false,
        "label_columns": {
          "train": [
            "AMI",
            "LAFB/LPFB",
            "LVH",
            "STTC",
            "IMI",
            "SEHYP",
            "CRBBB",
            "WPW",
            "LAO/LAE",
            "NORM",
            "AVB",
            "RAO/RAE",
            "ISC",
            "LMI",
            "ISCI",
            "ISCA",
            "NST",
            "CLBBB",
            "ILBBB",
            "IRBBB",
            "PMI",
            "RVH",
            "IVCD"
          ],
          "val": [
            "AMI",
            "LAFB/LPFB",
            "LVH",
            "STTC",
            "IMI",
            "SEHYP",
            "CRBBB",
            "WPW",
            "LAO/LAE",
            "NORM",
            "ISC",
            "AVB",
            "RAO/RAE",
            "LMI",
            "ISCI",
            "ISCA",
            "NST",
            "CLBBB",
            "ILBBB",
            "IRBBB",
            "PMI",
            "RVH",
            "IVCD"
          ],
          "test": [
            "AMI",
            "LAFB/LPFB",
            "LVH",
            "STTC",
            "IMI",
            "SEHYP",
            "CRBBB",
            "WPW",
            "LAO/LAE",
            "NORM",
            "ISC",
            "AVB",
            "RAO/RAE",
            "LMI",
            "ISCI",
            "ISCA",
            "NST",
            "CLBBB",
            "ILBBB",
            "IRBBB",
            "PMI",
            "RVH",
            "IVCD"
          ]
        },
        "leading_blank_line": {
          "train": false,
          "val": false,
          "test": false
        }
      },
      "ptbxl_super_class": {
        "rows": {
          "train": 17084,
          "val": 2146,
          "test": 2158
        },
        "record_key": "ecg_id",
        "record_overlap": {
          "train_val": 0,
          "train_test": 0,
          "val_test": 0
        },
        "patient_overlap": {
          "train_val": 0,
          "train_test": 0,
          "val_test": 0
        },
        "duplicate_records": {
          "train": 0,
          "val": 0,
          "test": 0
        },
        "column_schema_equal": true,
        "label_columns": {
          "train": [
            "HYP",
            "NORM",
            "MI",
            "CD",
            "STTC"
          ],
          "val": [
            "HYP",
            "NORM",
            "MI",
            "CD",
            "STTC"
          ],
          "test": [
            "HYP",
            "NORM",
            "MI",
            "CD",
            "STTC"
          ]
        },
        "leading_blank_line": {
          "train": false,
          "val": false,
          "test": false
        }
      }
    },
    "all_record_splits_disjoint": true,
    "all_available_patient_splits_disjoint": true
  },
  "arithmetic": {
    "calculated_average": [
      67.30166666666668,
      74.54166666666667,
      80.46499999999999
    ],
    "printed_average": [
      67.3,
      74.54,
      80.47
    ],
    "absolute_difference": [
      0.0016666666666793617,
      0.0016666666666651508,
      0.005000000000009663
    ],
    "agrees_at_printed_precision": true,
    "cpsc_1pct_gain_over_heartlang_points": 10.650000000000006,
    "csn_1pct_gain_over_heartlang_points": 4.530000000000001
  }
}
````
