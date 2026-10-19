# Claim 4: multilingual benchmark


---
<!-- trackio-cell
{"type": "markdown", "id": "cell_25f8ca609099", "created_at": "2026-07-30T06:50:26+00:00", "title": "Claim 4: multilingual benchmark"}
-->
**SUPPORTED — 2/2.** Independent count-weighted aggregation of
16 detailed per-model CSVs reconstructs all matching
overall, QA, and SC paper values within displayed rounding. Maximum absolute
error is `0.000500`.


---
<!-- trackio-cell
{"type": "code", "id": "cell_00abb5ab83f8", "created_at": "2026-07-30T06:50:26+00:00", "title": "Claim 4: multilingual benchmark evidence", "language": "python"}
-->
````output
{
  "models_compared": 16,
  "max_score_abs_error": 0.000499568889751778,
  "rows_matching_paper_rounding": 16,
  "comparisons": [
    {
      "file": "Qwen2_5_7Binstruct_detailed.csv",
      "table_model": "qwen2.5_7binstruct",
      "reconstructed": {
        "labels": {
          "Question-Answer": {
            "count": 13584,
            "dimension_means": {
              "Accuracy": 2.09113663128644,
              "Completeness": 2.586425206110351,
              "Consensus Alignment": 3.8554181390937887,
              "Terminology Norms": 3.654299175496615,
              "Insightfulness": 2.233583627705683
            },
            "omitted_dimensions": [
              "Reasoning"
            ],
            "mean": 2.8841725559385756
          },
          "Single-Choice": {
            "count": 13916,
            "dimension_means": {
              "Accuracy": 8.232250646676269,
              "Completeness": 3.5558350100634506,
              "Consensus Alignment": 6.519545846367206,
              "Terminology Norms": 1.9083788444191576,
              "Insightfulness": 1.8730957172201776
            },
            "omitted_dimensions": [
              "Reasoning"
            ],
            "mean": 4.417821212949252
          }
        },
        "overall": 3.6602545454280513,
        "count": 27500
      },
      "table": {
        "sc": 4.418,
        "qa": 2.884,
        "overall": 3.66
      },
      "absolute_errors": {
        "qa": 0.0001725559385756803,
        "sc": 0.0001787870507481415,
        "overall": 0.000254545428051145
      }
    },
    {
      "file": "Qwen2_5_72binstruct_detailed.csv",
      "table_model": "qwen2.5_72binstruct",
      "reconstructed": {
        "labels": {
          "Question-Answer": {
            "count": 13584,
            "dimension_means": {
              "Accuracy": 4.008613074135899,
              "Completeness": 4.170053003484247,
              "Consensus Alignment": 6.602252650019213,
              "Terminology Norms": 6.62433745587883,
              "Insightfulness": 4.127502944663649
            },
            "omitted_dimensions": [
              "Reasoning"
            ],
            "mean": 5.106551825636368
          },
          "Single-Choice": {
            "count": 13916,
            "dimension_means": {
              "Accuracy": 8.631790744571644,
              "Completeness": 3.659600459931804,
              "Consensus Alignment": 6.776013222293185,
              "Terminology Norms": 1.9324518540845068,
              "Insightfulness": 1.8926415635676206
            },
            "omitted_dimensions": [
              "Reasoning"
            ],
            "mean": 4.578499568889752
          }
        },
        "overall": 4.839338181822335,
        "count": 27500
      },
      "table": {
        "sc": 4.578,
        "qa": 5.107,
        "overall": 4.839
      },
      "absolute_errors": {
        "qa": 0.00044817436363242535,
        "sc": 0.000499568889751778,
        "overall": 0.0003381818223342492
      }
    },
    {
      "file": "deepseek_distill_qwen_32b_detailed.csv",
      "table_model": "deepseek-r1-distill-qwen-32b",
      "reconstructed": {
        "labels": {
          "Question-Answer": {
            "count": 13575,
            "dimension_means": {
              "Accuracy": 3.9510128914270353,
              "Reasoning": 4.950497237547182,
              "Completeness": 4.209355432787402,
              "Consensus Alignment": 6.512044198962721,
              "Terminology Norms": 6.775543278184604,
              "Insightfulness": 4.89406998158512
            },
            "omitted_dimensions": [],
            "mean": 5.215420503415677
          },
          "Single-Choice": {
            "count": 13902,
            "dimension_means": {
              "Accuracy": 8.458495180431086,
              "Reasoning": 6.9989929507015525,
              "Completeness": 6.381312041484895,
              "Consensus Alignment": 8.381240109399949,
              "Terminology Norms": 6.9426701193508125,
              "Insightfulness": 6.261329305041003
            },
            "omitted_dimensions": [],
            "mean": 7.237339951068216
          }
        },
        "overall": 6.2384115199482535,
        "count": 27477
      },
      "table": {
        "sc": 7.237,
        "qa": 5.215,
        "overall": 6.238
      },
      "absolute_errors": {
        "qa": 0.0004205034156772314,
        "sc": 0.00033995106821560483,
        "overall": 0.0004115199482530585
      }
    },
    {
      "file": "deepseek_distill_qwen_8b_detailed.csv",
      "table_model": "deepSeek_r1_distill_qwen_8B",
      "reconstructed": {
        "labels": {
          "Question-Answer": {
            "count": 11344,
            "dimension_means": {
              "Accuracy": 3.4617418900270636,
              "Reasoning": 4.548307475316116,
              "Completeness": 3.791960507826429,
              "Consensus Alignment": 6.095733427350759,
              "Terminology Norms": 6.508727080472497,
              "Insightfulness": 4.631435119893249
            },
            "omitted_dimensions": [],
            "mean": 4.839650916814352
          },
          "Single-Choice": {
            "count": 11580,
            "dimension_means": {
              "Accuracy": 7.697754749658118,
              "Reasoning": 6.501381692519173,
              "Completeness": 6.029792746191624,
              "Consensus Alignment": 7.82271157168178,
              "Terminology Norms": 6.78963730578921,
              "Insightfulness": 5.86934369607599
            },
            "omitted_dimensions": [],
            "mean": 6.785103626985982
          }
        },
        "overall": 5.822391380249506,
        "count": 22924
      },
      "table": {
        "sc": 6.785,
        "qa": 4.84,
        "overall": 5.822
      },
      "absolute_errors": {
        "qa": 0.0003490831856476362,
        "sc": 0.00010362698598154196,
        "overall": 0.0003913802495061347
      }
    },
    {
      "file": "deepseek_r1_detailed.csv",
      "table_model": "deepseek_r1",
      "reconstructed": {
        "labels": {
          "Question-Answer": {
            "count": 13571,
            "dimension_means": {
              "Accuracy": 4.288188048007441,
              "Reasoning": 5.128877754047603,
              "Completeness": 4.418686905952986,
              "Consensus Alignment": 6.734581091997346,
              "Terminology Norms": 6.935671652857931,
              "Insightfulness": 4.541301304246924
            },
            "omitted_dimensions": [],
            "mean": 5.341217792851705
          },
          "Single-Choice": {
            "count": 13868,
            "dimension_means": {
              "Accuracy": 8.753965964718851,
              "Reasoning": 6.931136429226058,
              "Completeness": 6.343885203258363,
              "Consensus Alignment": 8.717262763178901,
              "Terminology Norms": 7.2210124026083795,
              "Insightfulness": 5.880227862728872
            },
            "omitted_dimensions": [],
            "mean": 7.307915104286571
          }
        },
        "overall": 6.3352102238797565,
        "count": 27439
      },
      "table": {
        "sc": 7.308,
        "qa": 5.341,
        "overall": 6.335
      },
      "absolute_errors": {
        "qa": 0.00021779285170442364,
        "sc": 8.48957134289563e-05,
        "overall": 0.00021022387975655477
      }
    },
    {
      "file": "deepseek_v3_2_detailed.csv",
      "table_model": "deepSeek-v3.2",
      "reconstructed": {
        "labels": {
          "Question-Answer": {
            "count": 13575,
            "dimension_means": {
              "Accuracy": 4.691491712675875,
              "Reasoning": 4.958600368347253,
              "Completeness": 4.740257826922945,
              "Consensus Alignment": 7.09907918977385,
              "Terminology Norms": 7.303057090137463,
              "Insightfulness": 4.620257826857459
            },
            "omitted_dimensions": [],
            "mean": 5.568790669119141
          },
          "Single-Choice": {
            "count": 13903,
            "dimension_means": {
              "Accuracy": 8.899518089638136,
              "Reasoning": 6.318564338789976,
              "Completeness": 6.032223261142775,
              "Consensus Alignment": 8.902898654996115,
              "Terminology Norms": 7.201251528452926,
              "Insightfulness": 5.498669351891463
            },
            "omitted_dimensions": [],
            "mean": 7.142187537485232
          }
        },
        "overall": 6.364879782624263,
        "count": 27478
      },
      "table": {
        "sc": 7.142,
        "qa": 5.569,
        "overall": 6.365
      },
      "absolute_errors": {
        "qa": 0.00020933088085861584,
        "sc": 0.00018753748523181457,
        "overall": 0.00012021737573686408
      }
    },
    {
      "file": "claude_sonnet_thinking_detailed.csv",
      "table_model": "claude_sonnet_thinking",
      "reconstructed": {
        "labels": {
          "Question-Answer": {
            "count": 13391,
            "dimension_means": {
              "Accuracy": 4.987304906120976,
              "Reasoning": 5.5649316706302026,
              "Completeness": 5.130236726240758,
              "Consensus Alignment": 7.576506608933759,
              "Terminology Norms": 7.68090508557344,
              "Insightfulness": 5.342095437139198
            },
            "omitted_dimensions": [],
            "mean": 6.0469967391063895
          },
          "Single-Choice": {
            "count": 13819,
            "dimension_means": {
              "Accuracy": 5.129097619208407,
              "Reasoning": 7.125117591763947,
              "Completeness": 6.665315869549098,
              "Consensus Alignment": 8.875244228951587,
              "Terminology Norms": 7.922715102162744,
              "Insightfulness": 6.316882552922716
            },
            "omitted_dimensions": [],
            "mean": 7.005728827426416
          }
        },
        "overall": 6.533902976831286,
        "count": 27210
      },
      "table": {
        "sc": 7.006,
        "qa": 6.047,
        "overall": 6.534
      },
      "absolute_errors": {
        "qa": 3.260893610246285e-06,
        "sc": 0.0002711725735844439,
        "overall": 9.7023168713406e-05
      }
    },
    {
      "file": "Qwen3_8B_detailed.csv",
      "table_model": "Qwen3_8B",
      "reconstructed": {
        "labels": {
          "Question-Answer": {
            "count": 13572,
            "dimension_means": {
              "Accuracy": 3.639257294449012,
              "Reasoning": 4.378573533862732,
              "Completeness": 3.9764220454473196,
              "Consensus Alignment": 6.199896846436189,
              "Terminology Norms": 6.671676982013927,
              "Insightfulness": 4.261567934103669
            },
            "omitted_dimensions": [],
            "mean": 4.854565772718809
          },
          "Single-Choice": {
            "count": 13906,
            "dimension_means": {
              "Accuracy": 8.383431612263484,
              "Reasoning": 6.69502373082252,
              "Completeness": 6.018337408270893,
              "Consensus Alignment": 8.205594707247592,
              "Terminology Norms": 6.789515317129871,
              "Insightfulness": 5.829138501399184
            },
            "omitted_dimensions": [],
            "mean": 6.986840212855591
          }
        },
        "overall": 5.933662081203491,
        "count": 27478
      },
      "table": {
        "sc": 6.987,
        "qa": 4.855,
        "overall": 5.934
      },
      "absolute_errors": {
        "qa": 0.00043422728119146825,
        "sc": 0.00015978714440922914,
        "overall": 0.00033791879650912904
      }
    },
    {
      "file": "Qwen3_32B_detailed.csv",
      "table_model": "qwen3_32b",
      "reconstructed": {
        "labels": {
          "Question-Answer": {
            "count": 13587,
            "dimension_means": {
              "Accuracy": 2.8920291455249862,
              "Reasoning": 3.8906307500067716,
              "Completeness": 3.3877971589907263,
              "Consensus Alignment": 5.09472289678104,
              "Terminology Norms": 5.801133436330612,
              "Insightfulness": 3.7451240155277854
            },
            "omitted_dimensions": [],
            "mean": 4.135239567193653
          },
          "Single-Choice": {
            "count": 13908,
            "dimension_means": {
              "Accuracy": 7.1570319240565095,
              "Reasoning": 6.01610583828444,
              "Completeness": 5.708800690410195,
              "Consensus Alignment": 7.236266896746117,
              "Terminology Norms": 6.567299395981667,
              "Insightfulness": 5.454989933764669
            },
            "omitted_dimensions": [],
            "mean": 6.356749113207265
          }
        },
        "overall": 5.258962235531799,
        "count": 27495
      },
      "table": {
        "sc": 6.357,
        "qa": 4.135,
        "overall": 5.259
      },
      "absolute_errors": {
        "qa": 0.00023956719365347823,
        "sc": 0.0002508867927355496,
        "overall": 3.776446820147328e-05
      }
    },
    {
      "file": "gemini_3_flash_preview_thinking_detailed.csv",
      "table_model": "gemini-3-flash-preview-thinking",
      "reconstructed": {
        "labels": {
          "Question-Answer": {
            "count": 12862,
            "dimension_means": {
              "Accuracy": 4.8247550924794735,
              "Reasoning": 5.3929404446613285,
              "Completeness": 4.9433991602589025,
              "Consensus Alignment": 7.3820556678242095,
              "Terminology Norms": 7.432436635166847,
              "Insightfulness": 5.491136681678121
            },
            "omitted_dimensions": [],
            "mean": 5.911120613678148
          },
          "Single-Choice": {
            "count": 13901,
            "dimension_means": {
              "Accuracy": 8.381411409243938,
              "Reasoning": 6.038126753493776,
              "Completeness": 5.2992590462214215,
              "Consensus Alignment": 8.180418674856556,
              "Terminology Norms": 6.680023019992663,
              "Insightfulness": 5.17200201419912
            },
            "omitted_dimensions": [],
            "mean": 6.625206819667913
          }
        },
        "overall": 6.282024934922541,
        "count": 26763
      },
      "table": {
        "sc": 6.625,
        "qa": 5.911,
        "overall": 6.282
      },
      "absolute_errors": {
        "qa": 0.0001206136781481959,
        "sc": 0.00020681966791258333,
        "overall": 2.493492254096452e-05
      }
    },
    {
      "file": "qwq_32B_detailed.csv",
      "table_model": "qwq_32B",
      "reconstructed": {
        "labels": {
          "Question-Answer": {
            "count": 13509,
            "dimension_means": {
              "Accuracy": 3.907395069960768,
              "Reasoning": 4.860389370083349,
              "Completeness": 4.242208897729219,
              "Consensus Alignment": 6.589162780447112,
              "Terminology Norms": 6.884225331308832,
              "Insightfulness": 4.681175512665115
            },
            "omitted_dimensions": [],
            "mean": 5.194092827032399
          },
          "Single-Choice": {
            "count": 13862,
            "dimension_means": {
              "Accuracy": 8.587505410528278,
              "Reasoning": 6.855215697508295,
              "Completeness": 6.143557927954193,
              "Consensus Alignment": 8.381402394975112,
              "Terminology Norms": 6.839128552837541,
              "Insightfulness": 5.952604241741522
            },
            "omitted_dimensions": [],
            "mean": 7.126569037590824
          }
        },
        "overall": 6.172792371468513,
        "count": 27371
      },
      "table": {
        "sc": 7.127,
        "qa": 5.194,
        "overall": 6.173
      },
      "absolute_errors": {
        "qa": 9.282703239943402e-05,
        "sc": 0.0004309624091760611,
        "overall": 0.00020762853148692528
      }
    },
    {
      "file": "kimi_k2_thinking_detailed.csv",
      "table_model": "kimi-k2-thinking",
      "reconstructed": {
        "labels": {
          "Question-Answer": {
            "count": 13430,
            "dimension_means": {
              "Accuracy": 4.948026805616902,
              "Reasoning": 5.553909158645572,
              "Completeness": 5.0585256888049885,
              "Consensus Alignment": 7.33082650788049,
              "Terminology Norms": 7.711392404998735,
              "Insightfulness": 4.901712583778927
            },
            "omitted_dimensions": [],
            "mean": 5.917398858287602
          },
          "Single-Choice": {
            "count": 13858,
            "dimension_means": {
              "Accuracy": 8.837566748371916,
              "Reasoning": 7.619930725882016,
              "Completeness": 7.301991629328401,
              "Consensus Alignment": 9.145114735087532,
              "Terminology Norms": 8.061408572702337,
              "Insightfulness": 6.503680184868232
            },
            "omitted_dimensions": [],
            "mean": 7.911615432706739
          }
        },
        "overall": 6.930146340268708,
        "count": 27288
      },
      "table": {
        "sc": 7.912,
        "qa": 5.917,
        "overall": 6.93
      },
      "absolute_errors": {
        "qa": 0.0003988582876024438,
        "sc": 0.00038456729326075134,
        "overall": 0.00014634026870830041
      }
    },
    {
      "file": "glm_4_7_detailed.csv",
      "table_model": "glm-4.7",
      "reconstructed": {
        "labels": {
          "Question-Answer": {
            "count": 13492,
            "dimension_means": {
              "Accuracy": 4.7969908093866005,
              "Reasoning": 6.575378001849614,
              "Completeness": 4.957975096354875,
              "Consensus Alignment": 7.47643047728758,
              "Terminology Norms": 7.802846131100281,
              "Insightfulness": 5.298695523388228
            },
            "omitted_dimensions": [],
            "mean": 6.1513860065611965
          },
          "Single-Choice": {
            "count": 13682,
            "dimension_means": {
              "Accuracy": 8.692442625310628,
              "Reasoning": 8.559348048456878,
              "Completeness": 8.224528577748645,
              "Consensus Alignment": 9.327291331764657,
              "Terminology Norms": 8.691127028208014,
              "Insightfulness": 7.408492910406666
            },
            "omitted_dimensions": [],
            "mean": 8.483871753649249
          }
        },
        "overall": 7.325783224183142,
        "count": 27174
      },
      "table": {
        "sc": 8.484,
        "qa": 6.151,
        "overall": 7.326
      },
      "absolute_errors": {
        "qa": 0.0003860065611966945,
        "sc": 0.00012824635075148194,
        "overall": 0.00021677581685786151
      }
    },
    {
      "file": "chatGPT_detailed.csv",
      "table_model": "chatGPT_22243",
      "reconstructed": {
        "labels": {
          "Question-Answer": {
            "count": 12529,
            "dimension_means": {
              "Accuracy": 2.7177747624883875,
              "Completeness": 2.7294277277441132,
              "Consensus Alignment": 2.9951312953757676,
              "Terminology Norms": 2.6967036475384303,
              "Insightfulness": 1.92194109667659
            },
            "omitted_dimensions": [
              "Reasoning"
            ],
            "mean": 2.6121957059646577
          },
          "Single-Choice": {
            "count": 9707,
            "dimension_means": {
              "Accuracy": 7.73977541992768,
              "Completeness": 3.8016895024024926,
              "Consensus Alignment": 6.819202637336352,
              "Terminology Norms": 4.1684351499541545,
              "Insightfulness": 2.2352941175593894
            },
            "omitted_dimensions": [
              "Reasoning"
            ],
            "mean": 4.952879365436014
          }
        },
        "overall": 3.63400791510697,
        "count": 22236
      },
      "table": {
        "sc": 4.953,
        "qa": 2.612,
        "overall": 3.634
      },
      "absolute_errors": {
        "qa": 0.0001957059646575665,
        "sc": 0.00012063456398614392,
        "overall": 7.915106970290964e-06
      }
    },
    {
      "file": "grok_3_detailed.csv",
      "table_model": "grok-3-mini",
      "reconstructed": {
        "labels": {
          "Question-Answer": {
            "count": 13542,
            "dimension_means": {
              "Accuracy": 4.55708167188761,
              "Reasoning": 4.958499482966401,
              "Completeness": 4.657731501916849,
              "Consensus Alignment": 6.959164082015654,
              "Terminology Norms": 7.341382366149237,
              "Insightfulness": 4.834957908740363
            },
            "omitted_dimensions": [],
            "mean": 5.551469502279352
          },
          "Single-Choice": {
            "count": 13893,
            "dimension_means": {
              "Accuracy": 8.76556539271115,
              "Reasoning": 6.721586410384798,
              "Completeness": 5.9388900885695675,
              "Consensus Alignment": 8.666522709346149,
              "Terminology Norms": 7.25588425831174,
              "Insightfulness": 5.4465558195179575
            },
            "omitted_dimensions": [],
            "mean": 7.132500779806894
          }
        },
        "overall": 6.352098900445569,
        "count": 27435
      },
      "table": {
        "sc": 7.133,
        "qa": 5.551,
        "overall": 6.352
      },
      "absolute_errors": {
        "qa": 0.00046950227935216304,
        "sc": 0.0004992201931059981,
        "overall": 9.890044556826183e-05
      }
    },
    {
      "file": "Qwen3_8B100_detailed.csv",
      "table_model": "Qwen3_8b 100%微调",
      "reconstructed": {
        "labels": {
          "Question-Answer": {
            "count": 13582,
            "dimension_means": {
              "Accuracy": 3.7029892504323385,
              "Reasoning": 4.8153438375081,
              "Completeness": 4.098807244956048,
              "Consensus Alignment": 6.256884111203357,
              "Terminology Norms": 7.147989986785522,
              "Insightfulness": 4.823810926359299
            },
            "omitted_dimensions": [],
            "mean": 5.140970892874111
          },
          "Single-Choice": {
            "count": 13906,
            "dimension_means": {
              "Accuracy": 8.927081835261971,
              "Reasoning": 7.799151445425498,
              "Completeness": 7.626492161635334,
              "Consensus Alignment": 9.130447288955272,
              "Terminology Norms": 8.198115921139365,
              "Insightfulness": 6.6971810729114045
            },
            "omitted_dimensions": [],
            "mean": 8.063078287554807
          }
        },
        "overall": 6.619245974016056,
        "count": 27488
      },
      "table": {
        "sc": 8.063,
        "qa": 5.141,
        "overall": 6.619
      },
      "absolute_errors": {
        "qa": 2.9107125889105134e-05,
        "sc": 7.828755480687732e-05,
        "overall": 0.000245974016055861
      }
    }
  ]
}
````
