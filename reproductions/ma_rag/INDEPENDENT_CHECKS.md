# Independent checks

Run on 2026-07-29 against official code commit
`423f031cc0ffca679a047c885be999b9604100a9` and arXiv `2603.03292v3`.

## C1: released MA-RAG architecture — 1/2

The released entropy and evaluator branches both contain:

- an eight-round configurable loop;
- eight candidate generations by default;
- unanimity-based early stopping;
- conflict-query extraction;
- BM25 top-32 → reranked top-2 retrieval per generated query;
- history construction and final majority-vote accuracy.

Two specification mismatches prevent full credit:

1. Equation 4 and its text require history in descending quality order. The
   released evaluator branch uses ascending `np.argsort(scores)`, while the
   entropy branch places highest entropy first even though the paper defines
   quality as negative entropy. Figure 2 instead draws Lower → Medium → Higher,
   matching the code but contradicting Equation 4 and the prose.
2. The paper describes a configurable confidence threshold `epsilon`; the
   release exposes no such parameter and continues until either unanimous
   predictions or the maximum round.

This verifies a substantial multi-round implementation, but not an unambiguous
implementation of the declared ranking/gating specification.

## C2: seven released benchmark files — 1/2

All paper-declared evaluation sizes match exactly:

| Dataset | Rows |
|---|---:|
| MedQA | 1,273 |
| MedMCQA | 4,183 |
| MedBullets | 308 |
| MMLU-Pro | 818 |
| NEJM | 655 |
| MedExpQA | 125 |
| MedXpertQA | 490 |
| **Total** | **7,852** |

All files parse, all IDs are unique within each dataset, and all ground-truth
labels refer to released option keys when NEJM multi-answer labels are treated
as sets of keys.

Critical runner mismatch: the paper correctly declares NEJM as multi-answer,
and the release contains 41 multi-letter labels. Both MA-RAG runners call
`judger(..., single_answer=True)`, whose implementation reduces predictions to
one letter. Those 41 rows cannot be scored correctly by the released canonical
path.

Additional quality findings:

- MMLU-Pro: 40 exact / 134 whitespace-normalized duplicate questions and three
  normalized-question groups with different target texts.
- MedBullets: 10 exact duplicate questions.
- MedExpQA: one duplicate/conflicting normalized question; 67 rows contain one
  blank placeholder option, which the released prompt builder filters out.
- NEJM: one duplicate/conflicting normalized question.
- Two exact questions overlap across benchmark files:
  MMLU-Pro↔MedMCQA and MedQA↔MedXpertQA.

## C3–C6 — 0/8

The 62.2 average, +6.8 headline gain, baseline margins, cumulative ablations,
four-run variance, and timing claims were not rerun. The release omits MedCorp,
its built index, evaluator training CSVs/checkpoint, canonical predictions,
complete main-environment requirements, model/API credentials, and hardware
traces.

Source-table arithmetic independently reproduces the printed rounded values:

- Qwen3-8B mean: `55.3714` → 55.4
- MA-RAG-ext mean: `62.1714` → 62.2
- difference: `6.8000`
- MA-RAG-ext minus TC-RAG: `5.3000`

This is table concordance only, not experimental reproduction.

## Preflight

- Four mutation/unit tests pass.
- Repository and audit package pass `python -m compileall -q`.
- `ma_rag_entropy.py --help` runs.
- No local `API_KEY`, `BASE_URL`, `RETRIEVER_HOST`, `RERANKER_HOST`, corpus,
  evaluator training data, `.env`, or NVIDIA runtime was present.
- PDF pages 4, 6, 7, 9, and 23 were rendered and visually inspected.

Prepared result: **2/12**, inside the frozen plausible range of 2–4/12 but
below the 4/12 point expectation.
