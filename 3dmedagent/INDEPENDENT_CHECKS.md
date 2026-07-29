# Independent checks

Run on 2026-07-29 against official code commit
`3ee6a49cec9bfba58e0d16da8c05b6b16342139e`.

## C1: released architecture — 2/2

Static checks independently locate the OAMI memory builder, global/detail/slice
CT-CLIP paths, runtime-tool and T1S switches, and the loop bound used at
runtime. The paper/README invocation explicitly sets `--t1s-max-iters 5`;
the CLI default is only `1`, so five turns are not the unconfigured default.

The official CLI help runs. A CPU dry run over one row from each recognition
subtype selected three records, recognized the five-iteration cap, completed
with zero schema warnings, and produced zero visual iterations—as expected
without reports, volumes, masks, embeddings, and model services. This verifies
released routing and configuration, not a clinical end-to-end rerun.

`python -m compileall -q .` passes from the official repository root.

## C2: DeepChestVQA release — 2/2

Independent standard-library parsing of the released CSV confirms:

| Check | Result |
|---|---:|
| Rows / unique question IDs | 1,020 / 1,020 |
| Unique scans | 892 |
| Subtypes | 17, each with 60 rows |
| Recognition | 3 subtypes / 180 rows |
| Visual reasoning | 8 subtypes / 480 rows |
| Medical reasoning | 6 subtypes / 360 rows |
| Source datasets | ReXGroundingCT 990; NSCLC 30 |
| Split labels | `train` 1,020 |

Quality caveats discovered independently:

- 1,020 rows reduce to 996 unique `(scan, MCQ text)` pairs: 24 extra rows
  occur in 23 repeated groups.
- Two repeated groups for scan `train_1378_c_2` contain conflicting correct
  options for the same MCQ text.
- 180 MCQ strings contain options only (for example `A: No B: Yes`), although
  their separate `question` field is populated.
- Every released row is labelled `train`; no held-out split is encoded in this
  CSV.

These caveats do not change the paper's benchmark composition counts.

## C3–C6 — 0/8

The headline 40+ task gain, model comparisons, staged ablations, radiologist
agreement, and matched per-turn improvements were not rerun. The compact
release omits the necessary CT volumes, masks, reports, intermediate caches,
canonical predictions, complete credentials, expert traces, and matched
per-turn outputs. Paper tables are visual/source anchors only.

## Tests and hashes

- Four mutation/unit tests pass: valid release row, duplicate ID rejection,
  missing-answer rejection, and allowance for the same text on different scans.
- CSV SHA-256:
  `ea6caa6359f1dcde7c4a007412a36e56bbd2a96c138d21b8c6a40202e9994931`
- Pipeline SHA-256:
  `5d2bdd39c3b7864c4dabcb46c99cbd636ded037a2b02109db2b9208ab85b061f`
- Memory helper SHA-256:
  `8be765a1edfb090c82cd296105f538e8d48aa0fe7a8eddabb18f93a8f5296a97`

Prepared result: **4/12**, equal to the frozen forecast.
