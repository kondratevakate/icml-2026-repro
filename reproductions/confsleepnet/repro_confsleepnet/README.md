# ConfSleepNet reproduction

Reproduction of *A Conflict-aware Evidential Framework for Reliable Sleep Stage
Classification* (ICML 2026, OpenReview `fve4MEzeSp`, arXiv 2605.17021) for the
ICML-2026 reproduction challenge. No upstream code exists — the model is
reimplemented from the paper text alone (see `src/model.py` docstring).

## What is here

| File | Purpose |
|---|---|
| `src/preprocess.py` | EDF → `data/<set>/*.npz`, per-recording 30 s epochs, EEG+EOG. |
| `src/model.py`      | Full architecture (7.42M params): f1–f4 evidential nets, U mapping, CMAM fusion. |
| `src/train.py`      | EDL loss (Eq. 11) + annealed KL (Eq. 13), LOSO CV, scores both fusions. |
| `verify_prop.py`    | Analytic check of Propositions 1–2 (claim 5). numpy-only, no data, no GPU. |
| `run_l4.sh`         | One-command full run on an L4 GPU (preprocess + train, edf20 then edf78). |

## Claims targeted

Gradeable claims live in `claims_anchored.json` (challenge canon). Status:

- **Claim 5 (Propositions 1–2)** — DONE analytically via `verify_prop.py`. Partial
  refutation: **Eq. 8 as printed breaks the paper's own normalization** `sum_k b + u = 1`
  (Appendix A.5) by ~2× as the conflict degree C→1. Published in the `.trackio` logbook.
- **Claim 1 (multi-view + CMAM design)** — architecture reimplemented; Eq. 8 obstacle
  resurfaces here a second, independent time (see Deviation 1).
- **Claim 3 (Table 1 accuracy)** — SleepEDF-20/78 runnable (this bundle). MASS-SS3 and
  SHHS need weeks-long data-access applications → **out of scope, data access pending**.
- **Claims 2, 4** — code present; claim 4 (MASS-SS3) BLOCKED on data.

## Data

SleepEDF-Expanded (PhysioNet, sleep-cassette). Point `--src` at the extracted
`sleep-cassette/` dir. `--max-subject 20` → SleepEDF-20; `83` → SleepEDF-78.
Preprocess reproduces the published DeepSleepNet SleepEDF-20 per-class epoch counts
**exactly** (off by 1 W epoch), so protocol comparability to Table 1 is verified.

## Protocol (Appendix A.9)

20-epoch sequences, batch 16, Adam lr 1e-3, 100 training epochs, subject-wise
k-fold CV with k=20 on SleepEDF-20 (= leave-one-subject-out). Per fold: one subject
tests, the next validates, the rest trains. Metrics: overall accuracy and macro-F1,
pooled over folds. CMAM is parameter-free (Eq. 13 sums only the four per-view losses),
so ConfSleepNet and ConfSleepNet- (average fusion) share weights and differ only at
inference — one training per fold serves both.

## Deviations from the paper (must be read with any verdict)

1. **Eq. 8 is NOT used as printed.** The printed form breaks `sum_k b + u = 1` by ~2×
   as C→1 (the same typo claim 5 found analytically). `cmam_combine` uses Eq. 9 for the
   belief masses as printed and recovers `u` from the normalization constraint. The
   printed variant is kept available for an ablation.
2. **Architecture is underspecified.** Appendix A.2/A.3 give block composition only — no
   kernel sizes, channel counts, or hidden dims (those live in Figures 5–6, not
   extractable from text). Gaps are tagged `GAP(1..5)` in `model.py`/`train.py` and filled
   from the cited DeepSleepNet lineage. `GAP(1)` (branch kernel sizes Fs/2 and Fs*4)
   drives results the most — worth a sensitivity note.
3. **Epoch-selection policy: 30-min wake margin** (`--wake-margin-epochs 60`). The paper
   is silent on this and it shifts accuracy by more than the 0.9-pt CMAM margin, so it is
   an explicit logged assumption. R&K stages 3+4 merged into N3; `Sleep stage ?` dropped.
4. **SleepEDF-78 uses LOSO**, a defensible superset of the paper's 10-fold split.

## Run

Analytic claim (seconds, no data/GPU):

    python verify_prop.py

Full empirical run on an L4 GPU (torch auto-selects cuda):

    bash run_l4.sh

CPU is infeasible: 561 s/epoch → full protocol ≈ 13 days. Use a GPU.

## Reproducibility note

`--seed 0` fixes torch/numpy RNG. LOSO on n=20 subjects has high per-fold variance;
single-fold numbers are not indicative — only the pooled 20-fold result is a verdict.
A CPU pilot (`runs/cpu_pilot/`, 1 fold, 15 epochs) did not reproduce the direction of
the 0.9-pt CMAM-over-average margin, but n=1/15-epoch is inside the noise band; the full
run settles it.
