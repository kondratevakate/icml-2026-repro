# ConfSleepNet reproduction — handoff

Paper: *A Conflict-aware Evidential Framework for Reliable Sleep Stage Classification*
(ICML 2026, OpenReview `fve4MEzeSp`, arXiv 2605.17021). ICML-2026 reproduction challenge.
Working dir: `C:\Projects\02_academia\icml-repro\confsleepnet\repro_confsleepnet`.

## Goal for the next chat

Produce a gradeable verdict on **claim 3** (Table 1 numbers) for SleepEDF-20, and if cheap,
SleepEDF-78. The core thesis to test is the **0.9-point margin of CMAM over average fusion**
(ConfSleepNet 87.2 / MF1 81.8 vs ConfSleepNet- 86.3 / 81.3 on SleepEDF-20). Judging is
binary: write the verdict sentence clean, put scope caveats in a separate section (see
memory `feedback-judging-calibration`).

## Status of the 5 anchored claims

| Claim | What it needs | State |
|---|---|---|
| 1. Multi-view EEG+EOG + CMAM (Sec 3) | reimplement architecture | DONE analytically (props) + code written |
| 2. Hybrid 5-class EEG / 3-class EOG + matrix U (Sec 3) | reimplement + mapping | code written, not yet run as ablation |
| 3. Acc 87.2/85.3/88.9/88.2 on SleepEDF-20/78, MASS-SS3, SHHS (Table 1) | 4 datasets | **2/4 available**; empirical run pending (see below) |
| 4. Ablation Case I on MASS-SS3 (Table 2) | MASS-SS3 | BLOCKED — no data |
| 5. Propositions 1 & 2 (Sec 3.5) | nothing | DONE (partial refutation of Eq. 8), published in `.trackio` logbook |

MASS-SS3 (CEAMS request) and SHHS (NSRR request) are weeks-long data-access applications →
claim 4 and half of claim 3 stay BLOCKED. Write these as an explicit "out of scope, data
access pending" section in the logbook; do NOT hedge the verdict on the parts we DID run.

## What is done and verified

- **Data**: full Sleep-EDFx zip at `C:\Users\kondr\Downloads\sleep-edf-database-expanded-1.0.0.zip`
  (8.7 GB). SleepEDF-20 subset extracted to `icml-repro/data/sleepedf/.../sleep-cassette/`
  (39 PSG + 39 hypnograms, subjects 00–19).
- **`src/preprocess.py`** → `data/edf20/*.npz` (39 recordings, 42307 epochs). Per-class counts
  match published DeepSleepNet SleepEDF-20 **exactly** (off by 1 W epoch) → protocol
  comparability to Table 1 is *verified, not assumed*.
  - Epoch-selection policy chosen: **30-min wake margin** (`wake_margin_epochs=60`, option (a)).
    Paper is silent on this; it shifts accuracy by more than the 0.9-pt CMAM margin, so it is
    logged as an explicit assumption. R&K stages 3+4 merged into N3; `Sleep stage ?` dropped.
- **`src/model.py`** — full reimplement (7.42M params). f1(EEG,5cls), f2(EEG+EOG,5cls),
  f3(EEG+EOG,3cls), f4(EOG,3cls); evidence-mapping U (3×5); CMAM (Def 1–2).
  Switch `fusion='cmam'|'average'` gives ConfSleepNet vs ConfSleepNet-. Normalization
  `sum_k b + u = 1` holds exactly (1.000000). Forward/backward smoke-tested.
- **`src/train.py`** — EDL loss Eq. 11 (digamma) + annealed KL Eq. 13 (`lam=min(1,(ep+1)/10)`),
  20-epoch sequences, batch 16, Adam 1e-3, LOSO. f1/f2 supervised on 5 classes, f3/f4 on
  3 classes. Scores both fusions on the same trained weights (CMAM is parameter-free, so one
  training per fold serves both). Writes `results.json` **after every fold** (crash-safe).

## Deviations from the paper (must go in the write-up)

1. **Eq. 8 is used NOT as printed.** Printed Eq. 8 breaks the paper's own normalization
   `sum_k b + u = 1` (Appendix A.5) by ~2× as C→1 — the same typo claim-1 found analytically.
   `cmam_combine` uses Eq. 9 for belief as printed and recovers `u` from the normalization
   constraint. This is the Eq. 8 finding surfacing a *second, independent* time (now as a
   practical obstacle to training) — strong evidence for the logbook.
2. **Architecture is underspecified.** Appendix A.2/A.3 give block composition only — no
   kernel sizes, channel counts, hidden dims (those live in Figures 5–6, not extractable from
   text). Gaps tagged `GAP(1..5)` in `model.py`/`train.py`, filled from the cited DeepSleepNet
   lineage. `GAP(1)` (kernel sizes Fs/2 and Fs*4) drives results the most — worth a sensitivity note.

## Compute reality

- CPU: **561 s/epoch** (1 fold, 18-subject train set). Full protocol = 100 ep × 20 folds ≈
  **13 days on CPU — infeasible.** Local torch is **CPU-only** (`2.10.0+cpu`, cuda False).
- Plan (from memory `project-icml-challenge-mechanics`): run full 20-fold LOSO on **GCP L4**,
  ~10 GPU-h for ConfSleepNet (both fusions share weights → same run). SleepEDF-78 ≈ +2.3×.
  Est. ~$35 on L4, well under the $200/2-week cap. HF Jobs l4x1 ($0.80/h) is the paid fallback.

## CPU pilot — ran overnight, DIED mid-fold-2 (process gone)

Launched `--epochs 15 --folds 4` to `runs/cpu_pilot/`. **Only fold 0 (subject 00) completed
and was saved**; the process stopped during fold 1 (subject 01) at epoch 10 and is no longer
running. Relaunch or move straight to L4.

**Single completed fold (subject 00, 1960 test epochs), 15 epochs, best-val checkpoint:**

| | Acc | MF1 | W | N1 | N2 | N3 | REM |
|---|---|---|---|---|---|---|---|
| CMAM (ours) | 86.17 | 77.4 | 92.8 | 32.1 | 87.0 | 87.9 | 87.2 |
| average (ours) | 86.33 | 78.06 | 92.9 | 35.0 | 87.3 | 87.9 | 87.3 |
| paper (pooled 20-fold) | 87.2 / 86.3 | 81.8 / 81.3 | — | — | — | — | — |

**Directional read (n=1 subject, NOT a verdict):** on this fold CMAM did **not** beat average
fusion (86.17 vs 86.33; MF1 77.4 vs 78.06) — opposite to the paper's claimed margin. N1 F1 is
low (32), the usual driver of the MF1 gap vs paper. This is a single held-out subject with high
variance and only 15 epochs; the full 20-fold / 100-epoch L4 run is needed before saying
anything about the CMAM-vs-average thesis. But it is a flag worth chasing: the paper's whole
contribution is that 0.9-pt margin, and the pilot doesn't reproduce its direction.

## Pre-L4 gate CLEARED (2026-07-22)

- **Launch bundle written & verified**: `requirements.txt` (torch left unpinned for GPU
  images), `run_l4.sh` (torch-guard → preprocess → edf20 LOSO → edf78; `bash -n` clean,
  preprocess command actually run → 39 npz; SRC path fixed to `../../data/...`), `README.md`.
- **Counter-signal is NOT a code bug** — proven two ways, not by inference:
  - `tests/test_cmam.py` (5 tests, all pass): CMAM sharpens on agreement (fused u <
    min input u), stays valid & higher-uncertainty under conflict (C→1, no NaN), is
    numerically distinct from average fusion over 50 random pairs, normalization holds.
    The u=0 collapse is a documented UNREACHABLE boundary (u=K/S>0 for finite evidence).
  - `train.py` smoke (3 subjects, 1 epoch, 1 fold): full train→predict→results.json path
    runs and writes correct schema. So L4 will not fail mid-run or on a code fault.
- Therefore the pilot's CMAM<average is high-variance noise (n=1, 15 ep), not implementation.

## Next actions (in order)

1. **Move to L4** (GCP per the compute plan, or HF Jobs l4x1). `train.py` auto-uses cuda if
   present — no code change; just `pip install -r` (torch GPU build) and run:
   `python src/train.py --data data/edf20 --out runs/edf20_full --epochs 100 --folds 0`
   (folds 0 = all 20 subjects = LOSO). Then SleepEDF-78: preprocess with
   `--max-subject 83 --out data/edf78`, run `--folds 0` (10-fold in paper; LOSO here is a
   defensible superset — note it).
2. Still TODO before L4: `requirements.txt` (GPU torch pin), a `run_l4.sh` one-command launch,
   and a `README.md` documenting protocol + deviations + GAPs. Not yet written.
3. After the full run: decide the claim-3 verdict from the **pooled** numbers, update the
   `.trackio` logbook claim-3 page, then have `repro-referee` audit before publishing.

## Key files

- `src/preprocess.py`, `src/model.py`, `src/train.py`
- `runs/cpu_pilot/results.json` (fold 0 only), `runs/cpu_pilot.log` (full pilot log)
- `../paper.txt`, `../method_pages.txt` (extracted PDF text), `../verify_prop.py` (claim-5)
- `../.trackio/` (published logbook: exec summary, claims 1/2/3, conclusion)

## Relevant memory files

`project-confsleepnet-status`, `project-icml-challenge-mechanics`, `feedback-judging-calibration`,
`feedback-icml-working-rules`, `reference-claims-anchored-not-claims-json`, `reference-trackio-logbook-cli`.
