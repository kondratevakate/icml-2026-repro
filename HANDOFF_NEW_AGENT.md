# HANDOFF — ICML-2026 AI-Scientist Repro BENCHMARK (method paper)
**Status: 2026-08-03 · branch `hermes-agentic-research` · READ THIS FIRST for any new session.**

This is the **benchmark-method project**, distinct from the older HF-leaderboard "challenge"
work documented in the (Russian) `HANDOFF.md` in this same folder. That file is still useful
lore for *scoring rules and Kate's profile* — read it second, not first.

---

## 1. GOAL OF THE STUDY

We are writing a **method paper** that introduces a **reusable benchmark** where AI-Scientist
agent *architectures* reproduce **anchored claims** from ICML-2026 public submissions, scored
against a **reusable ground-truth (GT) corpus** (Trackio logbooks + `audit_*.py` code-execution
+ `score_run.py`).

- The product is the **GT corpus + judge + logbook format** so *other labs* can run their own
  AI-Scientist on ICML and compare. **HF Spaces are only one serialization of agent logbooks,
  NOT the goal** — do not optimize Space scores.
- **Pure architecture comparison:** the LLM is held constant at `tencent/hy3:free` (via local
  router `localhost:8319`) across ALL arms, so differences reflect *architecture*, not model.
- Stimulus set = **42 ICML-2026 papers** (`groundtruth/out/selected_for_runs.json`, "alive" —
  extends as new logbooks land). 13 Clinical/EHR orids in it are benchmark *stimuli*, not
  hand-repro targets.

## 2. GIT BRANCHES (this repo)

| branch | purpose |
|---|---|
| `hermes-agentic-research` | **ACTIVE — all benchmark work lives here** (you should be on this) |
| `codex/medical-reproducibility-map` | SEPARATE checkout at `/mnt/d/projects/02_academia/icml-repro` (different machine path, no pilot runs) — do NOT confuse with this repo |
| `gt-composition` | GT corpus composition work |
| `hermes-run03-arm3-sakana-pilot` | Sakana arm3 pilot |
| `master` | base |

Remote: `git@github.com:kondratevakate/icml-2026-repro.git`. **Never push without explicit
permission; commit only after validation.** (`/mnt/d/...` is a Windows-mounted *separate*
checkout on branch `codex/...` — pilots are NOT there.)

## 3. WHERE DATA LIVES (and what is NOT in git)

**The git repo holds CODE + LOGBOOKS + METADATA, NOT the large reproduction datasets.**
GB-scale paper data lives on local D: and is fetched at run time by URL — it is gitignored
and must NOT be committed.

- **Anchored claims (precise, judge-used):** `groundtruth/data/claims_anchored.json` (≈6 MB;
  this is *text metadata* — paper IDs, claim wording, section/table anchors — not datasets).
- **GT corpus + stimulus set:** `groundtruth/out/corpus_all.json` (≈1.3 MB metadata),
  `corpus_medical.json`, `selected_for_runs.json` (the 42-paper pool). All metadata.
- **Challenge verdicts / leaderboard (external, URL-pinned):** pulled from
  `https://huggingface.co/datasets/ICML-2026-agent-repro/verdicts/...` via `huggingface_hub`
  (see `groundtruth/saturate_gt.py`). Cached locally in `groundtruth/data_live/` (e.g.
  `verdicts_live.json` 26 MB — metadata, not raw data).
- **REAL reproduction datasets (the GBs Kate mentioned):** live on **D:**
  `/mnt/d/projects/02_academia/icml-repro/data/` (sample/demo subsets: `sleepedf`,
  `mimic3demo`, `mimic-iv-note-2.2`, `cifar100n`, `mfiddr_sample`), and full-scale sets
  (Sleep-EDF Expanded 8.7 GB, full MIMIC, OpenNeuro `ds004504`, TCIA LIDC-IDRI) are pulled
  on demand from **HF / OpenML / author URLs** into `/mnt/d/Downloads/` or
  `~/.cache/huggingface` (currently 3.2 GB). Runs reference data by URL or by path under
  `data/`; they do NOT assume data is in git.
- **HF Spaces token:** `/home/kate/.cache/huggingface/token` (chmod 600). Publish via
  `HfApi().upload_folder(...)` — **git-over-HTTPS to HF fails** (credential prompt).

> Rule: never `git add` large datasets. If a run produced a big artifact (e.g.
> `*.ply`, `*.npy`, `results/*.json` > a few MB), confirm it is result output, not source
> data, before committing; prefer gitignoring raw data dirs (`**/data/`, `**/*.arff`,
> `**/*.ply` already ignored).

- **Per-run artifacts:** `experiments/run03_hermes_leaf_pilot/{pilot_4arm, benchmark_v1_prefed,
  benchmark_v1_arm2b, runs_incomplete, docs, infra}/` — see §5.
- **Pipeline scripts:** `groundtruth/*.py` (`local_to_trackio.py`, `score_run.py`, `corpus.py`,
  `build_icml_logbook.py`, …) and `experiments/run03_hermes_leaf_pilot/infra/` (`run_arm2.py`
  per-paper harness, `make_bundles.py`, `publish_logbook.py`, `snapshot_usage.py`).
- **Experiments also mirrored on HF Spaces** as `paper-<orid>` tags (one Space per paper,
  EDIT not recreate).

## 4. LOCKED DECISIONS (`experiments/run03_hermes_leaf_pilot/DECISIONS.md`)

- **D1 Arms:** arm2 (Hermes + skills) = primary workhorse; arm1 = control; arm3 (Sakana, ~7h,
  self-report overstates, crashes) & arm4 (ARC-light, didn't execute) = documented
  non-viable at scale (evidence: `pilot_4arm/20hdQQQrA4` all-4-arms run).
- **D2 Compaction:** **OFF**. CAffNet clean pilot regressed 4→3 verified; TBSyYj4VV6 confounded
  (prompt also changed) — rule rests on one clean data point.
- **D3 Claim feed:** pre-fed claims = primary scored condition; **autonomous claim-extraction =
  RQ8 (OPEN)**, needs an `extraction-GT` layer in `corpus_all` before promotion.

## 5. CURRENT REPO STATE (verified 2026-08-03)

`experiments/run03_hermes_leaf_pilot/`:
- `pilot_4arm/` — CAffNet `20hdQQQrA4` on all 4 arms (reference validation).
- `benchmark_v1_prefed/` — **18 complete pre-fed runs** (logbook.md + _score.json + Trackio json).
- `benchmark_v1_arm2b/` — 10 frozen compaction runs (decision: OFF).
- `runs_incomplete/` — **45 partial runs**: 10 have `results/claim*.json` (SALVAGEABLE via
  rewriter), 35 are empty shells (discard).
- `docs/` — consolidated handoffs + `REVIEW_handoff.md` (the review agenda).
- `infra/` — pipeline scripts.
- `DECISIONS.md` — the locked D1–D3.

**Trackio reproducibility rule (enforced):** every complete run has BOTH `logbook.md`
(judge-readable) AND regenerable `.trackio/logbook/logbook.json` (regen via
`groundtruth/local_to_trackio.py <dir>`). `.gitignore` drops Trackio static (html/js/css/svg)
but keeps `logbook.json` + `pages/*.md`.

## 6. HOW TO CONTINUE (next agent checklist)

1. **Read `DECISIONS.md` + `docs/REVIEW_handoff.md`** (the open-item agenda).
2. **Recover the 10 salvageable `runs_incomplete/` runs:** dispatch rewriter delegates (2-parallel)
   that read `results/claimN.json` + `TASK.md`, write `logbook.md` (verdicts taken ONLY from the
   agent's own JSON — never invent), run `score_run.py`, regen Trackio. Verify each `_score.json`.
   Then move to `benchmark_v1_prefed/`.
3. **Discard the 35 empty shells** (move to `runs_discarded/`, not delete).
4. **Optional, before locking D2:** a 2nd clean (prompt-held) compaction pilot.
5. **RQ8 (autonomous extraction):** build `extraction-GT` layer in `corpus_all` if promoting to
   primary axis; otherwise keep as open RQ in the method paper.
6. **Method paper:** lives in Kate's Obsidian
   (`.../AI Scientist Benchmark/06 Experiment design — 4 arms.md`) — has RQ1–RQ7 + appended
   RQ8 (autonomous-extraction) + v2 design sketch, all in English. NOTE: that doc is otherwise in
   Russian; the method paper should be fully English.

## 7. HARD RULES (do not violate)

- **Scientific integrity:** never invent a verdict. `not attempted`/`not executed`/`not
  reproduced` stay truthful 0. Canonical verdict words: `verified | supported | inconclusive |
  falsified` (case-insensitive, anywhere in page).
- **HF Spaces:** NEVER create a 2nd Space for an existing paper — EDIT `paper-<orid>`.
- **Model:** fixed `tencent/hy3:free` via `localhost:8319`. Do not swap model mid-benchmark
  (it would confound the architecture comparison).
- **No commit before validation; no push without permission.**
- **Reuse, don't re-derive:** `score_run.py` reads `logbook.md` (universal row parser), NOT the
  JSON verdict field — editing `results/claimN.json` verdict changes nothing.

## 8. QUICK VERIFY (run on arrival)

```bash
cd /home/kate/projects/02_academia/icml-2026-repro
git branch --show-current          # expect: hermes-agentic-research
ls groundtruth/out/selected_for_runs.json   # the 42-paper stimulus set
ls experiments/run03_hermes_leaf_pilot/benchmark_v1_prefed/ | wc -l   # expect 18
python3 -c "import json;print(json.load(open('groundtruth/data/claims_anchored.json'))['meta'])"
```
