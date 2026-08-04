# GT Composition — ICML-2026-Repro

**Branch:** `gt-composition` (created 2026-08-02 from `hermes-agentic-research`)
**Purpose:** track which papers form the Ground-Truth (hard-GT) benchmark set, in what order they should be reproduced, and how to keep the set saturated as new leaderboard submissions arrive.

---

## 1. What is "GT" here

GT = **hard-GT**: papers with **consensus≥3** on the official ICML-2026 agent-repro leaderboard (≥3 independent submissions whose judge verdicts agree), AND **CPU-feasible** (explicit cpu-only / no-gpu, or cpu-mention without hard-gpu requirement in the leaderboard logbook), AND **non-Theory** (theory-only papers excluded because they need symbolic/asymptotic checks, not numeric reproduction).

The live GT source is the public HF dataset (no creds):
```
https://huggingface.co/datasets/ICML-2026-agent-repro/verdicts/resolve/main/verdicts.json
```
Snapshot taken 2026-08-02: **6471 verdict entries, 2082 orids, 633 consensus≥3.**

---

## 2. Current GT lists (in `groundtruth/out/`)

| File | Contents | Count | How built |
|------|----------|-------|-----------|
| `selected_for_runs.json` | **Original 45** | 45 | July snapshot: consensus≥3 + CPU-feasible + non-Theory, area-cap 8/area, priority disputed |
| `new_cpu_friendly_strict.json` | **Strict-CPU 41** | 41 | Live snapshot (2026-08-02): strict CPU-signal scan of 497 new orids (excludes gpu/CUDA/A100), sorted by repro-time + claim-count |
| `top10_new_cpu.json` | Batch-1 (first 10 of strict-41) | 10 | Sent to arm1 leaf (run03) |
| `top10b_new_cpu.json` | Batch-2 (next 10 of strict-41) | 10 | Sent to arm1 leaf (run03) |

**Area distribution of original 45:**
Probabilistic Methods 8 · Reinforcement Learning 5 · Applications 8 · Deep Learning 8 · Optimization 8 · General ML 7 · Social Aspects 1.

**Strict-41 ordering:** `strict_cpu_by_time` (sorted by estimated reproduction time, CPU-friendly first) and `strict_cpu_by_claims` (sorted by number of anchored claims). Top of `strict_cpu_by_time`: JOyxs9ElI7, NinueNAODD, arc2pWtZLN … Bottom: DpIc1cpNKG, wpKA7G7Cqu, dfqmQ9WhCP.

---

## 3. Reproduction order (saturation sequence)

The GT is consumed in **layers**, not all at once:

1. **Layer A — Original 45** (`selected_for_runs.json`): the baseline hard-GT set. Already partially reproduced (9 published to HF via commit `b09d6a2` + `1KRpajnd6u`; see `HANDOFF_arm_configs_and_scores.md`).
2. **Layer B — Strict-41 new papers** (`new_cpu_friendly_strict.json`): papers NOT in the original 45, discovered in the live snapshot. Consumed in `strict_cpu_by_time` order (fastest/most-CPU-friendly first):
   - Batch-1 = first 10 (`top10_new_cpu.json`) → run03 arm1 leaf.
   - Batch-2 = next 10 (`top10b_new_cpu.json`) → run03 arm1 leaf.
   - Remaining 21 → future batches (Batch-3+).
3. **Layer C — Re-disputed / new consensus**: when a paper's consensus crosses ≥3 in a later snapshot, it enters the GT automatically (see §4).

**Why this order:** CPU-feasible + fast-repro papers first → maximum throughput per agent-hour; disputed/hard-GT papers prioritized for benchmark value; area-cap ensures diversity (no single area dominates the benchmark).

---

## 4. How to saturate GT (refresh procedure)

GT is **not static** — new submissions land on the leaderboard daily. To keep the set saturated:

```bash
# 1. Pull fresh verdicts (public, no creds)
cd groundtruth/data_live
curl -L -o verdicts_live.json \
  https://huggingface.co/datasets/ICML-2026-agent-repro/verdicts/resolve/main/verdicts.json

# 2. Recompute consensus≥3 + CPU-feasible + non-Theory, area-cap 8
/tmp/run03_base_venv/bin/python - <<'PY'
import json
V=json.load(open("groundtruth/data_live/verdicts_live.json"))
# group by orid -> count submissions with agreeing judge verdict
# filter CPU-feasible (hw_signals in leaderboard logbook; strict: no gpu/cuda/a100)
# cap 8 per area, target 20-50
# write groundtruth/out/new_cpu_friendly_strict.json (append new orids)
PY

# 3. Diff against previous snapshot -> new papers to enqueue
# 4. Append new orids to strict_cpu_by_time / by_claims, write next batch file
```

**Saturation rule:** a paper enters GT when (a) consensus≥3 on live verdicts AND (b) CPU-feasible AND (c) not already in original-45 or strict-41. New entries are appended to the strict list in `strict_cpu_by_time` order and consumed by the next batch.

**Cadence:** re-pull + recompute when starting a new batch (or on a schedule, e.g. weekly). The 2026-08-02 snapshot is the reference baseline; future snapshots extend it.

---

## 5. Companion artifacts

- `groundtruth/build_index.py` — groups verdicts by orid → `n_solutions`.
- `groundtruth/competitors.py` — `MIN_LOGBOOKS=5` (multiple submissions per paper).
- `groundtruth/select_sample.py` — selection logic (consensus + CPU + area-cap).
- `groundtruth/process_gt.py` / `process_gt_levelB.py` — GT processing (level A/B).
- `groundtruth/hf_dataset/hardgt_spaces.json` — hard-GT spaces with tag.
- `HANDOFF_reproduction_pipeline_plan.md` — pipeline plan (arm2 + compaction, cross-branch + HF Space sync).
- `experiments/run03_hermes_leaf_pilot/HANDOFF_arm_configs_and_scores.md` — per-paper arm configs + judge scores (kept on run03 branch, referenced here).

---

## 6. Open questions

- Should Theory papers (JOyxs9ElI7, DpIc1cpNKG, dfqmQ9WhCP in strict-41) be excluded from GT (they need symbolic checks, not numeric repro)? Currently included in strict-41 but original-45 excluded Theory.
- Batch-2 only 4/10 fully done (rest broken/524) — re-run needed before Layer B is "saturated".
- Router 524 blocker (JOyxs9ElI7, 418BWmKIzX, ugjBMARbyt) prevents some papers from completing until router-fix or fallback.
