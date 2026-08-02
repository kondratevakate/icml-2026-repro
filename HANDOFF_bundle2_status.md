# Bundle 2 batch run — STATUS (2026-08-02, night)

## Decision (locked, per user)
- Wait for `deleg_f26563e7` (arm2+compaction on CAffNet 20hdQQQrA4_arm2b).
- Compare vs original arm2 via `compare_runs.py 20hdQQQrA4_arm2 20hdQQQrA4_arm2b`.
- IF compaction verdicts == original → run Bundle 2 on **arm2+compaction**.
- IF compaction worse → run Bundle 2 on **original arm2** (no compaction).
- Launch **1-2 papers in parallel** (not all 10) to avoid router saturation / hang.
- HF limit: ~20 NEW Spaces/day. Bundle 2 = 10 new (none in space_map) → within limit.
  If we hit the limit later, switch to EDITING existing spaces.

## Prepared (ready to dispatch)
- `groundtruth/run_batch.py` — prepares dirs + prompts (verified).
- `experiments/run03_hermes_leaf_pilot/batch_prompts/*.txt` — 10 prompts (arm2+compaction variant).
- All 10 papers have local TASK.md + input_bundle.json.

## Bundle 2 papers (section 6e, CPU-friendly)
TBSyYj4VV6, l35QweVxgn, tRsnpaRO0m, LJdacnMXkr, KqMqJpSMnQ,
omkG80XURl, uiw8P2JGbW, ugjBMARbyt, rZTiFcDihH, vqxprtjuKH

## Token metering
- Router `/v1/usage` (localhost:8319) tracks tokens. Read before/after batch to report cost.
- Compaction goal: cut token volume on 33+ paper batch vs no-compaction baseline.

## TODO
- [ ] deleg_f26563e7 finishes → score + compare. [AUTOMATED: watchdog_compaction.sh proc_d7122fa3f68b polls logbook.md, scores, compares, writes COMPACTION_DECISION.txt + notifies]
- [ ] Dispatch Bundle 2 in pairs (background delegate_task). [after decision]
- [ ] Per paper: score_run.py → publish_local.py (compare-and-publish, dedup-guarded).
- [ ] Report final table + token usage.
- [ ] Router status: localhost:8319 ALIVE (2026-08-02 23:45). /v1/usage needs token for read; agents proxied without key.
