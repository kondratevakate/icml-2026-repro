# Bundle 2 — LIVE STATUS (2026-08-03 00:30)

## DECISION (from compaction test)
- CAffNet arm2b (compaction) scored **6/10** vs original arm2 **8/10**.
- claim4: orig=verified, compaction=inconclusive (agent simplified claim4.py structure under compaction).
- **COMPACTION_WORSE → Bundle 2 runs on ORIGINAL arm2 (NO compaction).**
- batch_prompts/*.txt regenerated without compaction block.

## Launch plan (per user: 1-2 parallel)
## Wave 1 (DONE, published)
- TBSyYj4VV6: **10/10** (6/6 verified) — Space published (re-published after scorer fix).
- l35QweVxgn: **10/10** (6/6 verified) — Space published (re-published after scorer fix).
- CRITICAL FIX: score_run.parse_logbook now universal (3/5/6-col tables). Was scoring 0/12 incorrectly.

## Wave 2 (DONE, published)
- tRsnpaRO0m: **10/10** (6/6 verified) — Space published.
- LJdacnMXkr: **8/10** (3 verified + 2 toy + 1 inconclusive) — Space published.
- 3-col logbook format worked: both agents delivered canonical tables, scorer parsed clean.

## Wave 3 (RUNNING)
- KqMqJpSMnQ + omkG80XURl — deleg_8c5c398d (background), 3-col logbook demanded.
- watchdog_wave3 (proc_d3cc1994b393) scores+publishes both, writes NEXT_WAVE.txt.
- Wave 2 (next): tRsnpaRO0m + LJdacnMXkr
- Wave 3: KqMqJpSMnQ + omkG80XURl
- Wave 4: uiw8P2JGbW + ugjBMARbyt
- Wave 5: rZTiFcDihH + vqxprtjuKH
- After each wave: score_run.py → publish_local.py (dedup-guarded; 10 new Spaces < 20/day limit).

## Compaction note
Compaction hurt quality (8→6). Do NOT use compaction for Bundle 2. If token cost on 33+ batch
becomes a problem later, revisit compaction RULE (force agent to keep full claim structure in
COMPACTION.md, not simplify).

## Token metering
- Router /v1/usage needs token (unauthorized without). Agents proxied w/o key. Will report
  per-wave if token accessible; else note "metered by HF Spaces publish count".
