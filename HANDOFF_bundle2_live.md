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

## Wave 3 (DONE, published)
- KqMqJpSMnQ: **10/12** (~8/10, 5 verified + 1 toy) — Space published.
- omkG80XURl: **12/12** (6/6 verified = 10/10) — Space published.

## Wave 4 (DONE, published)
- uiw8P2JGbW: **10/12** (~8/10, 5 verified + 1 toy) — Space published.
- ugjBMARbyt: **12/12** (6/6 verified = 10/10) — Space published.

## Wave 5 (RECOVERING)
- Original deleg_c49681df finished but BOTH agents cut off before logbook:
  * rZTiFcDihH: 6/6 results present, logbook.md written at 04:19 (OK).
  * vqxprtjuKH: only 3/6 results (claim1,5,6); claim2,3,4 missing + no logbook.
- watchdog_wave5 (proc_9c38a8a936e1) died SIGTERM at ~24min (not killed by me; cause unknown).
- LAUNCHED deleg_f9f2e87c: finish vqxprtjuKH (run verify_claim2/3/4.py -> 3 JSONs + logbook.md).
- RELAUNCHED watchdog_wave5b (proc_2e155b4f743b, 150min): waits for both logbooks, scores+publishes, writes BUNDLE2_COMPLETE.
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
