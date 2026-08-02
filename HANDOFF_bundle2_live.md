# Bundle 2 — LIVE STATUS (2026-08-03 00:30)

## DECISION (from compaction test)
- CAffNet arm2b (compaction) scored **6/10** vs original arm2 **8/10**.
- claim4: orig=verified, compaction=inconclusive (agent simplified claim4.py structure under compaction).
- **COMPACTION_WORSE → Bundle 2 runs on ORIGINAL arm2 (NO compaction).**
- batch_prompts/*.txt regenerated without compaction block.

## Launch plan (per user: 1-2 parallel)
## Wave 1 (finalizing)
- TBSyYj4VV6: done, logbook 87 lines, ALL_CLAIMS_REGENERATED (6/6 verified -> ~10/10).
- l35QweVxgn: results/claim1-6.json done, but logbook.md missing in _arm2b (agent cut off;
  first rewriter deleg_2a3f55f2 misread sibling l35QweVxgn/ logbook and did nothing).
  -> launched deleg_fbdbfe87 (strict: write ONLY _arm2b/logbook.md from _arm2b/results).
- Watchdog: old (proc_41d83b965118) killed at 54min; relaunched watchdog_wave1b (proc_80cf76bcbc0b, 120min)
  to score+publish both when l35QweVxgn_arm2b/logbook.md appears, write NEXT_WAVE.txt.
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
