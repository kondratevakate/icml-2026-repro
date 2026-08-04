# Arm4 (ARC) status — 20hdQQQrA4 (CAffNet) pilot

## Decision (2026-08-02)
Full ARC pipeline (`run_ais_v2.py --topic T20`, BFTS tree-search) was attempted
on `tencent/hy3:free` via the local Hermes router. It ran ~7h on a single paper
and did NOT produce a submission (killed before completion). Reason: hy3:free is
too slow for BFTS — every agent step re-sends the full inflated context, and the
model frequently returns empty tool-call arguments (mitigated with a retry patch,
but wall-clock remained ~7h/paper).

## What IS committed
- `20hdQQQrA4_arm4/` = **lightweight** ARC run: a single `researchclaw.llm.client`
  call producing verify-script stubs + verdicts in text. NOT executed. Honest label
  in `_run_meta.json` ("lightweight, single LLM call, no tree-search execution").
- Patches to make ARC route through hy3:free live in the **separate repos**
  (`AI-Scientist-v2`, `AutoResearchClaw`) — see those repos for:
  - `AI-Scientist-v2/ai_scientist/llm.py` — AVAILABLE_LLMS + hy3/tencent branch in
    get_response_from_llm + create_client PROXY_API_KEY routing to localhost:8319
  - `AI-Scientist-v2/ai_scientist/treesearch/backend/backend_openai.py` — retry on
    empty/invalid tool-call arguments (graceful recovery instead of crash)
  - `AutoResearchClaw/experiments/arc_bench/scripts/run_ais_v2.py` — CODE_MODEL/
    CHAT_MODEL default to tencent/hy3:free
  - `AutoResearchClaw/experiments/arc_bench/{manifests,rubrics}/T20.{yaml,json}`

## Recommendation
For the 4-arm benchmark, arm4 is currently lightweight-only. Full BFTS on hy3:free
is ~7h/paper → not viable for 9 papers. Options: (a) keep arm4 lightweight,
(b) run full ARC only on the pilot paper for a demo, (c) drop arm4 to 3 honest arms.
