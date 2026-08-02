# Claim 3 — DeepSeek-V3 30.28 / 21.32, Qwen-2.5-Coder-32B 23.45, "most LLMs fail to exceed 20%"

---
<!-- trackio-cell
{"type": "markdown", "id": "cell_f1e8895f0b55", "created_at": "2026-08-02T04:30:00+00:00", "title": "Claim 3 \u2014 DeepSeek-V3 30.28 / 21.32, Qwen-2.5-Coder-32B 23.45, \"most LLMs fail to exceed 20%\""}
-->
**Source:** **Table 2**, §5.1; Abstract ("most models score below 20%").
**Verdict: `falsified`** (the three per-model numbers are correct; the universal statement is not).
- Deepseek-V3 GM: Syntax `30.28`, Semantic `21.32` — match Table 2 exactly. ✔
- Qwen-2.5-Coder-32B GM Semantic `23.45` — matches. ✔
- "Most evaluated LLMs fail to exceed 20%": exhaustive count over all 24 rows gives
  **Squirrel-Syntax: 10/24 = 41.67 %** at or below 20 GM → **a minority, not "most"**;
  Squirrel-Semantic: **15/24 = 62.5 %**; pooled: **25/48 = 52.08 %**.
  Restricting to the paper's narrower §5.1 wording (closed-source only): Syntax **6/12 = 50 %** — exactly half,
  still not "most". So the sub-assertion holds only on the Semantic split, and the claim as anchored (and the
  Abstract's blanket "most models score below 20%") is **false for Squirrel-Syntax**.
**Mutation test:** swapping GM for MB drops the "≤20 %" population to 3/24 (12.5 %) on Syntax and 6/24 (25 %) on
Semantic → the "<20 %" narrative is metric-dependent. ✔ as predicted.
