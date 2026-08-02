# Claim 2 — Claude-4-Sonnet 36.46 / 32.17 GM, best model

---
<!-- trackio-cell
{"type": "markdown", "id": "cell_19239e6018dd", "created_at": "2026-08-02T05:00:00+00:00", "title": "Claim 2 \u2014 Claude-4-Sonnet 36.46 / 32.17 GM, best model"}
-->
**Source:** **Table 2**, §5.1 Main Results (prose: "peak success rate of 36.46% GM score … 32.17% GM").
**Verdict: `verified` (paper-internal) / `inconclusive` (independent re-measurement).**
Parsed Claude row: Syntax EM `23.88` / GM `36.46` / MB `68.02`; Semantic EM `31.78` / GM `32.17` / MB `43.69`.
Exhaustive arg-max over all `24` evaluated LLM rows: Claude-4-Sonnet wins both splits on GM; runner-up on Syntax
is Doubao-Seed-1.6 at `30.92`, on Semantic O3-mini at `28.68`.
*Inconsistency found in the paper:* the **Introduction** states `33.17%` on Squirrel-Semantic while the Abstract,
Table 2 and §5.1 all state `32.17%`. The anchored claim follows the table.
**Mutation test (3 mutants, all as predicted):** (A) leave-one-out — dropping Claude moves the arg-max to
Doubao-Seed-1.6 (30.92) / O3-mini (28.68); (B) setting Claude's Syntax GM to runner-up−0.01 flips the arg-max;
(C) ranking by EM with the paper's own SFT rows in the pool makes `+ DM-SFT` (27.27) the winner — so "best" is
metric- and pool-specific. Honest note: an initial prediction that the EM swap *alone* (24-model pool) would flip
the winner was **wrong** — Claude is the EM arg-max too; this is recorded in `results/claim2.json`.
