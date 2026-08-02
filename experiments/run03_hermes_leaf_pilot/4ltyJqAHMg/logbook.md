# logbook.md

**Paper:** *Beyond Text-to-SQL: Can LLMs Really Debug Enterprise ETL SQL?* (Squirrel Benchmark),
arXiv 2601.18119v2, OpenReview `4ltyJqAHMg`.
**Start:** 2026-08-01T22:56:40+04:00 · **End:** ~2026-08-01T23:25 +04:00 · **Elapsed: ~0.5 h** (budget 8 h hard / 4 h soft).
**LLM turns used:** ~15 of 60. Paper read ONCE (`paper/paper.txt`, 28 pp) → `notes_paper.md`. No official code exists to read.

---

## Feasibility finding that shapes every verdict

The paper releases **nothing**: a grep over the whole text for `github|huggingface|available at|release`
returns only citations and one third-party footnote (LLaMA-Factory). The Squirrel corpus (985 tasks), the seed
enterprise SQL (ByteDance-proprietary), the model outputs, and the Graph-Match scorer are all unavailable.
Independent re-measurement of any anchored number is therefore impossible — and would additionally require paid
inference on ~30 commercial LLMs, which is outside a CPU-only budget.

No toy substitute was built (TASK.md hard rule). What was built instead is an **executable audit**: the verify
scripts parse Table 1 and Table 2 straight out of the paper text (`paper_tables.py`) and test every quantitative
sub-assertion of each claim, including the *derived* assertions ("best among evaluated models", "most LLMs fail
to exceed 20%") by **exhaustive enumeration over all 24 evaluated model rows × 2 splits (48 cells)** — the claim
space is finite and fully enumerated; no sampling, no seeds. One derived assertion fails.

---

## Claim 1 — benchmark composition and complexity
**Source:** Abstract; §3.1 (seed corpus sentence); **Table 1**, §4.
**Verdict: `verified` (paper-internal) / `inconclusive` (independent re-measurement).**
Numbers (from Table 1, parsed): Squirrel-Syntax `469` tasks, `163.69` lines, AST depth `8.93`, AST width `11.69`;
Squirrel-Semantic `516` tasks, `141.58` lines, depth `8.75`, width `11.12`. All 9 checks pass: 469/516 exact,
both splits >140 lines, width >11, depth >8.7. The "1,000+ seed SQL scripts spanning 26 business scenarios"
sentence is present verbatim in §3.1.
*Caveat found:* §3.1 describes the **seed** corpus as averaging ">120 lines, depth >8, width >12" — the width>12
figure for seeds is inconsistent in direction with the benchmark widths (11.69 / 11.12) reported in Table 1.
**Mutation test:** substituting the BIRD-Critic-open row (9.73 lines, depth 8.03, width 6.01) makes all three
complexity thresholds fail → the thresholds are discriminative, not vacuous. ✔ as predicted.

## Claim 2 — Claude-4-Sonnet 36.46 / 32.17 GM, best model
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

## Claim 3 — DeepSeek-V3 30.28 / 21.32, Qwen-2.5-Coder-32B 23.45, "most LLMs fail to exceed 20%"
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

## Claim 4 — 420+ tokens, 17.34 / 21.62 functions per script
**Source:** §4 "Complexity of SQL Scripts"; **Table 1**.
**Verdict: `verified` (paper-internal) / `inconclusive` (independent re-measurement).**
Parsed: Squirrel-Syntax `496.90` tokens / `21.62` functions; Squirrel-Semantic `425.93` tokens / `17.34`
functions. Both ≥420 tokens; function counts match to the second decimal.
**Mutation test:** applying the predicate "≥420 tokens **and** ≥17 functions" to every prior benchmark row in
Table 1 → all fail (largest prior token count is Spider 2.0-snow at `154.63`). ✔ as predicted.

---

## Summary table

| Claim | Verdict | One-line evidence |
|---|---|---|
| 1 | **verified** (paper-internal) | Table 1: 469/516 tasks; 163.69/141.58 lines; width 11.69/11.12 > 11; depth 8.93/8.75 > 8.7; mutant (BIRD-Critic-open) fails all thresholds |
| 2 | **verified** (paper-internal) | Table 2: Claude GM 36.46 / 32.17, arg-max over all 24 models on both splits (runner-ups 30.92 / 28.68); 3 mutants flip as predicted; Intro's 33.17 contradicts the table |
| 3 | **falsified** | 30.28 / 21.32 / 23.45 all match Table 2, but only 10/24 (41.67 %) models are ≤20 GM on Squirrel-Syntax — "most LLMs fail to exceed 20 %" is false there (Semantic 15/24, pooled 25/48) |
| 4 | **verified** (paper-internal) | Table 1: 496.90 / 425.93 tokens ≥ 420 and 21.62 / 17.34 functions; no prior benchmark satisfies the predicate (max 154.63 tokens) |

Gate: `.venv/bin/python check_reproducibility.py` re-asserts every number above against `results/claim*.json`.

---

## Evidence boundary — what this evidence does **not** cover

1. **No independent re-measurement of any number.** Every "verified" above means *the anchored claim is entailed
   by the paper's own Table 1 / Table 2, checked mechanically*. It does **not** mean the benchmark statistics or
   the LLM scores were reproduced. In the reproduction sense, claims 1, 2, 4 and the numeric part of claim 3 are
   **inconclusive** — see item 2.
2. **Root cause of that limit:** no public code, no public data, no public model outputs, no public Graph-Match
   scorer; the seed SQL is proprietary enterprise ByteDance code; re-running ~30 commercial LLMs over 985 long
   ETL scripts is neither CPU-feasible nor free. This is a refusal, not a gap I filled with a toy.
3. **Not tested:** whether the 469/516 tasks actually exist; whether the AST depth/width/token/function
   statistics are computed correctly (parser and definition unknown — "tokens" and "functions" are never
   formally defined); whether the bug injection is minimal-change; whether the attack–defense filtering was
   applied as described; the human-validation protocol; the claimed correlation with real-world debugging.
4. **Not tested:** the GM metric's validity (whether AST-graph match implies semantic equivalence), the SFT/agent
   baselines (§5.2+), Figure 3 distributions, and Appendix tables.
5. **Claim 3's falsification is a falsification of the *stated aggregate*, not of the individual model scores** —
   those match. It is also metric-conditional: read on GM (the paper's own "success rate"), which is the only
   defensible reading given §5.1's wording.
6. **Table extraction risk:** all numbers come from PDF text extraction of the arXiv v2 PDF via PyMuPDF and a
   row-assembly parser (`paper_tables.py`). Spot-checked against the raw text lines; a systematic extraction
   error would propagate. The parser recovers exactly 24 model rows + 3 SFT rows and 9 benchmark rows, matching
   the paper's layout.
7. **Version scope:** arXiv v2 (24 Jul 2026) only. The OpenReview camera-ready may differ.
