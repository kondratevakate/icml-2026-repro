# Claim 1 — benchmark composition and complexity

---
<!-- trackio-cell
{"type": "markdown", "id": "cell_be9626997e46", "created_at": "2026-08-02T04:30:00+00:00", "title": "Claim 1 \u2014 benchmark composition and complexity"}
-->
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
