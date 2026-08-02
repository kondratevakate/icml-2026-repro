# Claim 4 — 420+ tokens, 17.34 / 21.62 functions per script

---
<!-- trackio-cell
{"type": "markdown", "id": "cell_44b2daa44ec7", "created_at": "2026-08-02T05:00:00+00:00", "title": "Claim 4 \u2014 420+ tokens, 17.34 / 21.62 functions per script"}
-->
**Source:** §4 "Complexity of SQL Scripts"; **Table 1**.
**Verdict: `verified` (paper-internal) / `inconclusive` (independent re-measurement).**
Parsed: Squirrel-Syntax `496.90` tokens / `21.62` functions; Squirrel-Semantic `425.93` tokens / `17.34`
functions. Both ≥420 tokens; function counts match to the second decimal.
**Mutation test:** applying the predicate "≥420 tokens **and** ≥17 functions" to every prior benchmark row in
Table 1 → all fail (largest prior token count is Spider 2.0-snow at `154.63`). ✔ as predicted.

---
