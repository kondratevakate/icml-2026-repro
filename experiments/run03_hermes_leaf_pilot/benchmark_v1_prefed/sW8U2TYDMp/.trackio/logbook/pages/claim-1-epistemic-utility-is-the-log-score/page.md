## Claim 1 — epistemic utility is the log score
**Source:** Sec. 2 (`U_i(o) = log P_i(o)`), Def. 2 (log pool), Def. 7 + Prop. 32 (welfare gap).
**Method (`verify_claim1.py`, 4000 trials, \|O\| = 5, n = 3):** checked that (a) the log score is strictly
proper, (b) `softmax(log P) = P`, (c) `softmax(Σ β_i log P_i)` equals the logarithmic pool exactly, and
(d) the identity `Δ_i = H(P_i) − H(P) − KL(P‖P_i)`.
**Result:** properness held in all trials; max errors `3.3e−16` (b), `0.0` (c), `1.8e−15` (d).
**Mutation:** replacing the log score with the linear score `W_i = P_i` breaks softmax recovery
(err 0.49) and the identity (err 3.79) — i.e. the whole apparatus is specific to the log score.

---
<!-- trackio-cell
{"type": "markdown", "id": "cell_<built-in function hash>", "created_at": "2026-01-01T00:00:00+00:00", "title": "Claim 1 \u2014 epistemic utility is the log score"}\n-->
