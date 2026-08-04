## Claim 5 — — Section 6.1, universal approximation on sparse graphs
600 bofop graphs (n∈[60,90], r≤6), target F = tanh(mean deg) + 0.5·sin(mean nbr deg) — a
permutation- and size-invariant, DIDM-continuous functional, exactly the class Sec. 6.1
addresses. Model: MPNN with random tanh message-passing layers (depth 3) + mean readout +
ridge head; width swept 4→256. **Uniform (sup-norm) test error** — the right norm for a
universal-approximation claim, not RMSE — falls to 0.0115, about 1.2% of the target's range.
Swapping in a DIDM-discontinuous target (indicator of mean degree above median) leaves sup
error at ≈0.52 regardless of width, confirming the approximation is coming from DIDM-continuity
and not from raw capacity.
*Limitation:* random-feature MPNNs on a sampled family demonstrate the approximation, they do
not reprove density in C(bofop-DIDM). Directionally faithful, toy in scale.

---
<!-- trackio-cell
{"type": "markdown", "id": "cell_<built-in function hash>", "created_at": "2026-01-01T00:00:00+00:00", "title": "Claim 5 \u2014 \u2014 Section 6.1, universal approximation on sparse graphs"}\n-->
