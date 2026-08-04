## Claim 3 — Theorems 8/9/10 frontier
**Source:** Sec. 3.1 Thms 8, 9, 10.
**Method (`verify_claim3.py`):** random search over sparse Dirichlet beliefs for a log pool with all
`Δ_i > 0`, for every (\|O\|, n) with \|O\| ∈ {3,4,5,6}, n ∈ {2,3,4}; then the |O| = 2 mutation with a
dense 400 × 57 × 99 grid over `(x₁, x₂, β)` plus 40 000 random draws.
**Result:** strict unanimity found for **every** size; best `min_i Δ_i = 0.2254` at \|O\| = 3, n = 2
(explicit witness stored in `results/claim3.json → witness_K3_n2`), and 0.145–0.293 across the grid.
**Mutation:** for \|O\| = 2, best `min_i Δ_i = −8.4e−7` (grid) and `−2.6e−8` (random) — never positive,
matching Thm 8.
*Honesty note:* an earlier run with an 8 000-draw budget missed the K = 5, n = 4 cell (reported −0.107);
that was a **search-budget artifact**, not a falsification — it disappeared at 30 000 draws with two
concentrations. Only the fixed run is recorded in the JSON.

---
<!-- trackio-cell
{"type": "markdown", "id": "cell_<built-in function hash>", "created_at": "2026-01-01T00:00:00+00:00", "title": "Claim 3 \u2014 Theorems 8/9/10 frontier"}\n-->
