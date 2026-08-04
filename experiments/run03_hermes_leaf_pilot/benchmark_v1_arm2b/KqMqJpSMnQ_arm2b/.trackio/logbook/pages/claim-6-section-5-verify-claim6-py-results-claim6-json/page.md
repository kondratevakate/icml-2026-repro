## Claim 6 — Section 5 (`verify_claim6.py` → `results/claim6.json`)

6×6 grid (36 nodes, 60 edges), 50 train + 50 test routes as random weighted shortest
paths, edge weight = train-route frequency, LP rounding at κ=1 (θ=0.5) with the smallest
vertex budget reaching φ train-route coverage; baselines = greedy top-weight edges and
greedy route-cover. 10 trials.

| φ | LP edges | greedy-weight edges | greedy-route edges | LP test cov |
|---|---|---|---|---|
| 0.60 | 35.9 | 16.6 | 35.6 | 0.376 |
| 0.70 | 40.2 | 21.6 | 39.7 | 0.426 |
| 0.75 | **41.5** | 24.4 | 42.5 | 0.444 |
| 0.80 | 43.4 | 27.6 | 43.4 | 0.468 |
| 0.90 | 46.6 | 36.1 | 46.8 | 0.528 |

* **41.5 edges at φ=0.75 vs the claimed 52** — same order, but not a match (±3 tolerance
  failed).
* LP is marginally better than the coverage-valid greedy-route baseline at φ=0.75 and
  φ=0.9 and tied at φ=0.8; the top-weight greedy keeps fewer edges but does **not** attain
  the required route coverage, so it is not a like-for-like baseline. The claimed clear
  advantage for φ≤0.8 is **not** reproduced.
* **Mutation:** shuffling edge weights (destroying route structure) forces the LP set to
  54.8–60 edges (essentially the whole grid) at every φ — so the compression measured on
  real route structure is genuine signal, not an artefact.

Verdict: **inconclusive** — the route generator, edge weighting, and the exact definition
of "calibration-set size" are not specified in the available bundle (no PDF), and the
absolute count is strongly protocol-dependent. Not marked *falsified*: our numbers are in
the same regime and the qualitative compression effect reproduces.

---
<!-- trackio-cell
{"type": "markdown", "id": "cell_<built-in function hash>", "created_at": "2026-01-01T00:00:00+00:00", "title": "Claim 6 \u2014 Section 5 (`verify_claim6.py` \u2192 `results/claim6.json`)"}\n-->
