## Claim 4 — — Corollary 5.3, compactness of bofop-DIDMs under the mover's distance
DIDM = pushforward of the vertex measure under v ↦ (deg v, neighbour-degree distribution),
represented as a joint histogram on the compact grid [0,r]², r=6. Mover's distance computed
**exactly by linear programming** over the transport polytope (scipy HiGHS), ground metric ℓ¹.
Compactness tested as total boundedness: a greedy ε-net (ε=0.35) over N sampled bofop DIDMs
saturates at 5 elements for N=20 and N=40. Removing the degree bound (ER graphs of growing
size, 1-D degree DIDM, exact 1-D W₁) gives net sizes 10/20/40 — every sample is its own net
point, no saturation. Properness: a DIDM with all mass above the bound sits at mover's
distance 0.124 > 0 from all 20 sampled bofop DIDMs, so the inclusion is strict.
*Limitation:* a finite-sample ε-net is a surrogate for a topological statement. It is strong
evidence of total boundedness and a clean demonstration that the fiber bound is what buys it,
but it is not a proof.

---
<!-- trackio-cell
{"type": "markdown", "id": "cell_<built-in function hash>", "created_at": "2026-01-01T00:00:00+00:00", "title": "Claim 4 \u2014 \u2014 Corollary 5.3, compactness of bofop-DIDMs under the mover's distance"}\n-->
