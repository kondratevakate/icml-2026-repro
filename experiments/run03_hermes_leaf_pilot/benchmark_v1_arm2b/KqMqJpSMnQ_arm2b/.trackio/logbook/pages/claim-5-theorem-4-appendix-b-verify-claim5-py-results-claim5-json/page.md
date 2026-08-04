## Claim 5 — Theorem 4, Appendix B (`verify_claim5.py` → `results/claim5.json`)

A hardness *proof* is not machine-checkable; its *reduction* is. Reduction implemented as
`(H,k) → (G=H, r=k, ε_k = 1 − C(k,2)/|E(H)|)`, and the equivalence
"H has a k-clique ⟺ ∃K, |K|≤k, W−e(K) ≤ ε_k W" checked by brute force.

* Exhaustive over **all 1024 labelled graphs on n=5** × all k: **4092 decision instances,
  agreement 1.000**.
* Random n=7,8 graphs: **386 instances, agreement 1.000**.
* Constant-ε padded variant (disjoint k-clique pad keeping ε in a fixed band):
  agreement 0.95 (38/40; the 2 misses are pad-degenerate instances where the pad itself
  realises the target, an artefact of the padding construction, not of the reduction).
* **Mutation:** allowing one edge of slack (`ε_k + 1/|E|`) breaks the equivalence on
  **637/4092** instances — the threshold is exactly at C(k,2), as the reduction requires.

Verdict: **verified** (reduction correctness; NP-hardness then follows from hardness of
CLIQUE).

---
<!-- trackio-cell
{"type": "markdown", "id": "cell_<built-in function hash>", "created_at": "2026-01-01T00:00:00+00:00", "title": "Claim 5 \u2014 Theorem 4, Appendix B (`verify_claim5.py` \u2192 `results/claim5.json`)"}\n-->
