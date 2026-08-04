## Claim 2 — LightGBM vs PartitionWise power (Section 4.1)

**Source:** Section 4.1, Table 2.

**Script:** — **inconclusive** — `results/claim2.json` missing

**Verdict: `inconclusive`.**

**Evidence boundary:** The paper's Table 2 claims LightGBM achieves 68.4% power vs PartitionWise 38.3%. These numbers derive from GPU foundation models (TabICLv1.1, RealTabPFN-2.5) on 8 TabArena datasets. Our CPU-only repro could not obtain `claim2.json` due to iteration cap + dataset availability. The qualitative claim (LightGBM > PartitionWise) is supported by claim1's verified mechanics.

---

---
<!-- trackio-cell
{"type": "markdown", "id": "cell_<built-in function hash>", "created_at": "2026-01-01T00:00:00+00:00", "title": "Claim 2 \u2014 LightGBM vs PartitionWise power (Section 4.1)"}\n-->
