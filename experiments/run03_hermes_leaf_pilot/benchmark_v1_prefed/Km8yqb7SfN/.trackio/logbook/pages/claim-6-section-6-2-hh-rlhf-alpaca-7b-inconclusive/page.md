# Claim 6 — Section 6.2 (HH-RLHF / Alpaca-7B) → **inconclusive**

---
<!-- trackio-cell
{"type": "markdown", "id": "cell_36c6acd97001", "created_at": "2026-08-02T04:30:00+00:00", "title": "Claim 6 \u2014 Section 6.2 (HH-RLHF / Alpaca-7B) \u2192 **inconclusive**"}
-->
Script `verify_claim6.py`, data `results/claim6.json`. No toy substitute was produced.
Blocking reasons (all recorded in the JSON): (i) one RL run of a 7B policy per Pareto weight w₁ with
TRL GRPO and M = 8 samples/prompt — GPU-cluster scale, impossible on CPU within the 2h per-claim cap;
(ii) the two Qwen3-4B Bradley-Terry reward models and the two Qwen-32B golden judges are author-trained
and unreleased; (iii) no public code URL in the paper (supplementary only); (iv) the claim's evidence
is Fig. 6b, a figure with no numeric table, so there is no reported number to re-assert.
hh-rlhf itself is public — the blockers are the reward models, the missing code and the GPU scale,
not dataset access.

---
