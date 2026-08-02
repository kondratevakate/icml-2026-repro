# Conclusion

---
<!-- trackio-cell
{"type": "markdown", "id": "cell_c3e058624e96", "created_at": "2026-08-02T05:00:00+00:00", "title": "Overall findings"}
-->
**Overall findings** in per-claim pages. Verdicts follow the evidence; mutation tests confirm mechanism, not correlation; inconclusive claims state the blocker honestly.

<!-- trackio-cell
{"type": "markdown", "id": "cell_8aa56e1c457e", "created_at": "2026-08-02T05:00:00+00:00", "title": "Evidence boundary"}
-->
What this evidence does **NOT** cover:

1. **No trained FluxNet model was reproduced.** Every "verified" result concerns the *update
   rule and head parameterizations*, exercised with randomly sampled head outputs. Propositions
   1–3 are stated to hold for arbitrary fluxes, so this is a valid (indeed stronger) test of the
   propositions — but it says nothing about whether a trained FluxNet reaches the reported
   accuracies.
2. **No accuracy claim is tested.** All MAE numbers (Tables 2–5) are untouched. Claims 3, 4 and 5
   are recorded as inconclusive with reasons; no synthetic stand-in was produced for them.
3. **My implementation is a faithful re-implementation from the paper text, not the authors'
   code** (no code was ever released). If the authors' code differs from Sec 3.2–3.4 — e.g. a
   different inflow-shift convention, an ε in the L-head, or a hidden clip — my results describe
   the *published equations*, not their software.
4. **Conservation magnitude is dtype-dependent.** The ~1e-7 vs ~1e-8 distinction the paper draws
   between methods is float32 accumulation roundoff and scales with grid size, rollout length and
   summation order; in float64 the same code gives 4e-16. So "3.3e-8 beats 3.3e-7" is not
   evidence of a better conservation mechanism — both are exactly conservative up to roundoff.
5. **Periodic boundaries only**, as in the paper. Nothing here speaks to Dirichlet/Neumann
   boundaries, source terms, or non-uniform meshes (the paper's own stated limitations).
6. **Claim 6's DCL experiment optimizes the head logits directly, not network weights** on real
   PDE data. It establishes the causal link DCL-residual → bound violations, but the "near-zero
   violations" observed in Table 4 (0.52 % / 2.87 %) for an actually trained model on real
   traffic data are not reproduced.
7. **Single-machine timing only** — no speedup measurement of any kind was made (no GPU).
8. The **claim-4 1.87 % discrepancy** is reported, not adjudicated; I could not determine whether
   it is a transcription error in the anchored claim or a number from a table version I did not see.

<!-- trackio-cell
{"type": "markdown", "id": "cell_c4738209de87", "created_at": "2026-08-02T05:00:00+00:00", "title": "Artifacts"}
-->
```
notes_paper.md          single-read paper notes (no notes_code.md: no code released)
plan.md                 per-claim feasibility classification
fluxnet_core.py         numpy re-implementation of Eq.2 update, L/U/D heads
verify_claim1.py        -> results/claim1.json
verify_claim2.py        -> results/claim2.json
verify_claim6.py        -> results/claim6.json  (+ results/claim6.log)
verify_claims345.py     -> results/claim3.json, claim4.json, claim5.json
check_reproducibility.py  reproducibility gate — PASSES (exit 0, 0 failed assertions)
```

---
<!-- trackio-cell
{"type": "artifact", "id": "cell_d1ffb50f3bed", "created_at": "2026-08-02T05:00:00+00:00", "title": "Reproduction bundle", "artifact": "repro-fluxnet-learning-capacity-constrained-local-transport-operators-for-conservative-and-bounded-pde-surrogates/repro-bundle:v0", "artifact_type": "dataset"}
-->
**📦 Artifact** `repro-fluxnet-learning-capacity-constrained-local-transport-operators-for-conservative-and-bounded-pde-surrogates/repro-bundle:v0` · dataset

https://huggingface.co/buckets/kondratevakate/repro-fluxnet-learning-capacity-constrained-local-transport-operators-for-conservative-and-bounded-pde-surrogates-artifacts#repro-fluxnet-learning-capacity-constrained-local-transport-operators-for-conservative-and-bounded-pde-surrogates/repro-bundle:v0

---
<!-- trackio-cell
{"type": "dashboard", "id": "cell_8817be01bf40", "created_at": "2026-08-02T05:00:00+00:00", "title": "Dashboard: repro-fluxnet-learning-capacity-constrained-local-transport-operators-for-conservative-and-bounded-pde-surrogates", "dashboard_project": "repro-fluxnet-learning-capacity-constrained-local-transport-operators-for-conservative-and-bounded-pde-surrogates"}
-->
**🎯 Trackio dashboard** `repro-fluxnet-learning-capacity-constrained-local-transport-operators-for-conservative-and-bounded-pde-surrogates`

trackio-local-dashboard://repro-fluxnet-learning-capacity-constrained-local-transport-operators-for-conservative-and-bounded-pde-surrogates
