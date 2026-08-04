# Claim 3 — Table 3 (shallow water) — inconclusive

---
<!-- trackio-cell
{"type": "markdown", "id": "cell_bb359902269b", "created_at": "2026-08-02T05:00:00+00:00", "title": "Claim 3 \u2014 Table 3 (shallow water) \u2014 inconclusive"}
-->
**Paper source:** Section 4.2, Table 3. Quoted: FluxNet-LAP depth MAE 3.12e-3 vs 6.74e-3
(Box+Mass projection), E_cons 3.3e-7 → 3.3e-8.
**Record:** `verify_claims345.py` → `results/claim3.json`.

Blockers: (i) no code repository anywhere in the paper; (ii) the 2D SWE dataset with dry regions
is authors' own and unreleased, with generation parameters not in the main text; (iii) the number
requires pushforward training of ResNet **and** FNO backbones plus three baselines, multi-seed —
outside an 8h CPU budget even with data.
Additional observation, recorded rather than resolved: **3.12e-3 is the ResNet-backbone
FluxNet-LAP entry while 6.74e-3 is the FNO-backbone projection entry.** Within the ResNet
backbone the projection baseline is 22.1e-3; within the FNO backbone FluxNet-LAP is 2.22e-3.
The anchored comparison crosses backbone categories.
The *conservation-error* half of the claim is covered generically by claim 1 (a flux-form model
lands at 1e-8…1e-7 in float32 by construction, and a projection baseline that renormalizes mass
globally lands in the same band — so the 3.3e-7 → 3.3e-8 "improvement" is within the float32
roundoff band rather than a qualitative difference). No toy substitute was produced.
