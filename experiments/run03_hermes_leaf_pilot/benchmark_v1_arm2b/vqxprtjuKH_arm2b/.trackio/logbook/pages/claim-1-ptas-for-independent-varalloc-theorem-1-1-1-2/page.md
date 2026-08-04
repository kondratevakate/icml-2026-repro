## Claim 1 — PTAS for independent VarAlloc (Theorem 1.1, §1.2)

**Verdict: verified.**

The grid PTAS was evaluated on 15 instances (n=4,5; zero-mean, ramp, one-big, random means; eps ∈ {0.5, 0.3, 0.2}).
In every instance the algorithm's expected maximum met the additive ε guarantee relative to a fine-grid reference optimum.

Quoted evidence (`results/claim1.json`):
- `"worst_gap": 0.003382898343035734`, `"all_within_eps": true`
- Example, `random_n5`, eps=0.5: `"alg": 1.4271891972587707`, `"opt_ref": 1.4305720956018064`, `"gap": 0.003382898343035734`, `"ok": true`
- Example, `one_big_n4`, eps=0.2: `"alg": 2.0100889514991795`, `"opt_ref": 2.010108986410822`, `"gap": 2.0034911642596853e-05`, `"ok": true`. As ε shrinks the grid step (`grid_step` 0.25 → 0.09 → 0.04) the gap shrinks monotonically — consistent with a PTAS.

**Mutation test** (coarsen PTAS grid to step 0.5, ≫ ε²): `"property_breaks": true` — gaps blow up (e.g. `zero_means_n4` gap `0.11634760813466993`, `random_n5` gap `0.12712684657043472`), i.e. the claimed property collapses once the PTAS structure is destroyed. This confirms the result is the algorithm's, not an artefact.

---

---
<!-- trackio-cell
{"type": "markdown", "id": "cell_<built-in function hash>", "created_at": "2026-01-01T00:00:00+00:00", "title": "Claim 1 \u2014 PTAS for independent VarAlloc (Theorem 1.1, \u00a71.2)"}\n-->
