## Claim 2 — PTAS for correlated VarAlloc (Theorem 1.2, §1.2)

**Verdict: inconclusive.**

The correlated PTAS (n=3,4) met the additive ε guarantee against a 120 000-sample Monte-Carlo reference for OPT, but the claim could not be *certified* because its mutation test did not cleanly break the property.

Quoted evidence (`results/claim2.json`):
- `"worst_gap": 0.0045302300672041`, `"all_within_eps": true`
- Example, n=4, eps=0.3: `"alg": 0.5945257115536037`, `"opt_ref": 0.5990559416208078`, `"gap": 0.0045302300672041`, `"ok_strict": true`
- `"mc_se": 0.00097…` — the reference itself is MC-estimated, so the ~4.5e-3 gap is well within noise but the comparison is approximate.

**Mutation test** (halve trace(Sigma) budget): `"property_breaks": false`. The halved-budget allocation still landed within the ε tolerance (e.g. n=4, eps=0.2: `"alg": 0.4235965186301658`, `"opt_ref": 0.5990559416208078`, `"gap": 0.175459422990642` — gap 0.175 < eps 0.2, so `"ok": true`). Because perturbing the setup did **not** falsify the guarantee, the test cannot distinguish the stated PTAS from a weaker result; combined with the MC-approximated reference, the claim remains **inconclusive** rather than verified.

---

---
<!-- trackio-cell
{"type": "markdown", "id": "cell_<built-in function hash>", "created_at": "2026-01-01T00:00:00+00:00", "title": "Claim 2 \u2014 PTAS for correlated VarAlloc (Theorem 1.2, \u00a71.2)"}\n-->
