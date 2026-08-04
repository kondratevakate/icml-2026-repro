## Claim 3 — Entropic penalty ε_t = ηt^{−η} is lower order (`verify_claim3.py` → `results/claim3.json`)

**Anchor:** v2 Lemma C.3 (Carlier et al. 2023): Ent.(µ,ν,c,ε) − Kant.(µ,ν,c) ≤ Cε log(1/ε),
plus the entropic term of Thm 4.1.

**Setup.** Discrete OT on m = 40 points of [0,1] with the 1-Lipschitz cost |x−y|; exact
Kantorovich value by LP (`highs`, value `0.037724`), entropic value by log-domain Sinkhorn
to tolerance 1e−13, ε swept over {0.5 … 0.005}.

**Numbers.** gap = Ent(ε) − Kant is non-negative and monotone in ε
(`0.2628, 0.2024, 0.1439, 0.09164, 0.04400, 0.02312, 0.01163`); the ratio
gap/(ε log(1/ε)) stays in `[0.439, 0.758]` (spread 1.73) — i.e. the Cε log(1/ε) rate holds
with C ≈ 0.76 and is not tight from below, exactly as an upper bound should behave.
Cumulating with ε_t = ηt^{−η}, η = 0.75: `11.6, 37.8, 101.7, 245.8` at T = 10²…10⁵, matching the
predicted shape T^{1−η} log T to within a spread of `1.51`, and dominated by the √T log T
trajectorial term at all T.

**Mutation.** Constant ε (η = 0) makes the cumulated approximation error exactly **linear**:
`17.5, 174.6, 1746, 17462`, log-log slope `1.0000`.

**Verdict: verified.** Both ingredients of the claim reproduce: the ε log(1/ε) convergence rate
for a Lipschitz cost, and the sublinearity of the cumulated penalty under the decaying schedule.

---
<!-- trackio-cell
{"type": "markdown", "id": "cell_<built-in function hash>", "created_at": "2026-01-01T00:00:00+00:00", "title": "Claim 3 \u2014 Entropic penalty \u03b5_t = \u03b7t^{\u2212\u03b7} is lower order (`verify_claim3.py` \u2192 `results/claim3.json`)"}\n-->
