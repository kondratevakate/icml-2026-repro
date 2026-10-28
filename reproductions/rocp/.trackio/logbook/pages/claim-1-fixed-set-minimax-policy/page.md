# Claim 1: fixed-set-minimax-policy


---
<!-- trackio-cell
{"type": "markdown", "id": "cell_e796116c7ab1", "created_at": "2026-07-28T15:58:37+00:00", "title": "Claim 1: fixed-set-minimax-policy"}
-->
**Anchored claim (verbatim).** "The paper proposes a decision-theoretic framework that minimizes expected loss (risk) against a worst-case distribution consistent with a prediction set's coverage guarantee, characterizing the minimax-optimal policy for a fixed prediction set."

**Verdict -- VERIFIED.**

For action `a`, fixed prediction set `S`, and allowed miscoverage `alpha`, let `l_in` and `l_out` be the largest action losses inside and outside `S`. The adversary's exact primal optimum is

```text
l_in + alpha * max(l_out - l_in, 0).
```

This follows because its in-set mass is constrained only by `Q(S)>=1-alpha`: if `l_out>l_in`, the adversary puts masses `1-alpha` and `alpha` on the two maximizers; otherwise it puts all mass on the in-set maximizer. Minimizing this value over actions reproduces Eq. (4) and Theorem 2.2.

`audit_claim1.py` independently solves the defining adversarial problem as a finite linear program. It exhaustively covers the complete 2-label/2-action integer-loss family, then adds 120 seeded asymmetric 3-label cases and five alpha values. **All 28,950 action-value comparisons pass**, maximum absolute closed-form/LP error is `1.42e-14`, and there are zero minimizing-action-set failures.

**Destructive control.** With losses `[[1,2],[1,2],[100,2]]`, set `S={0,1}`, and `alpha=0.05`, the naive max-min action selects action 0 from its in-set value `1`, but its robust value is `5.95`; ROCP selects action 1 with robust value `2`. Removing the out-of-set penalty therefore changes the policy exactly as the theory predicts.
