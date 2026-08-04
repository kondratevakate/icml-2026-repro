## Honest notes

- These are THEORETICAL claims (closed-form bounds / a decomposition), not empirical statements requiring a full-width NN training run. The reproduction evaluates the paper's bound *expressions* and their scaling / regime-sufficiency directly (method A) and on synthetic XOR-cluster data via the paper's own kernel-regime forgetting formula (method B). No GPU, no full network training.
- Vanishing is ASYMPTOTIC (o_d(1)): under the regime every bound decreases with d as a poly-logarithmic factor (1/sqrt(log d), 1/log d, 1/log^2 d), so at moderate d the bounds are still O(1) in magnitude; they tend to 0 only as d -> infinity. This matches the paper's o_d(1) statements.
- Term 1 of Theorem 2.1 under the regime equals Theta(1/sqrt(polylog d)) (the d shows up only through the polylog hidden in n=O~(d^2 K)); it vanishes with d only via that poly-log factor, consistent with the paper's K=O~_d(1) assumption.
- Theorem B.1's improved gap uses the learning RATE eta (not eta*T) in both its prefactor and exponent; with that correction it scales as eta d^2 log^3(T)/n (poly-log in T), strictly slower than Theorem 2.3's linear-in-T dependence.

Overall status: OK

---
<!-- trackio-cell
{"type": "markdown", "id": "cell_<built-in function hash>", "created_at": "2026-01-01T00:00:00+00:00", "title": "Honest notes"}\n-->
