## Overall assessment

All six anchored claims reproduce on CPU: two are exact algebraic identities (1, 6), one is a
numerically confirmed convergence rate (3, C ≈ 0.76 for a 1-Lipschitz cost), and three are
high-probability regret bounds (2, 4, 5) that held in **every** simulated run (60/60, 125/125,
and all simulated (q,T) cells) with the certified rates matching the analytic exponents.
Every verified claim has a mutation that breaks it (non-unitary transform, Cauchy noise,
constant ε, broken feedback, frozen truncation order, shrunken β).

**Honest limitations.** (i) The paper is infinite-dimensional; all bandit experiments are
finite-dimensional/basis-truncated instantiations — they cannot falsify the infinite-dimensional
statements, only the finite instantiations of them. (ii) The regularised-optimism step (11) is
not numerically implementable in finite time (the paper says so itself, §6); we use the
extreme-point/finite-basis surrogate instead of an ε-optimal Sinkhorn plan. (iii) All bounds are
loose by one to two orders of magnitude on these instances, so this reproduces validity, not
tightness. (iv) Section/theorem numbers differ between the TASK anchors and the public v2 PDF;
the mapping is recorded per claim above.

---
<!-- trackio-cell
{"type": "markdown", "id": "cell_<built-in function hash>", "created_at": "2026-01-01T00:00:00+00:00", "title": "Overall assessment"}\n-->
