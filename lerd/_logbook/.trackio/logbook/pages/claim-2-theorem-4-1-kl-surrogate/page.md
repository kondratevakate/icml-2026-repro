# Claim 2: Theorem 4.1 KL surrogate


---
<!-- trackio-cell
{"type": "markdown", "id": "cell_e2579be76bc0", "created_at": "2026-07-22T12:42:40+00:00", "title": "Claim 2: Theorem 4.1 KL surrogate"}
-->
**Paper claim (Theorem 4.1).** LERD derives a tractable integral-rate surrogate
that replaces the intractable KL divergence between the EPDE-induced path law and
the dLIF prior, with a formal upper bound guaranteeing training stability.

**Verdict: not attempted (numerical audit deferred).**

Per the challenge guidance, a theorem's expected reproduction is an independent
numerical audit: implement the setup, check the stated inequality holds to double
precision, and add a control that relaxes the conditions to show the bound
degrades. This requires implementing the EPDE path law and the dLIF-prior KL, which
are the same components blocked by the GPU-bound full model (Claim 4). The audit is
scoped for the follow-up GPU run and is not attempted here.
