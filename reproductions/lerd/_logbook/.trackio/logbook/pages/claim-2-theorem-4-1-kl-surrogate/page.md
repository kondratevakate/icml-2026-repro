# Claim 2: Theorem 4.1 KL surrogate


---
<!-- trackio-cell
{"type": "markdown", "id": "cell_e2579be76bc0", "created_at": "2026-07-22T12:42:40+00:00", "title": "Claim 2: Theorem 4.1 KL surrogate"}
-->
**Paper claim (Theorem 4.1).** LERD derives a tractable integral-rate surrogate
that replaces the intractable KL divergence between the EPDE-induced path law and
the dLIF prior, with a formal upper bound guaranteeing training stability.

**Verdict: REPRODUCED (numerical audit).**

The theorem's IVP identity was audited independently (`thm41_audit.py`, numpy +
scipy.quad, CPU, no GPU - a theorem needs a numerical audit, not a GPU run).
Concrete `q`, `r`, and `S` were instantiated per the hypotheses and the
event-prior KL was computed two independent ways: directly as
`K_S = integral_0^S q log(q / p_r^S) dt`, and via the theorem's change of variables
`m = -exp(-t)`, `K_S = integral_{-1}^{-exp(-S)} g(m) dm = G(-exp(-S))` (Eq 9).

**Eq 9 holds to machine precision** across all test configurations:

| Case | K_S direct | K_S via IVP (Eq 9) | abs diff |
| --- | --- | --- | --- |
| const r, trunc-exp q, S=3 | 0.39124380 | 0.39124380 | 9.4e-16 |
| linear r in [1,4], S=2.5 | 0.38755709 | 0.38755709 | 1.8e-15 |
| sinusoidal r in (0.5,3.5), S=4 | 2.16351647 | 2.16351647 | 1.1e-13 |

Also verified: the normalization identity `Z_S = 1 - exp[-R(S)]`; `KL >= 0` (Gibbs);
and the S -> inf limit (Eq 10) converges (K_S = 0.221, 0.533, 0.772, 0.807 at
S = 2, 4, 8, 16).

**Both stated hypotheses are shown load-bearing (controls).**
- Relaxing `r >= a > 0` with an integrable `r(t)=1/(1+t)^2` gives `R(inf) < inf`, so
  `Z_inf = 1 - exp[-R(inf)] -> 0.632 != 1` - the theorem's `Z_inf = 1` needs `r`
  bounded away from 0.
- Using the wrong normalization (`Z = 1` instead of `Z_S` on a short window) makes
  the KL of a distribution with itself spuriously `1.71` instead of `0` - the `Z_S`
  normalization is required for a valid KL.

This is a numerical audit of the stated identity, not a proof replacement. Script:
`thm41_audit.py`.
