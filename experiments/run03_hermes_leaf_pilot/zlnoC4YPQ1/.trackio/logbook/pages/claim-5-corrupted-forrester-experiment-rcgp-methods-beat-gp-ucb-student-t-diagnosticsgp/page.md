# Claim 5 — corrupted Forrester experiment: RCGP methods beat GP-UCB / Student-t / DiagnosticsGP

---
<!-- trackio-cell
{"type": "markdown", "id": "cell_82f252f5cd00", "created_at": "2026-08-02T04:30:00+00:00", "title": "Claim 5 \u2014 corrupted Forrester experiment: RCGP methods beat GP-UCB / Student-t / DiagnosticsGP"}
-->
**Source:** Sec 5.3 and Figure 1 (+ configuration in Appendix I.4.2).
**Verdict: toy** — reproduced with a from-scratch CPU proxy of the paper's BoTorch pipeline
(fixed RBF hyperparameters instead of weighted LOO-CV fitting, grid argmax, own DiagnosticsGP).
Direction of the claim reproduces under the paper's recommended settings, but it is *not*
statistically resolved against GP-UCB and it flips under another configuration the paper endorses.

Setup: 1-D Forrester, σ_noise² = 1, 5 uncorrupted quasi-random initial points, 10 seeds,
greedy clairvoyant adversary (low = −10, high = 25), budget T_c = ⌈T^{1/3}⌉.
Two settings, because the main text and the appendix disagree on the adversary radii:
A = Sec 5.3 text (T = 100, near 0.2 / far 0.5, T_c = 5); B = Appendix I.4.2 (T = 30, near 0.1 /
far 0.4, T_c = 4). Mean cumulative regret over the 10 seeds (± s.e.):

| method | A (T=100) | B (T=30) |
|---|---|---|
| FC-RCGP-UCB (L = 1.96, T_c = 0 in β) | **219.27 ± 27.40** | **148.42 ± 8.10** |
| A2-RCGP-UCB (L = 1.96, T_c = 0 in β) | 224.62 ± 29.18 | 151.47 ± 8.51 |
| GP-UCB | 247.37 ± 33.34 | 157.59 ± 8.28 |
| Student-t Process UCB | 438.82 ± 17.58 | 191.62 ± 2.99 |
| DiagnosticsGP | 980.93 ± 60.80 | 255.02 ± 17.95 |
| FC-RCGP-UCB (L = 1.96, adaptive T_c estimate) | 793.61 ± 2.41 | 235.02 ± 3.26 |
| A2-RCGP-UCB (L = 1.96, adaptive T_c estimate) | 778.99 ± 3.82 | 215.22 ± 6.03 |

* Under the App I.1 configuration "c = 1, L = 1.96, T_c = 0 inside β_t" (both explicitly endorsed
  by the authors) the claimed ordering holds in **both** settings: RCGP < GP-UCB < Student-t <
  DiagnosticsGP. The advantage over GP-UCB is **11.4 %** (A) / **5.8 %** (B) of mean regret — well
  inside one standard error, so 10 seeds do not resolve it; the advantage over Student-t and
  DiagnosticsGP is large and unambiguous.
* Under the *other* configuration the paper endorses (T_c estimated as the number of points outside
  the plateau, which drives Ψ(T_c) and hence over-exploration), both RCGP methods become **worse
  than every baseline except DiagnosticsGP** (793.61 / 778.99 vs GP-UCB 247.37). The claim is
  therefore configuration-sensitive in this reimplementation.
* Discrepancy with the paper: Sec 5.3 states GP-UCB suffers the *highest* cumulative regret; here
  DiagnosticsGP is the worst baseline in both settings (980.93 / 255.02) — most likely because my
  DiagnosticsGP uses a MAD z-score filter rather than the paper's variational Student-t-likelihood
  GP detector.
* Uncorrupted control (Fig 3, T = 30, 10 seeds): A2 34.70, GP-UCB 34.99, FC 40.90, Student-t 42.29,
  DiagnosticsGP 233.43 → the Fig 3 statement (RCGP ≈ GP-UCB, better than Student-t/DiagnosticsGP)
  reproduces.

**Mutation test:** setting L = ∞ (plateau everywhere ⇒ the RCGP degenerates to a plain GP) makes
both RCGP curves land **exactly** on GP-UCB (247.37312941 in A, 157.58672285 in B, equal to
GP-UCB to <1e-9) and worsens their regret relative to the intact algorithms — so the improvement is
attributable to the P-IMQ down-weighting mechanism and to nothing else in the pipeline.

---
