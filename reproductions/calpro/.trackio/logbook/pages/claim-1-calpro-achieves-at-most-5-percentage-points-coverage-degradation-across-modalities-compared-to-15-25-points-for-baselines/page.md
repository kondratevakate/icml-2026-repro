# Claim 1-2 (A2/A4): Theorem 4.2's coverage guarantee is refuted; the modality-shift degradation number (A4) was not actually tested

---
<!-- trackio-cell
{"type": "markdown", "id": "cell_d975239dc91c", "created_at": "2026-07-18T16:28:12+00:00", "title": "Claim 1-2 (A2/A4): Theorem 4.2's coverage guarantee is refuted; the modality-shift degradation number (A4) was not actually tested"}
-->
## Correction notice (2026-07-20)

An earlier version of this page reported a "Claim 1 reproduced" verdict against anchored claim **A4** ("Under X-ray to cryo-EM modality shift, CalPro's coverage degrades by only about 3.8 percentage points, versus 14-18 points for pLDDT-based and vanilla conformal baselines," Section 5.2, Table 3). **That verdict was invalid and is withdrawn.**

What the earlier scripts (`verify_calpro.py`, `verify_calpro_phase2.py`) actually tested was a self-constructed 1-D synthetic heteroscedastic regression, `y = g(x) + sigma(x)*eps`, with a hand-made covariate shift from `x in [-2,2]` to `x in [2,3.5]`. This does **not** correspond to A4 on any dimension that matters:

- **No proteins, no modality shift.** A4 is about a real shift between experimental techniques (X-ray vs cryo-EM structures); the scripts used an arbitrary shift in a synthetic covariate's support, chosen by the person reproducing the paper, not derived from the paper's data.
- **No pLDDT baseline.** A4's main comparator (14-18pp degradation) is AlphaFold's own confidence estimate. The scripts compared against a degree-3 polynomial least-squares baseline — an unrelated object.
- **No CalPro.** CalPro is a geometric graph evidential head plus a differentiable conformal calibration layer trained end-to-end, plus domain priors (disorder, flexibility) imposed as soft constraints — the priors are the paper's central contribution (Table 3 attributes the 5.3 -> 3.8pp improvement to them). None of this is implemented in the scripts; what was implemented is closest to the paper's own "Evidential-only" ablation row (7.7pp).
- **Wrong shift regime.** The baseline in the scripts degrades by 58.9pp, far outside the paper's reported 14-18pp range for its own baselines, so even the qualitative comparison is not informative about A4's specific claim.

**Honest status of A4: NOT TESTED.** Neither the old positive verdict nor a naive negative re-run of the same synthetic setup would be a verdict on A4 — both would be judging the wrong experiment. Testing A4 for real would require open PDB/AlphaFold structures, an RMSD-based nonconformity score, and a real pLDDT baseline; this was assessed as feasible in principle (the data is open) but **not attempted**, because the RMSD/structural-alignment pipeline carries high silent-error risk against a target with no independent check available here, and because any result could not honestly be labeled a reproduction of "CalPro" itself, since the graph architecture and the paper's family/temporal-disjoint train split are not released. Flagged as future work, not undertaken.

## The real finding: anchored claim A2 (Theorem 4.2) is refuted

**A2:** "Theorem 4.2 provides structure-aware coverage guarantees via PAC-Bayesian bounds over ambiguity sets, formally controlling nonconformity drift under distribution shift (Section 4)."

Verification: text/theoretical audit of the paper's own Section 4, cross-checked against `paper.txt` (extracted via PyMuPDF; authenticity confirmed independently because Table 3 numbers in the extracted text — CalPro 3.8, pLDDT 18.4, Conformal-no-priors 14.1 — match the anchored claim). Numeric check: `audit_thm42.py`.

**(a) The bound is not structure-aware, and the paper says so itself.** Section 4.3 opens by stating the Theorem 4.2 bound is **agnostic to the presence of priors**. Structure/priors only enter Theorem 4.4, which bounds interval *width*, not coverage. The word carrying A2's main claim is contradicted by the paper's very next paragraph.

**(b) The shift term contains an unknowable quantity.** The shift term is `L_s * epsilon`, where `epsilon` is the radius of the ambiguity ball (Wasserstein / Levy-Prokhorov) presumed to contain the true test distribution D1. `epsilon` cannot be estimated without knowing D1 — exactly what the theorem is supposed to certify in advance. The paper's own workaround is to report the bound "for varying epsilon" (Sec. 4.4), which makes it a parametrized family of statements, not a guarantee. Nothing in it can certify "3.8pp on cryo-EM" ahead of time.

**(c) Empirically vacuous at realistic shift magnitudes.** Estimating both terms via the paper's own stated procedures (`L_s` by local finite differences on the calibration set; `epsilon` as W1 distance between calibration and shifted distributions), on the repo's existing synthetic toy setting:

| quantity | value |
| --- | --- |
| L_s (99th pct local finite differences) | 9.82 |
| epsilon (W1, calibration to shifted) | 4.02 |
| L_s * epsilon | 39.47 |

The coverage lower bound `1 - alpha - KL-term - L_s*epsilon` goes deeply negative at every tested tau (0.80, 0.90, 0.95) — i.e. it "guarantees" coverage is at least roughly -38, which is formally true and useless. **Caveat, stated honestly:** this numeric check used the existing synthetic toy setting, not the real X-ray/cryo-EM data, so it demonstrates that the bound *goes vacuous at moderate shift magnitudes in general*, not the bound's specific numeric value in the paper's own reported experiment.

**(d) Two assumption problems, one self-contradictory.** Assumption 4.1's Lipschitz requirement covers only pairs with both points inside `supp(D0)`, but the theorem is applied to D1 under modality shift, which pushes mass outside `supp(D0)` — the assumption does not cover the regime in which the theorem is used. Separately, the theorem requires an i.i.d. calibration sample from D0; calibration is per-residue, and the paper's own Section 5.3 states residue errors are structured (dependent) — directly undermining the theorem's own i.i.d. premise.

**Open question, not a confirmed error:** the printed complexity term `(KL(rho||Pi) + log(1/delta)) / (2*n_cal)` lacks the square root that standard PAC-Bayes bounds (McAllester, Maurer) require. This may be a typo, but Appendix C (which would contain the derivation) was not present in the extracted text, so this is flagged as an open question rather than a confirmed error.

**Verdict — Claim A2 refuted.** The theorem's own text contradicts its "structure-aware" framing, its shift term is not estimable ahead of time, it goes vacuous under realistic shift magnitudes in a controlled numeric check, and its assumptions do not cover the regime it is applied to. Refutation is scored the same as confirmation on this challenge; this is the primary, load-bearing result of this page.
