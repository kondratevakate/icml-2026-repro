# Reproduction task — Efficient Bayesian Inference from Noisy Pairwise Comparisons

OpenReview: https://openreview.net/forum?id=NinueNAODD
Area: Probabilistic Methods
Anchored claims (6):

1. BBQ's EM algorithm has closed-form updates that guarantee monotonic improvement of the likelihood and convergence to a stationary point, in contrast to gradient-based methods like Crowd-BT which lack such convergence guarantees (Section 3.3).

2. BBQ converges within seconds on all tested datasets, whereas Crowd-BT takes approximately 15 minutes on the HUMAINE dataset (105,220 comparisons) despite BBQ using an unoptimized plain NumPy implementation (Figure 3).

3. On the unscreened IHQ dataset, BBQ achieves 61.92% top-1 agreement with final rankings, compared to 33.15% for Crowd-BT and 24.32% for Bayes-BT (Table 1).

4. BBQ ranks first in Kendall's Tau agreement on 5 of 8 evaluated datasets and second on the remaining three, and achieves 100% top-1 agreement on the MT-Bench, WD, and HiFiC datasets (Table 1).

5. At the 99% confidence level, BBQ and Crowd-BT both show well-calibrated Type I error rates of about 1%, while Bayes-BT is overly conservative at about 0.1% (Appendix G, Figure 6).

6. BBQ's rater-quality parameter (mixture weight q_r modeling whether a rater follows Bradley-Terry or guesses randomly) correlates with Pearson r=0.724 with rater agreement to final rankings on the unscreened IHQ data, enabling identification of unreliable crowdsourced raters without a separate screening procedure (Figure 2, Equation 2).

## Your job (autonomous)
Reproduce each anchored claim on CPU with numpy/scipy/sympy (no GPU). For every claim:
- Write `verify_claim<N>.py` that computes the claim's quantity from first principles.
- Save numeric result to `results/claim<N>.json`.
- Run a MUTATION test (perturb the setup; the claimed property must break or shift) for every verified claim.
- Write verdict: verified / falsified / toy / inconclusive (honest if data/GPU blocks full repro).
- Record exact source (section/equation/theorem) and a reproducible seed.

Hard budget: 8h total, 4h soft, 2h per claim. Write `logbook.md` when done.
