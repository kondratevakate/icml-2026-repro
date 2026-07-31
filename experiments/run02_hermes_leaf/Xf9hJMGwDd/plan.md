# Reproduction plan — Semi-knockoffs (ICML 2026, OpenReview Xf9hJMGwDd, arXiv 2601.23124)

Paper read: arXiv HTML v2. Official code cloned: https://github.com/AngelReyero/loss_based_KO
-> `official_code/src/semi_KO.py` (Semi_KO.fit/predict/score), `official_code/src/utils.py::knockoff_threshold`.

Environment: WSL2 Ubuntu, python3.12 venv, numpy/scipy/sympy only, CPU. **sklearn is NOT installed**
(task restricts to stdlib+numpy/scipy/sympy), so the paper's RF / NN / GradientBoosting black boxes and the
sklearn-shipped WDBC dataset are out of reach. This constrains claims 5 and 6 (see below).

## Claim-by-claim feasibility

| # | Claim (source) | Route | Feasible on CPU? |
|---|---|---|---|
| 1 | Thm 3.3 (Sec 3.1): nonparametric paired test on the two semi-knockoff imputations gives **valid p-values without a train-test split**, needing only nu_j, rho_j | Monte-Carlo over many seeds with **oracle** nu_j, rho_j (both available in closed form for a jointly-Gaussian (X,y) design). Check super-uniformity of the null p-value. Mutation = drop the symmetric two-sided sampling and use the HRT-style `l(m(Xtilde1)) - l(m(X))` statistic on the *same* data the model was trained on -> must break validity. | YES |
| 2 | Thm 3.4 (Sec 3.2): FDR(S_SKO) <= q | Monte-Carlo FDP averaged over many seeds, p=20, oracle nu/rho, knockoff threshold Eq. (1) copied from the paper/official `utils.knockoff_threshold`. Mutations: (a) remove the `1+` offset from Eq. (1); (b) break sign-exchangeability by using the biased no-split statistic. | YES |
| 3 | Thm 4.1 (Sec 4.2): for a null feature, `\|\|theta~^j - theta^\|\|_2 <= O_P(sqrt(log(1/delta)/n))` for l2-regularized ERM | **Exact** — ridge ERM of Eq. (2) has a closed form, so no optimizer noise. Check the two functional dependencies separately: (i) log-log slope in n must be ~ -0.5; (ii) the (1-delta)-quantile must grow ~ sqrt(log(1/delta)). Mutation: run the *same* estimator on a **non-null** coordinate -> the quantity must plateau at \|beta_j\|-ish instead of decaying. | YES (cheap, closed form) |
| 4 | Thm 4.3 (Sec 4.4): double robustness, loss difference decays at compound rate `O_P(a_n b_n)` | Requires controlled, independently-tunable error injection into both the predictive model and the sampler, plus a 2-D rate fit. Doable in principle but not within the remaining budget after 1/2/3/5, and the theorem is stated in the paper as supporting a *conjecture* of control. **STOP -> inconclusive.** | NOT within budget |
| 5 | Fig 4/5 (Sec 5.1): on simulated adjacent-support data SKO keeps type-I control **and has higher power than HRT**; derandomization with 5 permutations under masked correlation raises power | The data-generating process is fully specified in the paper (Gaussian, Sigma_ij=0.6^\|i-j\|, adjacent block of 0.25p important coords; masked-correlation setting in App F.4.2). But the paper's black box is a NN / GB / RF. With numpy only I can substitute a **ridge black box**. That is *not* the paper's configuration -> at best a mechanism-level check, verdict capped at `toy`. Run anyway because the split-vs-no-split power mechanism is model-independent in principle. | PARTIAL -> `toy` |
| 6 | Fig 6 (Sec 5.2): WDBC real data across RF / NN / GB | Real dataset + three sklearn models. sklearn unavailable; substituting a synthetic dataset would violate the hard rule. **STOP -> inconclusive.** | NO |

## Budget allocation
Claims 1, 2, 3 get full treatment (many seeds + mutation tests). Claim 5 gets a best-effort run explicitly
labelled `toy`. Claims 4 and 6 are refused with reasons.
