# HANDOFF — onlinecox_colsa (ORID TvLOddG9Ay)

Paper: "Online Learning and Inference for Cox PH via Renewable Sieve Estimation" (COLSA).
Code: `CollaborativeInference/COLSA` (R) vendored in `repo_ref/`. PDF: `../17342_Online_Learning_and_Infe.pdf`, text in `paper_text.txt`.

## Environment
- R 4.6.1 at `C:\Program Files\R\R-4.6.1\bin\Rscript.exe` (not on PATH).
- Personal lib: set `R_LIBS_USER=...\onlinecox_colsa\.rlib`. Deps: survival, splines2, statmod, MASS, flexsurv, coxphSGD.
- Source files, not installed pkg → must `library(statmod)` (gauss.quad.prob) manually.

## Claim 1 — REPRODUCED (done)
`reproduce_claim1.R` → `runs/claim1_console.txt`, `claim1_result.rds`. COLSA matches README to
every digit; vs pooled coxph max|coef diff|=0.0021, SE ratio ∈[0.983,1.021]; constant memory
(9544→10120 bytes over 6 batches). See REPORT.md.

## Claim 2 — UPGRADE IN PROGRESS (full Figure-2 panel)
`reproduce_claim2_full.R` adds the Online baseline (eha::pchreg piecewise-constant PH = the
Online method's model class, with the paper's adaptive partition J0=5/split r=2, refit on
cumulative data = conservative upper bound on true streaming accuracy) + vectorized DGP for
large N. Runner `run_full_panel.ps1` sweeps K=6/20/50/200 (reps 200/120/60/25), saves
runs/claim2full_K<K>.rds. Smoke (3 reps): K=6 COLSA 0.0744 best; K=200 COLSA 0.0166 ≈ Oracle
0.0179, beats Online 0.0188, Meta 0.0309 (cov 0.889), SGD 0.130. Full panel running (bg).
TODO after it finishes: rebuild claim-2 page with the 4-K table + republish logbook.

## Claim 2 — REPRODUCED (vs SGD + Meta baselines)  [superseded by full panel above]
Results: K=6/200reps COLSA RMSE 0.1157≈Oracle 0.1156, beats Meta 0.1172 & SGD 0.3255;
K=50/40reps COLSA 0.0498≈Oracle 0.0501, Meta degrades to 0.0574 (coverage 0.94→0.90).
Gaps: Wu-2021 "Online" not reimplemented; SGD(Offline) not an online method; fewer reps/K
than paper. See REPORT.md "Claim 2".

### earlier notes
`reproduce_claim2.R` — faithful Appendix B DGP (mixture Weibull, MVN covariates, Exp(6) censoring
~12% events). Methods: Oracle (coxph), COLSA (their pkg), Meta (inverse-variance weighting),
SGD + SGD(Offline) (coxphSGD). Online (Wu 2021) NOT implemented → documented gap.
Params via env: `K` (default 6), `REPS` (default 200). Saves `runs/claim2_K<K>.rds`.
- Smoke K=6/8reps: COLSA RMSE 0.095 ≈ Oracle 0.097, beats SGD 0.426; Meta competitive at K=6 (0.088).
- Running: K=6/200reps (background). TODO: K=50 trend point (fewer reps) to show COLSA tracks Oracle
  while Meta/SGD degrade as batches shrink — that is the "outperforms all" regime in Fig 2.
- Pace ~7 min / 50 reps at K=6.

## Not published (Kate's rule: no publish without her word).
