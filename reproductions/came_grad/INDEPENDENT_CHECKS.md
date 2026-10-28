# CAME-Grad independent checks

Date: 2026-07-29

## Verdicts

| Claim | Verdict | Points |
|---|---|---:|
| C1 three-stage equation mechanics | VERIFIED AT EQUATION LEVEL, WITH QUALIFICATIONS | 2 |
| C2 SDE resolution/convergence/flat-minimum mechanism | NOT ESTABLISHED BY PRINTED EQUATIONS | 0 |
| C3 eight-backbone clinical gains | INCONCLUSIVE — NOT RERUN | 0 |
| C4 multi-task optimizer comparison | INCONCLUSIVE — NOT RERUN | 0 |
| C5 stage ablations | INCONCLUSIVE — NOT RERUN | 0 |
| **Total** |  | **2/10** |

The outcome matches the frozen forecast.

## C1

An independent NumPy/SciPy implementation of equations 8–14 was tested on 500
random three-task problems:

- maximum trust-region boundary residual: `4.44e-16`;
- maximum Stage-2 target-norm relative error: `1.90e-12`;
- maximum fusion-equation residual: `0`;
- maximum simplex-sum residual: `4.97e-14`;
- with `nu=1, kappa=1`, the final update equals the original joint gradient
  exactly.

This verifies the printed three-stage algebra. It does not verify the withheld
PyTorch implementation or clinical performance.

Two qualifications were found:

1. Eq. 10 is undefined when the optimized dual gradient has zero norm. A
   symmetric pair `[1,0]`, `[-1,0]` produces the unaddressed `0/0` case.
2. “Adaptive Gradient Fusion” uses a fixed tuned hyperparameter `nu`; Algorithm
   1 and Eq. 13 contain no data-, conflict-, or time-dependent adaptation rule.

## C2

The trust-region identity establishes bounded displacement from the mean
gradient when Eq. 10 is defined. It does not by itself guarantee global
convergence or movement toward flat minima.

A covariance counterexample holds update norm exactly at 1 in two
distributions:

- constant updates have covariance trace 0;
- direction-varying updates have covariance trace 0.5.

Therefore magnitude restoration alone does not determine the gradient-noise
covariance `Sigma` in the paper's SDE. The claimed diffusion compensation
requires additional distributional assumptions or measurements.

## C3 source arithmetic

Independent parsing of the published TeX table cells reproduces:

- MIMIC-CXR mean CE difference: `0.023125`, rounded to 2.3 percentage points;
- IU X-Ray mean CE difference: `0.019`, or 1.9 percentage points.

This confirms table arithmetic only. It is not an experimental reproduction.

## Official-release audit

- `modules.CAME_Grad`: absent;
- dataset package: absent;
- `requirements.txt`: absent;
- CheXbert checkpoint path: placeholder `xxxxxxxx`;
- weights, processed splits and output logs: absent;
- the repository history contains no earlier copies of these files;
- README explicitly says the core optimizer is temporarily withheld.

All 13 released Python files parse, but `main_train.py --help` and
`main_test.py --help` fail on an undocumented `pycocoevalcap` dependency.
Importing `modules.trainer` fails on the absent optimizer.
