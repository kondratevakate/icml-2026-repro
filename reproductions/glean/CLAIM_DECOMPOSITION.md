# Claim decomposition

## C1 - Guideline-grounded evidence accumulation and calibration

GLEAN aggregates per-step guideline ratings, accumulates discounted logits,
and maps the resulting signal to correctness probabilities with Bayesian
logistic regression. Appendix A gives a capacity-controlled Brier-risk bound.

Full evidence requires either a released implementation or an independent
implementation check of the equations and proof.

## C2 - Data and trajectory construction

The experiment uses three diseases from the MIMIC-IV-Ext Clinical Decision
Making workflow and the `epfl-llm/guidelines` corpus. For each of two Qwen
backbones, the paper reports 2,000 trajectories with a balanced correct and
incorrect split.

Full evidence requires the exact case selection, trajectory texts, labels,
sampling seeds, retrieved guidelines, and judge ratings.

## C3 - Main verification performance

Table 1 reports AUROC, Risk@0.5, ECE, and Brier for three diseases and two
backbones. The headline says GLEAN improves AUROC by about 12% and reduces
Brier by about 50% relative to the best baseline.

Full evidence requires row-level labels and predictions for GLEAN and every
baseline.

## C4 - Active verification and component ablations

The paper attributes gains to guideline expansion, differential checks,
process context, guideline content, evidence accumulation, and calibration.

Full evidence requires triggered-case identities, added guidelines, revised
ratings, and per-ablation predictions.

## C5 - Best-of-N and computational efficiency

GLEAN reportedly raises Qwen3-30B diagnostic accuracy from 58.94% to 78.3% at
N=16 and offers a favorable model-call/token trade-off.

Full evidence requires the 16-candidate groups, verifier rankings, selected
answers, and measured call/token logs.

## C6 - Clinician utility study

Three clinicians reportedly reviewed 50 trajectories and rated usefulness,
interpretability, and clinical utility, with Cohen's kappa of 0.78 for error
identification.

Full evidence requires the study instrument, anonymized ratings, annotations,
and analysis code.

