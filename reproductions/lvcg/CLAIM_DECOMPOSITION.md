# Claim decomposition

## C1 - Fixed ECG/VCG geometry (2 points)

Recover a three-dimensional latent trajectory from visible ECG leads with the
regularized pseudoinverse, then project it through fixed lead directions.
Full credit requires executable official code and a successful independent
numerical conformance check.

## C2 - Model and training objective (2 points)

Exercise the released GRU training path and compare its implemented objective
with Equation 7, including masked-lead, VCG-beat, and temporal terms.

## C3 - Six-dataset linear probing (2 points)

Check split integrity and reproduce the PTB-XL, CPSC 2018, and Chapman
label-ratio results from an official checkpoint and released evaluation
configuration.

## C4 - Multi-lead reconstruction (2 points)

Regenerate CPSC 2018 and PTB reconstruction MSE/MAE for both 3-to-12 and
5-to-12 settings.

## C5 - Geometry and component ablations (2 points)

Regenerate fixed/learnable/shuffled geometry and bottleneck/temporal/VCG-loss
ablations from released configurations and outputs.

## C6 - Non-cardiac detection (2 points)

Reproduce chronic-kidney-disease, diabetes, and sepsis AUC on the paper's
MIMIC-IV-ECG-Ext-ICD dataset at 1%, 10%, and 100% label fractions.
