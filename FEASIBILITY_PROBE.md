# Feasibility Probe: Code, Data, Runnable State

Probe date: 2026-07-28. This document records observed state only. It is not a
reproduction verdict and does not authorize publication.

## Verdicts

| Paper | Code | Data/artifacts | Runnable evidence | Verdict | Next gate |
|---|---|---|---|---|---|
| Risk-Optimal Conformal Prediction (`VAXW59dyfk`) | Official repository pinned at `3ee0cf6`. | 20 cached COVID probability files plus BDD `scores`/`true_bits` `(10000,3)`. | Full CPU run completed: 20 COVID seeds under two loss matrices and 20 BDD splits; claim validator passes. | **COMPLETED; UPDATE READY** | Update the existing canonical Space after final logbook validation. |
| Subgroup Discovery with the Cox Model (`3tlaoBZrSg`) | Official repository located and pinned at commit `96725eec197ce3d08d24560d4bd2e697b3bfdbdb`. | Synthetic generator, clinical/NASA data loaders, and author result directories are present; author outputs were excluded from evidence. | Fresh 10-seed DDGroup/Base Synth-Nonlinear run completed all 20 jobs and reproduced Table 2 to rounding; C1/C2 audits passed. | **COMPLETED AND PUBLISHED** | Await judge verdict on the canonical Space; C3/C5/C6 remain not attempted. |
| SwitchCraft (`YtqaHnqv8c`) | Official repository cloned. Five supplied task YAMLs and direct `switchcraft.py --config ...` entrypoint present. | Configs and motif sources in repository; model weights/CCD ligand database are a separate download. | Static code is complete, but `nvidia-smi` is unavailable on this host. README requires CUDA PyTorch and Boltz weights. | **GO on GPU only** | On GPU, install CUDA torch, Boltz and weights; smoke-test `tasks/pos_allostery.yaml` with one design before allocating the three-primitive run. |
| Treatment Allocations (`9kWVQtpvJi`) | No public implementation found; code-index result says request code. | Synthetic DGP is specified in the paper; STAR and IST datasets are restricted. | No author entrypoint. | **GO only as theorem/simulation project** | Build a minimal independent certificate plus synthetic DGP. Exclude STAR and IST from expected-score calculation. |
| PyHealth 2.0 (`gMLVFN9hl8`) | Local `pyhealth_src` checkout and a custom probe script exist. | Local MIMIC-III demo path exists. | The prior log shows dataset loading begins; full task/model/interpretability completion is not yet evidenced. | **HOLD as repair** | Finish the existing CPU pipeline before estimating more than four points. Do not substitute demo results for the paper's MIMIC-IV throughput tables. |
| A Machine-Learned Comorbidity Index (`C6ZTjSXbz7`) | No official executable repository found; arXiv source fully specifies DeepSets+nHSIC and includes all proofs. | Full MIMIC-III/IV core tables and patient-level learned scores are not yet local; aggregate risk-curve CSVs are insufficient for table claims. | Independent C1 method audit and 100-seed C5 rank-one/threshold audit pass on CPU. | **GO after data; theory ready** | Build exact patient-disjoint cohorts when downloads finish; do not publish before C2-C4/C6 have patient-level evidence. |
| Safety Generalization in Diabetes (`kSUGLBHd0T`) | Official GlucoSim and GlucoAlg repositories cloned at commits `f5662cc` and `50d3134`. | HF publishes 72 policy-model repositories; the inspected CPO repository has three seeds and four checkpoints per seed. Transition data and the learned dynamics checkpoint required by the predictive shield were not found. | GlucoSim passes 22/22 tests and a 288-step, 24-hour CPU rollout. The official evaluator cannot import on this Windows host because vendored OmniSafe eagerly imports MuJoCo, whose DLL is blocked by application policy. | **HOLD for paper-level claims** | On GCP, run unshielded policies first. Proceed to the headline shield claim only if the authors provide the missing transition/dynamics artifacts or their training path is acceptably short. |

## ROCP Environment

- Clone: `%TEMP%/icml-repro-probe-rocp`.
- Environment: Python 3.11, PyTorch 2.10.0 CPU, NumPy 1.26.4, scikit-learn 1.8.0.
- `evaluation.py --help` and `evaluation_bdd.py --help` pass.
- The COVID cache is 20 seeds; the BDD cache has the required keys and shape.
- Full-scale execution was not certified in this Windows probe. The direct evaluator has a valid small real-data path, while the detached full-seed process stalled with no stdout/stderr and was stopped after verifying its command line.

## Safety Generalization in Diabetes Environment

- Clones: `%TEMP%/icml-repro-probe-glucosim` and `%TEMP%/icml-repro-probe-glucoalg`.
- Inspected model: `safe-diabetes-benchmark/safe-diabetes-t1d-adolescent-cpo`.
- GlucoSim CPU verification: 22 tests passed in 54.49 seconds.
- Seed-42 basic rollout: 288 steps, 67.5% time in range, zero hypoglycemia,
  32.5% hyperglycemia, and safety cost 1413.46.
- The policy checkpoint is small and downloadable, but its HF layout
  (`checkpoints/seed*/`, `config/seed*/`) differs from `eval_run.py`'s expected
  `saved_models/.../seed*/torch_save/` layout and needs deterministic mapping.
- `shield/predictive_shield.py` requires
  `saved_files/dynamics_predictor/fe_b5_p24_h24/1` plus aggregated transition
  data. Neither artifact is supplied by the inspected policy repository or the
  GlucoAlg checkout.
- The predictive-shield code also compares a full patient identifier such as
  `adolescent#001` against bare cohort names when computing its dynamics index.
  This requires an execution audit before trusting cross-cohort results.

## Decision

Start the full ROCP reproduction first, using a persistent Linux/GCP run for
the 20-seed COVID and BDD evaluations. It remains the best code/data-ready
medical-relevant candidate. Diabetes is suitable for a GCP unshielded-policy
probe but is not yet a high-throughput route to the headline shielding result.
Cox has now completed its independent local and publication path. MLCI is the
next local data-bound target: its C1/C5 audits are ready, while C2-C4/C6 wait
for the full MIMIC tables. SwitchCraft waits for GPU.
