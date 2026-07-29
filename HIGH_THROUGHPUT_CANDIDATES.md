# High-Throughput Medical Candidates

Selection date: 2026-07-28

This is the second-stage selection after the lexical scan. Expected score and
time are planning estimates, not judge predictions. A candidate proceeds only
after its PDF, repository, and exact claim scope pass the per-paper preflight.

## Execution update

- Subgroup Discovery with the Cox Model completed locally and was published on
  2026-07-29. The locked forecast is `5-6/12`, lower than the original `6-8`
  planning estimate because only C1, C2, and C4 have independent evidence.
- A Machine-Learned Comorbidity Index (`C6ZTjSXbz7`) is now the active local
  medical target. C1 and C5 audits pass; its four empirical claims wait for the
  full MIMIC-III/IV tables.
- Risk-Optimal Conformal Prediction completed all cached local runs: 20 COVID
  seeds under both loss matrices and 20 BDD splits. C1/C3/C4 now support a
  `6/8` update forecast for the existing canonical Space.

## Start Now

| Rank | Paper | Expected points | Estimated time | Expected points/hour | Why it is viable | Stop condition |
|---:|---|---:|---:|---:|---|---|
| 1 | Risk-Optimal Conformal Prediction (`VAXW59dyfk`) | 4-6 | 6-9 h | 0.7-1.0 | Official repo ships cached 20-seed COVID probabilities and BDD scores; no detector or classifier training is required, but Algorithm 1 needs an independent test-point-inclusive coverage audit. | Stop if the paper-faithful calibrator cannot pass exhaustive exchangeability tests or the full cached runs fail. |
| 2 | Subgroup Discovery with the Cox Model (`3tlaoBZrSg`) | 6-8 | 4-6 h | 1.0-2.0 | Another judged logbook already obtained two verified claims and one full falsification. The exact synthetic and clinical-table paths are known. | Stop if the official artifacts cannot be rerun from scratch rather than read as precomputed outputs. |
| 3 | SwitchCraft (`YtqaHnqv8c`) | 6 | 3-5 GPU h | 1.2-2.0 | A prior judge accepted two end-to-end method/loss claims and a partial primitive claim. The GPU path is compact: three primitives completed in about 15 minutes in the prior logbook. | Do not enter the cpGFP 13,858-design screen or claim the SAM mechanism without the full pipeline. |
| 4 | Treatment Allocations with Risk Control (`9kWVQtpvJi`) | 5-7 | 4-6 h | 0.8-1.75 | A high-quality prior logbook verified the finite-sample certificate and synthetic experiment. A direct Lemma 4.1 audit can plausibly lift the score beyond that baseline. | STAR and IST claims require restricted data; leave them explicitly inconclusive. |
| 5 | PyHealth 2.0 (`gMLVFN9hl8`) | 4 published; 5 upside | ~2 h source audit | 2.0 submitted | C1 is independently verified from the paper inventory and pinned release; C4 is falsified because the anchored `7/24/51` attribution conflicts with Table 2's `34/27/51`. Canonical Space commit `101ec071fcd5ca2b407dd8573e6e1577f590aa09` is published and remote hashes match. | Await judge. Keep C2/C3 blocked on full MIMIC-IV v2.2/GCP and C5 inconclusive; do not turn author tables or demo runs into claimed reproductions. |

## Reserve Queue

| Paper | Current estimate | Why not first |
|---|---:|---|
| MedMamba (`qPqJH0heR0`) | 4-6 pts / 5-8 h | Code is public, but the claim is a five-dataset, nine-baseline SOTA table. Start only after confirming all Medformer datasets and scripts run unchanged. |
| Off-Policy Evaluation with MNAR Rewards (`vpSFJoxyDz`) | 4-6 pts / 6-9 h | MIMIC-III is available, but the theorem, bridge estimator, and RL trajectory reconstruction make this a research implementation, not a throughput task. |
| Learning Cardiac Latent VCG (`hS6iw4PM8K`) | at most 4 pts initially | Official code is clean, but its pretraining route requires credentialed MIMIC-IV ECG and no released checkpoint is documented. |

## Excluded

- EviScreen: code exists, but previous attempts found real-data evaluation unavailable or incomplete; synthetic retrieval is only `toy`.
- Dynamic Decision Learning: a frozen LVLM still requires a meta-LLM API and 3B-72B vision-language models; its largest claims are not throughput-friendly.
- Broad foundation-model medical imaging and whole-slide papers: too many data, checkpoint, and GPU dependencies for the current score objective.

## Execution Order

1. Start ROCP; use cached predictions, run all 20 COVID seeds and BDD, and audit one theory claim.
2. Parallelize Cox and Treatment Allocation as theorem-plus-simulation work.
3. Finish PyHealth only as a CPU repair and run judge preflight before committing to its third claim.
4. Batch SwitchCraft only when a GPU is available.

For every candidate, record actual score, elapsed time, and every zero-point claim in `MODEL_QUALITY_BASELINE.md`. Do not publish an additional Space for a paper that already has one under `kondratevakate`; update the existing Space instead.
