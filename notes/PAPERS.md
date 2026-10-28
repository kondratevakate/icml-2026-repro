# ICML 2026 Reproduction: paper reference (Kate)

Translated from Russian on 2026-10-19; the Russian original is in the git history of this file. Paths follow
the 2026-10-19 layout (`reproductions/`, `notes/`).

The full reference list of papers selected for Kate's interests from the 6,341 ICML 2026 papers.
Link = OpenReview forum. Points = 2 × the number of checkable claims (2 per claim for a full reproduction OR a
full falsification, 1 for toy, 0 for none).

Fit legend: A = core (evaluation, reliability, segmentation, neurodegeneration), B = reconstruction,
robustness, 3D SSL, C = fMRI encoders, D = brain graphs, psychiatry, E = clinical, wearables, digital twin.
Access: OPEN = download immediately, REG = registration, CRED = credentialed PhysioNet (Kate has it),
REST = on application.
Code status: the repository was checked by hand where a URL is given; "walled" = OpenReview behind bot
protection, not confirmed.

---

# Status and priorities (updated 2026-07-19)

**Selection filter:** application to medical data (directly or indirectly) plus reproducibility plus a claim
that can be put to use. **The one criterion that cannot be skipped is data availability.** Code and text are
not criteria. **Papers are NOT exclusive**: repeat reproductions are welcome, "taken" does not rule a paper out.
**Mode:** first everything without GPU, then the selected ones in one batch on a GPU server. The full state
is in `notes/HANDOFF.md`.

**IMPORTANT about the search:** this list and `medical_neuro_papers.csv` were built from the area tag and
**miss medicine**. A scan of all 6,341 titles gives **368 medical papers, 113 of them outside the 255**
(Causality, LLM, CV, Time Series, Evaluation, Privacy). uqct, PyHealth and ProConMV were never among the 255.
Search `all_papers.csv`. **Data availability is not visible from the abstract**: read the PDF (ORBIT: "public"
in the introduction, a closed hospital dataset in the body).

## Logbooks (8 published, all canonical, validator green)

| Paper | orid | Result |
|---|---|---|
| ConfSleepNet | fve4MEzeSp | claim 1 analytically; **Eq. 8 as printed is probably a typo**. Claims 2-3 blocked |
| Solvable AE | wm3ABfhE7P | claim 2 convincingly; claim 1 k\*=2 cleanly, k\*=3 marginally (as for the authors) |
| CalPro | 3LRWjJTp0Y | **CORRECTED 2026-07-19:** the published verdict for claim 1 rested on one seed (0.984 coverage was luck). On 5-8 seeds (`reproductions/calpro/verify_claim1_degradation.py`) coverage degradation is 17.5 pp on average and 38.8 pp at worst: **the claim's threshold (5 pp) is not met**; the direction of the effect is right, but the stated size does not reproduce. The published logbook has not been updated yet |
| uqct | a6vCNSBBeq | SLM coverage guarantee on the classical path (1.000 at δ=0.1 and 0.05); "deep methods are tighter" needs GPU |
| Entropy/mislabeled | BUxrIaf7Zc | **toy on real CIFAR-100N**: claim 1 in direction, claim 2 SEI proxy AUC 0.71 |
| Conditional Coverage Diagnostics | vaApZm6MKM | claims 1-2 reproduced: with unspecified groups CovGap goes blind (0.005 against a truth of 0.095), ERT recovers 92%; the over/under decomposition matched the theory |
| When Can We Trust Survival Model Evaluation | Y9gsOEdaNE | **claims 1-2 reproduced**: metric bias depends on the mechanism (IBS: administrative −0.080, covariate-dependent ~0), and with close models the ranking matches the oracle only 29-42% of the time at 80% censoring |
| SurvFD / SurvSHAP-IQ | SldP4LGjdz | **claims 1-2**: Theorem 3.2 reproduced exactly (3/3 scenarios, common random numbers); the n-Shapley interaction is 1.96× larger under a true interaction (honest caveat about the exp link on the S scale) |

## Queue without GPU (do now)

- **LERD** `B5DAV1EA8Z`: neurodegeneration / Alzheimer EEG. Claim 1 includes **synthetic benchmarks**, so part of
  it needs no downloads. Real data: OpenNeuro **ds004504** (open).
- **DC-PnPDP** `jEBkuuETjr` claim 1: a convergence theorem; audit plus a small synthetic test.
- **ProConMV** `4F4Ziv1d9G` Corollary 1: already checked (`reproductions/proconmv/verify_corollary1.py`); the full
  claims are blocked (no code and no concept annotations).

## Queue for the GPU server (collect in one batch)

- **EviScreen** `FCVPVqRLDJ`: the most ready medical one: an eval-only path plus released weights plus open
  fundus data.
- **uqct** "deep methods give narrower regions": U-Net, ensembles, diffusion plus LIDC-IDRI.
- **Entropy claim 3**: full SEI plus three medical datasets (ISIC, DeepDRiD).
- **MedCRP-CL**, **DC-PnPDP** claim 2, **TACO**, **ECG-R1**, **SigmaPPG**, **xMAE**, **CortexMAE**, **MoQiswth2n**.


## Local data (for toy checks without GPU)

- **CIFAR-100N** (`data/cifar100n/`, 340 MB): CIFAR-100 plus human noisy labels. Used for Entropy.
- **MFIDDR sample** (`data/mfiddr_sample/`, 43 MB): 96 fundus images, 4 fields per patient, a git-clonable
  sample. No direct free use found: the fundus papers in the queue (EviScreen, ProConMV) need GPU or heavy
  models, or are blocked (ProConMV: code and annotations not released). Kept in reserve for a future fundus
  target.
- **MIMIC-III Clinical Database Demo** (`data/mimic3demo/`, 38 MB, 26 tables): **open without authorisation**
  (physionet.org/content/mimiciii-demo/1.4/, 100 patients). It unblocked **PyHealth 2.0** (`gMLVFN9hl8`), which
  had been marked "needs credentialed MIMIC" and is in fact compatible with the demo format. **Working
  environment: WSL Ubuntu-24.04 (Python 3.12.3, PyHealth's version, no hacks), not Windows.** Installed with
  `uv` (no sudo): `uv venv --python 3.12` plus
  `uv pip install 'git+https://github.com/sunlabuiuc/PyHealth.git@master'`.
  **The toy run is confirmed end to end in 6.8 s:** loading MIMIC-III demo (100 patients, 13,030 events) →
  task `MortalityPredictionMIMIC3` → `Transformer` → 3 training epochs → interpretability. On Windows the same
  pipeline hung twice; the cause turned out not to be the platform but a missing
  `if __name__ == "__main__":` guard in the script (dask spawned workers, each re-imported the module and
  rebuilt the cache, an endless recursion of processes). Claim 2 (v2 39× faster and 20× less memory than v1)
  was not attempted: it needs a controlled benchmark.

## Blocked (triggers for revisiting)

- **TCSeg** `bGxMpQsIuN`: the repository is README-only (0 .py files). Trigger: code release.
- **ProConMV** `4F4Ziv1d9G`: the repository is empty; the six lesion concepts (the authors' contribution) are
  not released. Data: the MFIDDR sample can be cloned; the full set plus DRTiD by questionnaire.
- **ORBIT** `pd6GY2x8FG`: longitudinal fundus/OCT, Shanghai General Hospital, IRB, not released. **Out because of
  the data.**
- **SzCORE**, **S2M-Net**: closed held-out set / withdrawn.

---

## Take now (data available, code exists)

| # | Paper | Fit | Points | Data (access) | Code |
|---|---|---|---|---|---|
| 1 | [PyHealth 2.0](https://openreview.net/forum?id=gMLVFN9hl8) · arXiv 2601.16414 | E | 10 | MIMIC-III/IV, eICU, SleepEDF, SHHS (CRED, Kate has it) | [sunlabuiuc/PyHealth](https://github.com/sunlabuiuc/PyHealth) · LIGHT |
| 2 | [TCSeg (Are We Overconfident?)](https://openreview.net/forum?id=bGxMpQsIuN) · arXiv 2605.25561 | A | 12 | Pancreas-CT, LA, BraTS2019 (TCIA, OPEN) | [DirkLiii/TCSeg](https://github.com/DirkLiii/TCSeg): **README-only, no code (GitHub API 2026-07-18: 0 .py). BLOCKED until code release** |
| 3 | [Principled Confidence CT](https://openreview.net/forum?id=a6vCNSBBeq) · arXiv 2602.05812 | A | 12 | LIDC-IDRI (TCIA, OPEN); 3 claims on closed industrial CT, so the real ceiling is about 6 | [SwissDataScienceCenter/uqct](https://github.com/SwissDataScienceCenter/uqct) · MEDIUM |
| 4 | [MedCRP-CL](https://openreview.net/forum?id=v0DWbfP3b9) · arXiv 2605.20297 | A | 10 | 16 segmentation tasks (endoscopy, dermatology, ultrasound, CXR; OPEN, to be collected) | [zygao930/MedCRP-CL](https://github.com/zygao930/MedCRP-CL) · MEDIUM |
| 5 | [DC-PnPDP](https://openreview.net/forum?id=jEBkuuETjr) · arXiv 2602.23214 | B | 12 | AbdomenCT-1K, fastMRI (OPEN) | [duchenhe/DC-PnPDP](https://github.com/duchenhe/DC-PnPDP) · MEDIUM |
| 6 | [EviScreen](https://openreview.net/forum?id=FCVPVqRLDJ) · arXiv 2605.15171 | A/E | 12 | JSIEC, RIADD, CheXpert, Derm12345 (OPEN; CheXpert is large, take the small version) | [DopamineLcy/EviScreen](https://github.com/DopamineLcy/EviScreen) · MEDIUM |
| 7 | [ConfSleepNet](https://openreview.net/forum?id=fve4MEzeSp) · arXiv 2605.17021 | E-wear | 10 | SleepEDF-20/78 (OPEN), MASS (REST), SHHS (REG) | [By4te/ConfSleepNet_ICML2026](https://github.com/By4te/ConfSleepNet_ICML2026): **README only, no code** |

**ConfSleepNet: reproduction status.** Claim 5 (Propositions 1 and 2) reproduced independently (CPU, script
`reproductions/confsleepnet/verify_prop.py`): the propositions hold under the normalisation Σb+u=1; Eq. 8 as printed
probably has a typo (it fails Prop 2 as C→1). Logbook published:
https://huggingface.co/spaces/kondratevakate/repro-a-conflict-aware-evidential-framework-for-reliable-sleep-stage-classification.

## Available but heavy (code and data exist, but foundation-model scale)

| # | Paper | Fit | Points | Data (access) | Code |
|---|---|---|---|---|---|
| 8 | [CortexMAE (ViT fMRI)](https://openreview.net/forum?id=s8cdRWLTCc) · arXiv 2510.13768 | C | 10 | HCP, NSD (REG, Kate has them); skip the UK Biobank part | [MedARC-AI/fmri-fm](https://github.com/MedARC-AI/fmri-fm) · HEAVY |
| 9 | [Foundation VAE CT](https://openreview.net/forum?id=f51haqkwRQ) | B | 4 | LiTS, KiTS19, MSD, CT-RATE (OPEN) | [qic999/Foundation-VAE](https://github.com/qic999/Foundation-VAE) · HEAVY |
| 10 | [ECG-R1](https://openreview.net/forum?id=dcLjhvVU47) · arXiv 2602.04279 | E-wear | 12 | PTB-XL, Code-15, CPSC2018, Chapman-Shaoxing, Georgia (OPEN), MIMIC-IV-ECG (CRED) | [PKUDigitalHealth/ECG-R1](https://github.com/PKUDigitalHealth/ECG-R1) · HEAVY |
| 11 | [SIGMA-PPG](https://openreview.net/forum?id=PKgGsbORt9) · arXiv 2601.21031 | E-wear | 12 | MIMIC-III waveform, BIDMC (CRED, Kate has them) | [ZonghengGuo/SigmaPPG](https://github.com/ZonghengGuo/SigmaPPG) · HEAVY |
| 12 | [xMAE](https://openreview.net/forum?id=NxbSbkLNMc) · arXiv 2605.00973 | E-wear | 12 | MIMIC-III waveform matched (CRED, Kate has it) | [hzhou3/xMAE](https://github.com/hzhou3/xMAE): weights not released · MED/HEAVY |
| 13 | [TACO (Beyond Instance-Level SSL 3D)](https://openreview.net/forum?id=Vtps0ZUlgv) · arXiv 2605.14654 | B | 12 | UPENN-GBM (TCIA), BraTS, ISLES22, ADNI, ADHD-200, IXI (OPEN) | [Ashespt/TACO](https://github.com/Ashespt/TACO): "upon publication" · HEAVY |
| 14 | [Omni-fMRI](https://openreview.net/forum?id=Rc8th2rXXe) · arXiv 2601.23090 | C | 12 | 9 datasets, not named | [OneMore1/Omni-fMRI](https://github.com/OneMore1/Omni-fMRI) (announced) · HEAVY |

## Watch list (important, no code yet, revisit in about 6 months)

| # | Paper | Fit | Points | Data | Trigger |
|---|---|---|---|---|---|
| 15 | [PhenoBrain](https://openreview.net/forum?id=9NqKL9QQ4a) **Spotlight** | D | 4 | ADNI/ABIDE (probably) | code release |
| 16 | [Deep Learning for BioImaging: What Are We Learning?](https://openreview.net/forum?id=ZgJlDylia4) | A | 6 | RxRx3, JUMP-CP, HEST (OPEN) | code release |
| 17 | [T-measure (segmentation metric)](https://openreview.net/forum?id=tRpJyIzxDj) | A | 6 | walled | release or abstract |
| 18 | [Real-World Unsupervised Predict Brain Responses OOD](https://openreview.net/forum?id=H1HHss5Zj4) | C | 4 | probably NSD | code release |
| 19 | [LERD (neurodegeneration)](https://openreview.net/forum?id=B5DAV1EA8Z) · arXiv 2602.18195 | A | 12 | OpenNeuro ds004504 (OPEN) | data open, can be reimplemented |
| 20 | [Geometry-Guided Brain Graphs](https://openreview.net/forum?id=ieUVj77Hsz) · arXiv 2511.04539 | D | 6 | HCP rest + WM fMRI, 1,067 subjects (REG, Kate has it) | data available, reimplement |
| 21 | [Med-SegLens](https://openreview.net/forum?id=vaRFU0xKQa) · arXiv 2602.10508 | A | 12 | BraTS (OPEN) | code release |
| 22 | [Disease-Centric VLP 3D CT](https://openreview.net/forum?id=LcSPuepCDU) · arXiv 2606.25546 | B | 12 | CT-RATE, Rad-ChestCT (OPEN) | code release |
| 23 | [Brain Networks Should Be Learned, Not Constructed](https://openreview.net/forum?id=VRJKiV0qGM) | D | 4 | ABIDE (probably) | code announced at yangliang.github.io, link did not open |
| 24 | [On the Spectral Unreachability of Brain Graph Learning](https://openreview.net/forum?id=n3ilFhSRIx) | D | 4 | brain-graph benchmarks | code announced at the same place, did not open |
| 25 | [Multimodal Scaling Laws Visual Cortex](https://openreview.net/forum?id=OQ6jQHJPTT) | C | 6 | NSD/Brain-Score (REG) | the previous version's repository is at EPFL NeuroAI; find the one for this version |
| 26 | [NeurIPS: sphere-based brain decoding](https://openreview.net/forum?id=gXTtBRI1yj) · arXiv 2605.24993 | C | 10 | Natural Scenes Dataset (REG) | code release |
| 27 | [FACT (multi-label diagnosis)](https://openreview.net/forum?id=T5u3haVBgi) | A | 6 | walled (probably CXR) | abstract or code |
| 28 | [Reference-Free Meta-Learning MRI recon](https://openreview.net/forum?id=jPVGiAUlNa) | B | 6 | walled (fastMRI?) | abstract or code |
| 29 | [SI-IGCL (psychiatry)](https://openreview.net/forum?id=ZoYQxFtDVM) | D | 4 | ABIDE/ADHD-200 (probably) | code release |
| 30 | [Structured Multi-modal Graph (psychiatry)](https://openreview.net/forum?id=cX3r7kAqr1) | D | 4 | REST-meta-MDD/ABIDE (probably) | code release |
| 31 | [MuHL (hypergraph)](https://openreview.net/forum?id=j5vTJrpoeL) · arXiv 2606.03310 | D | 12 | ABIDE/ADNI (probably) | code release |
| 32 | [ADHD Detection](https://openreview.net/forum?id=IaZmw8c9U4) | D | 4 | ADHD-200 (OPEN) | code release |
| 33 | [Alignment between Brains and AI](https://openreview.net/forum?id=XrRqY9QOSa) · arXiv 2507.01966 | C | 12 | Brain-Score style | code release |
| 34 | [Local Intrinsic Dimension](https://openreview.net/forum?id=B3Z3rWo4t6) · arXiv 2601.22722 | C | 10 | not named | abstract or code |
| 35 | [Mind the State (EEG-to-fMRI)](https://openreview.net/forum?id=Kw1Z8gOTvk) | C | 6 | not named | abstract or code |
| 36 | [Uncovering Latent Communication (AFR-Net)](https://openreview.net/forum?id=clFn9vQ8cK) · arXiv 2602.00561 | D | 10 | ABCD (REST), PPMI (REG); NOT HCP | anonymous repository only |
| 37 | [OSF (sleep FM)](https://openreview.net/forum?id=rlVGCyjqnt) · arXiv 2603.00190 | E-wear | 10 | NSRR: SHHS/MESA/MrOS (REG) plus SleepBench | code release |
| 38 | [Solvable AE: Test Loss Misaligns](https://openreview.net/forum?id=wm3ABfhE7P) · arXiv 2602.10680 | conceptual | 10 | synthetic (OPEN) | LIGHT, reimplement in an evening (theory, conceptually close to Dice vs ICC) |
| 39 | [TwinWeaver (digital twin)](https://openreview.net/forum?id=oWqRhYpJb6) · arXiv 2601.20906 | E | 12 | GENIE (REST) | data and code |
| 40 | [Marrying SDOH (digital twin)](https://openreview.net/forum?id=9GKURFWGeh) · arXiv 2605.09771 | E | 12 | UK Biobank (REST) | data and code |

## Avoid (broken for reproduction)

| # | Paper | Reason |
|---|---|---|
| 41 | [S2M-Net](https://openreview.net/forum?id=eh48NIgu9z) · arXiv 2601.01285 | withdrawn from arXiv by the authors |
| 42 | [SzCORE](https://openreview.net/forum?id=3vhn9kVVY5) · arXiv 2505.18191 · [esl-epfl/szcore](https://github.com/esl-epfl/szcore) | the key number is on a closed held-out test set |
| 43 | [SleepMaMi](https://openreview.net/forum?id=e9hvlHlUas) · arXiv 2602.07628 / [SleepLM](https://openreview.net/forum?id=9wpwfSJCp9) · arXiv 2602.23605 | proprietary PSG data, no code |
| 44 | [Cardio-mmFlow](https://openreview.net/forum?id=e43JwTIb2V) | no open data, no code |

---

# More: adjacent clusters (found, but outside the core 44)

## EEG foundation models / brain decoding (adjacent, NOT Kate's core)
A large cluster; EEG is not Kate's niche. Listed in case the topic is needed:
- [PATCHCODE](https://openreview.net/forum?id=NWcZ5vualM) · [Let EEG Models Learn EEG](https://openreview.net/forum?id=TP8OuKKmsf) (arXiv 2605.21280) · [EEG-FM-Bench](https://openreview.net/forum?id=vGeNaFHdET) (2508.17742) · [EmBrace](https://openreview.net/forum?id=BJ5rYj8O8W) · [MEG-XL](https://openreview.net/forum?id=cefix4VmhS) (2602.02494)
- [ViEEG](https://openreview.net/forum?id=DkK7GUr8n3) (2505.12408) · [PCRNet](https://openreview.net/forum?id=89MeH5Ax8r) · [EEG Hyperbolic MoC](https://openreview.net/forum?id=VSn4wLFd2p) (2604.12579) · [ROAMM](https://openreview.net/forum?id=zqLPdt09fE) · [See the Emotion](https://openreview.net/forum?id=AdTBZw18aH) · [Harnessing Spectrum Video](https://openreview.net/forum?id=iCjSoADDPs)
- [BIT-LLM (fMRI-to-Text)](https://openreview.net/forum?id=HZhlRnJOHq) · [KAST-BAR](https://openreview.net/forum?id=Ee4j4zMir5) (2605.13133) · [BrainJanus](https://openreview.net/forum?id=nJxailqsUW) · [Mind-Omni](https://openreview.net/forum?id=3gCdh3u2GK) (2605.29591) · [NeuroCLUS](https://openreview.net/forum?id=pFweJM4Uw8) · [Physiology as Language (respiration → sleep EEG)](https://openreview.net/forum?id=ceErPsyO38) (2602.00526)

## MIMIC / EHR trajectories (close to the process-mining angle; datasets probably MIMIC)
- [PathwayLLM (sepsis, clinical trajectory)](https://openreview.net/forum?id=oB7gZX7MKP): closest to process/pathway
- [Time-Conditioned Foreseeing (EHR foundation, irregular events)](https://openreview.net/forum?id=IalpB5Mzaz)
- [STT-LLM (longitudinal clinical profiles)](https://openreview.net/forum?id=oLtn3L43Ej)
- [HEARTS (LLM reasoning on health time series)](https://openreview.net/forum?id=qj4EnIvNU4) (2603.06638)

## Biosignals / wearables (clinical PPG, ECG, sleep; not consumer)
- [SGERA (ECG report)](https://openreview.net/forum?id=iEtOxzAs51) · [A robust PPG foundation model](https://openreview.net/forum?id=JF4045xnhl) (2606.07365) · [Learning Cardiac Latent (VCG)](https://openreview.net/forum?id=hS6iw4PM8K) (2605.31249) · [Cross-Modal Biosignal VAE](https://openreview.net/forum?id=aComqAqP6j)
- (see also the main list: ECG-R1 #10, SIGMA-PPG #11, xMAE #12, OSF #37, ConfSleepNet #7)

## Digital twin (medical): thin
- TwinWeaver #39, Marrying SDOH #40 (both in the main list, data closed). No brain or imaging digital twin at
  ICML 2026.

---

# Empty niches (checked; practically absent at ICML 2026)
Not "unimportant" but "the wrong venue": the area is under-represented, so it is potential for Kate's own
paper rather than a reproduction:
- **Brain age / face age:** 0 papers (checked for brain age, biological age, chronological age, aging, faceage).
- **Process mining / event logs / MIMICEL:** 0 (that is ICPM/BPM/CHIL/ML4H, not ICML).
- **Markov models in medicine (HMM, semi-Markov):** 0 by title.
- **Brain or imaging digital twin:** 0.

---

# Verified repositories (real code visible, GitHub API 2026-07-18)
SwissDataScienceCenter/uqct (54 .py, but data and weights "on request", Zenodo later) · zygao930/MedCRP-CL (7) ·
duchenhe/DC-PnPDP (22) · DopamineLcy/EviScreen (25, has a reproduce_directly eval-only path plus open fundus
data) · qic999/Foundation-VAE (402) · Ashespt/TACO (20) · sunlabuiuc/PyHealth (542) · OneMore1/Omni-fMRI (23) ·
MedARC-AI/fmri-fm (CortexMAE, 17) · ZonghengGuo/SigmaPPG (23) · PKUDigitalHealth/ECG-R1 (641) · hzhou3/xMAE (10) ·
esl-epfl/szcore. **README only (no code):** By4te/ConfSleepNet_ICML2026, **DirkLiii/TCSeg** (the "verified" mark
was a mistake).

# Datasets Kate has or will get
TCIA (Pancreas-CT, LIDC-IDRI, UPENN-GBM), OpenNeuro (ds004504), MIMIC-III/IV (credentialed), HCP (registration).
Plus open ones: BraTS, fastMRI, AbdomenCT-1K, CT-RATE, PTB-XL, SleepEDF, ADHD-200, IXI, NSD.

# How this was obtained (for re-checking)
Paper data: `fetchICML2026Papers()` from `icml2026-data.js` on `icml-2026-agent-repro-challenge.static.hf.space`.
Claims: `claims.json` plus `claims_anchored.json`, keyed by orid. Logbooks and agents: the HF Spaces API with
`filter=icml2026-repro`, tags `paper-<orid>`. For code and data of walled papers: OpenReview is behind bot
protection, marked "probably / not confirmed".
