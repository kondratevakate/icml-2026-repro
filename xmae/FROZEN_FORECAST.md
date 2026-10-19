# xMAE frozen forecast

Frozen: 2026-07-29. OpenReview `NxbSbkLNMc`; arXiv `2605.00973v1`.

Written after primary paper/repository inventory and before independent model,
checkpoint, HDF5, notebook, or runtime checks. Leaderboards and third-party
verdicts are excluded.

| Claim | Expected |
|---|---:|
| C1. xMAE implements asymmetric ECG masking with directional PPG→ECG cross-attention and ECG reconstruction. | 2/2 possible from independent architecture/config/source inspection and dependency-light tensor-shape checks. |
| C2. The compact synthetic PVC artifact and supplied placeholder checkpoint form a runnable released evaluation example. | 2/2 possible if artifact schemas, checkpoint/model compatibility, and CPU notebook-equivalent inference pass. |
| C3. xMAE outperforms unimodal and multimodal baselines on 15 of 19 downstream tasks. | 0/2 expected: the paper and README state that evaluated datasets and real pretrained weights are not released. |
| C4. Pretraining uses about 9.4k hours / 3.4M synchronized ECG–PPG segments from about 2.4k MIMIC-III subjects. | 0/2 expected without the protected waveform cohort and frozen preprocessing manifest. |
| C5. Learned PPG representations encode ECG–PPG timing structure. | 0/2 expected without real pretrained weights, paired held-out signals, and analysis outputs. |
| C6. Representations generalize across devices, body locations, acquisition settings, and six studies. | 0/2 expected without the six evaluation datasets and canonical probes. |

Frozen expectation: **4/12**; plausible range **2–4/12**. The repository
explicitly labels its included `.h5` and `.pth` as made-up, so they may support
only a release smoke test, never the paper's empirical claims.
