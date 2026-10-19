# Independent checks

## Result

Prepared score: **2/12**, matching the lower bound of the frozen 2-4 range.

| Claim | Score | Independent result |
| --- | ---: | --- |
| C1 asymmetric masked cross-modal reconstruction | 1/2 | Core PPG-to-ECG information flow and ECG reconstruction are implemented, but the canonical config disables curriculum and no code path promotes the loss-driven ladder. |
| C2 released synthetic/checkpoint example | 1/2 | HDF5/metadata align and the checkpoint strict-loads, but the checkpoint yields all-NaN embeddings in evaluation mode; the PVC notebook instead uses a fresh random encoder. |
| C3 15/19 downstream tasks | 0/2 | Inconclusive: real pretrained weights and evaluated datasets are explicitly unreleased. |
| C4 9.4k-hour cohort | 0/2 | Inconclusive: the protected waveform cohort and frozen manifest are absent. |
| C5 timing structure | 0/2 | Inconclusive: real weights and paired held-out analysis data are absent. |
| C6 six-study generalization | 0/2 | Inconclusive: the six evaluation datasets and canonical split/probe outputs are absent. |

## C1 findings

- `Bridger.forward` updates ECG query tokens by attending to PPG keys and values.
- The decoder reconstructs ECG and the configured loss is masked ECG MSE.
- The paper specifies an 80% to 90% loss-driven curriculum in 5% increments.
- `cfg/xmae.yaml` fixes ECG masking at 0.8 and sets `use_cl: false`.
- Trainer state contains `_ladder_idx`, but no validation hook or other path ever
  increments it.
- A reverse `ppg2ecg` bridge with 790,272 parameters is instantiated but never
  called. This is reported as dead release code rather than interpreted as part
  of the claimed directional path.

## C2 findings

- Synthetic HDF5 shape is `[3000, 1000]`; metadata, labels, 60 patient IDs, and
  all 150 label-noise indices align exactly.
- The 417-key placeholder checkpoint loads with `strict=True`.
- 58 of 60 BatchNorm running-variance tensors contain negative values (873
  values total). The checkpoint has no NaN/Inf values itself, but square-rooting
  those variances during `eval()` makes all 512 values in a two-row embedding
  output NaN.
- Training-mode embedding and masked reconstruction forward passes are finite,
  which explains why the simple notebook's cached smoke output succeeds: it
  never calls `eval()`.
- That notebook also assigns `ecg = ppg_signal2` rather than the separately
  generated `ecg_signal2`.
- The PVC notebook calls `build_and_load_xmae()` with no checkpoint path. An
  independent seeded rerun of its random-encoder path completed with finite
  2400x256/600x256 features and validation AUC 0.8753. This only demonstrates
  the easy synthetic probe pipeline; it is not evidence about pretrained xMAE.

Machine-readable evidence is in
`repro_xmae/evidence/artifact_audit.json` and
`repro_xmae/evidence/pvc_random_encoder_probe.json`.

No leaderboard or third-party verdict was used.
