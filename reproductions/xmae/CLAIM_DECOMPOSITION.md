# xMAE claim decomposition

- **C1 architecture:** trace asymmetric masking, PPG-query/ECG-key-value
  directional cross-attention, reconstruction target, curriculum bounds, and
  configuration.
- **C2 compact example:** validate synthetic HDF5/metadata alignment, checkpoint
  state structure, model loading, tensor shapes, and deterministic CPU output.
- **C3 15/19 tasks:** requires fresh linear probes over the declared downstream
  datasets and matched baselines.
- **C4 pretraining cohort:** requires protected MIMIC waveform access and the
  frozen cohort/preprocessing manifest.
- **C5 timing analysis:** requires real pretrained representations and paired
  ECG–PPG timing labels.
- **C6 generalization:** requires all six studies and canonical participant
  splits.
