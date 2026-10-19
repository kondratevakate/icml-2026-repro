# 3DMedAgent claim decomposition

- **C1 architecture:** OAMI creates organ-aware memory from VISTA3D outputs;
  CFLT localizes lesions/slices with CT-CLIP; T1S iteratively requests one
  informative slice, capped at five turns.
- **C2 benchmark:** independently validate the released DeepChestVQA CSV:
  record, scan, category and subtype counts; IDs/options/answers; duplicates
  and missing fields.
- **C3 headline gain:** fresh 40+ task evaluation, not table transcription.
- **C4 model comparison:** fresh predictions for the declared MLLMs.
- **C5 ablation:** canonical staged runs with identical cases/settings.
- **C6 expert/turn analysis:** released radiologist labels and matched T1S
  traces are required.
