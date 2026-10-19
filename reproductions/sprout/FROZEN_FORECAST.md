# Frozen forecast

Frozen before installing dependencies or executing released code.

- Expected score: **4 / 12**
- Plausible range: **2-6 / 12**

## Rationale

The official repository includes preprocessing, H&E reference-mask,
feature/prototype, partial-OT, prompt-sampling, SAM inference, NMS, and metric
code. Several image-only utilities should permit deterministic synthetic checks
without training or private data.

The release has no cached predictions, checkpoints, metric outputs, or tests.
Its advertised preprocessed-dataset hyperlink is empty, its SAM2 integration
must be supplied by replacing a missing directory with an external repository,
and its default `hf-hub:bioptimus/UNI2-h` backbone identifier does not resolve
on Hugging Face. Static inspection also finds that `OptimalTransport.solve()`
uses `self.numItermax` although the constructor never defines it, so the
canonical POT path is expected to fail immediately. These gaps make the main
benchmark and robustness tables unlikely to be independently recoverable
locally.

## Expected claim scores

| Claim | Forecast |
|---|---:|
| Fully training-free automatic prompting | 1 |
| Self-reference feature calibration | 1 |
| POT-Scan formulation and guarantee | 0 |
| Containment-aware refinement | 2 |
| Benchmark performance | 0 |
| Robustness and efficiency | 0 |
