# Claim 3: frequency-specific filter


---
<!-- trackio-cell
{"type": "markdown", "id": "cell_70d946dc9d9c", "created_at": "2026-07-29T16:40:28+00:00", "title": "Claim 3: frequency-specific filter"}
-->
**Verdict - FALSIFIED IN RELEASED IMPLEMENTATION.**

For a length-16 probe there are `9` real FFT bins,
but each released real and imaginary filter has shape
`[12]`. The same feature-wise gain is broadcast across
every frequency, so the implementation cannot amplify one physiological band
while suppressing another for the same feature.


---
<!-- trackio-cell
{"type": "code", "id": "cell_6f69ffbba950", "created_at": "2026-07-29T16:40:28+00:00", "title": "C3 machine-readable evidence", "language": "python"}
-->
````output
{
  "fft_bins_for_probe": 9,
  "has_frequency_axis": false,
  "id": "C3",
  "imag_weight_shape": [
    12
  ],
  "paper_anchor_present": true,
  "real_weight_shape": [
    12
  ],
  "reason": "The learned complex gain has shape D and is broadcast identically across all FFT bins.",
  "score": 2,
  "verdict": "FALSIFIED_IN_RELEASED_IMPLEMENTATION"
}
````
