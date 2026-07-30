# Claim 3: Spectral Homogenization


---
<!-- trackio-cell
{"type": "markdown", "id": "cell_cb8d79356c92", "created_at": "2026-07-30T07:55:08+00:00", "title": "Claim 3: Spectral Homogenization"}
-->
**PARTIAL - 1/2.** Official SH is finite and shape preserving.
Paper and code use different random-spectrum constructions; zero-padded
smoothing in native FFT order breaks Hermitian symmetry before the code
discards the inverse FFT's imaginary component.


---
<!-- trackio-cell
{"type": "code", "id": "cell_a10440f4932f", "created_at": "2026-07-30T07:55:08+00:00", "title": "Claim 3: Spectral Homogenization evidence", "language": "python"}
-->
````output
{
  "shape_preserved": true,
  "dtype_preserved": true,
  "finite": true,
  "reported_info": {
    "sigma": 1.0,
    "P_white": 1024.0,
    "xi_std": 0.9974527359008789,
    "r_std": 0.0070710680447518826,
    "P_r_mean": 0.027167348191142082,
    "P_xi_mean": 1023.9727783203125
  },
  "complement_hermitian_asymmetry_ratio": 0.0005224485648795962,
  "effective_psd_relative_rmse_to_white": 0.09309865794565178,
  "effective_psd_peak_ratio": 1.3469285670355333,
  "effective_psd_valley_ratio": 0.7619166556251002,
  "trials": 128,
  "paper_vs_code_sampling": "The paper uses sqrt(Delta S) times random phase; code multiplies the full random FFT by sqrt(Delta S / HW) and then drops the imaginary inverse-FFT component."
}
````
