# Frozen forecast

Frozen before installing dependencies or executing released code.

- Expected score: **6 / 12**
- Plausible range: **4–8 / 12**

## Rationale

The official release contains the DC-PnPDP, DiffPIR, and Spectral Homogenization implementations, CT physics code, an external pretrained CT checkpoint, and exact train/validation case lists. This should make the two fixed-point claims and the local SH/dual mechanics amenable to independent checks.

The release does not contain cached reconstructions, metric tables, test fixtures, or either medical-image dataset. Its runnable entry point is CT-only and depends on CUDA `torch-radon`; the paper's MRI path is not present in the pinned repository. Canonical CT reproduction additionally requires a multi-gigabyte external diffusion checkpoint and AbdomenCT-1K volumes. Therefore the main CT, MRI, and runtime claims are unlikely to be independently reproduced during a local-hour audit.

## Expected claim scores

| Claim | Forecast |
|---|---:|
| Dual-coupled fixed-point optimality | 2 |
| Loose-PnP fixed-point bias | 2 |
| Spectral Homogenization mechanics | 2 |
| CT reconstruction gains | 0 |
| MRI breadth | 0 |
| Components and efficiency | 0 |

