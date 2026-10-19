# Independent checks

## Prepared score: 5 / 12

| Claim | Score | Finding |
|---|---:|---|
| C1 Dual-coupled fixed-point optimality | 2/2 | An independent scalar quadratic satisfies consensus, both ADMM fixed-point equations, and stationarity of the objective with effective weight `lambda = rho * gamma` to below `4e-16`. |
| C2 Loose-PnP fixed-point bias | 1/2 | The released algebraic fixed-point equation is reproduced, and its residual is below `3e-16`. For an exact proximal denoiser it is stationarity of a Moreau-envelope objective, not the original objective. Main Theorem D.1, Appendix Theorem E.3, and its proof also use incompatible scalings. |
| C3 Spectral Homogenization mechanics | 1/2 | Official SH is finite and shape/dtype preserving. The paper samples fixed complementary amplitude with random phase, while code scales the full Gaussian FFT. Zero-padded smoothing in native FFT order breaks Hermitian symmetry before the code discards the inverse FFT's imaginary component. |
| C4 CT reconstruction gains | 0/2 | No data, cached reconstructions, or metric outputs are released. The example runs DiffPIR, not DC-PnPDP, and uses `w_tik=0`; the paper reports `1e-5 / sigma_t^2` for LACT. |
| C5 MRI breadth | 0/2 | No MRI implementation or configuration exists in the pinned repository. |
| C6 Components and efficiency | 1/2 | Both dual and SH components are present and all 22 Python files compile. The ablation and A100 timing results are not cached or independently regenerable locally. |

## Fixed-point checks

For

`f(x) = a/2 * (x-y)^2`, `phi(x) = q/2 * x^2`

with `a=2.5`, `y=1.7`, `q=0.8`, `rho=1.3`, and `gamma=0.4`, the independent
closed-form dual fixed point gives:

- consensus residual: `0`;
- data-update residual: `3.33e-16`;
- proximal-update residual: `5.55e-17`;
- original-objective residual with `lambda=rho*gamma`: `2.22e-16`.

The loose fixed point exactly satisfies
`grad f(x) + rho * (x - prox(x)) = 0`, but its residual for the original
`f + rho*gamma*phi` objective is `0.15225`. Its Moreau-envelope stationarity
residual is `2.78e-16`.

## Released CT penalty mismatch

The paper requires `rho > 0` and reports the LACT choice
`rho = 1e-5 / sigma_t^2`. The release's README and example shell command use
`--w-tik 0`. In the released linear system this makes both the matrix and
right-hand consensus terms vanish. An independent identity-operator check
shows two different `(z-u)` anchors produce exactly the same solved `x` at
zero penalty, but differ by `0.5` at penalty `0.2`.

## Spectral Homogenization

The official CPU path runs without extra imaging dependencies. Across 128
independent samples of a structured low-amplitude residual:

- effective PSD relative RMSE from the white target: `0.0931`;
- maximum/target PSD ratio: `1.3469`;
- minimum/target PSD ratio: `0.7619`;
- complementary-spectrum Hermitian-asymmetry ratio: `5.22e-4`.

These measurements are a mechanism audit, not a canonical image-quality
reproduction.

## Release inventory

- official commit: `f508749a00fe264b33da15421b7fa7577a0534b2`;
- tracked files: 30;
- Python files: 22, all compile;
- no tests, requirements lock, cached outputs, or MRI path;
- external CT checkpoint is linked through Google Drive;
- AbdomenCT-1K and fastMRI data are not included.

## Visual PDF check

Pages 7, 8, 16, 17, 22, 23, and 24 of arXiv v2 were rendered and visually
checked. The tables, theorem statements, equations, and appendix scalings are
legible and agree with the TeX evidence used by this audit.

