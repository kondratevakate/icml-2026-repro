# DC-PnPDP claim decomposition

Paper: **Plug-and-Play Diffusion Meets ADMM: Dual-Variable Coupling for Robust Medical Image Reconstruction**

OpenReview ID: `jEBkuuETjr`  
arXiv: `2602.23214v2`

Each claim is worth two points.

1. **Dual-coupled fixed-point optimality.** Under the paper's proximal-denoiser assumption, a fixed point of the deterministic DC-PnPDP backbone satisfies consensus and the first-order optimality condition of the original objective.
2. **Loose-PnP fixed-point bias.** Removing the dual state produces the extra non-zero consensus-gradient term characterized in the paper's second fixed-point theorem.
3. **Spectral Homogenization mechanics.** The released SH implementation matches the paper's frequency-domain construction and produces finite, shape-preserving outputs with the intended spectral effect.
4. **CT reconstruction gains.** The released artifacts independently support the reported LACT-90 and SVCT-20 fidelity gains under the stated evaluation protocol.
5. **MRI reconstruction gains and modality breadth.** The released artifacts independently support the reported fastMRI results and the claimed CT-to-MRI breadth.
6. **Component and efficiency evidence.** The released artifacts independently support the dual/SH ablation and the reported negligible per-step SH overhead.

