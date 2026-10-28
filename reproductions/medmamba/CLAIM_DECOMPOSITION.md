# MedMamba Claim Decomposition

Frozen forecast: `8/12` realistic, `10/12` stretch.

## C1: channel-wise multi-scale embedding

Paper claim: MCE maps `T x C` to channel-wise `T x C x D` features using
depthwise separable convolutions.

Audit: trace the released embedding shape and convolution groups. A full
verification requires retaining the channel axis and independent channel
filters. Expected: `2/2` if the released implementation contradicts both.

## C2: zero-padded differential view

Paper claim: the differential branch is
`[0; Z[2:T] - Z[1:T-1]]`, avoiding an artificial first-step spike.

Audit: execute the released `_compute_diff` on nonzero first samples and
compare it with the declared operator and a one-line corrected mutation.
Expected: `2/2`.

## C3: frequency-specific spectral modulation

Paper claim: a learnable filter can amplify or suppress specific physiological
frequency bands, with a frequency-indexed complex weight.

Audit: inspect parameter shape and demonstrate whether distinct FFT bins can
receive distinct learned gains for the same feature. Expected: `2/2`.

## C4: sample-conditioned graph affects predictions

Paper claim: each sample produces an adaptive adjacency and classification
gradients flow through graph diffusion to the graph projections.

Audit:

- compare adjacency for two different inputs;
- mutate graph parameters and compare the same block output;
- backpropagate classification output and inspect graph-parameter gradients;
- distinguish graph-regularizer gradients from classification gradients.

Expected: `2/2`.

## C5: graph diffusion and channel-wise spatial Mamba

Paper claim: SGM evaluates normalized graph diffusion and applies the spatial
SSM along the channel axis at each time step.

Audit: trace Mamba input shapes, test whether adjacency participates in the
returned tensor, and inspect the released graph-convolution path. Expected:
`2/2`.

## C6: five-dataset accuracy, ablations, and robustness

Paper claim: MedMamba leads on most subject-independent metrics across ADFTD,
APAVA, PTB, PTB-XL, and TDBRAIN; component and graph ablations support the
proposed mechanisms.

Full route: reconstruct all subject-level splits, run seeds 41-45, implement
all released and missing dataset configurations, and reproduce baselines and
ablations. Current expected score: `0/2`.

The released repository has one APAVA script, external dataset preparation,
no checkpoints, no cached metrics, and no complete five-dataset configuration.

