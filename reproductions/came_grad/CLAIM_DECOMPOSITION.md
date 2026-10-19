# CAME-Grad claim decomposition

## C1 — three-stage optimizer mechanics

CAME-Grad combines:

1. conflict-averse direction rectification inside a trust region centered at
   the unweighted mean task gradient;
2. rescaling of the rectified direction toward
   `kappa * ||g_joint||`;
3. interpolation between the enhanced direction and
   `kappa * g_joint`.

Independent target: implement the printed equations without author code and
test mathematical invariants and degenerate cases.

## C2 — SDE mechanism and optimization claims

The paper describes linear scalarization as causing drift deviation and
diffusion decay, while CAME-Grad is said to restore escape energy, favor
flatter minima, ensure geometric validity, and guarantee global convergence
stability.

Independent target: distinguish exact algebraic consequences from causal or
convergence assertions requiring assumptions or empirical evidence.

## C3 — eight-backbone clinical-efficacy gains

The reported headline is an average CE improvement of 2.3% on MIMIC-CXR and
1.9% on IU X-Ray across WCL, XProNet, DCL, PromptMRG, CAMANet, DDaTR, TGRG,
and REVTAF.

Independent target: fresh paired baseline/CAME-Grad runs under the paper's
standardized CheXbert protocol. Recomputing an average from published table
cells is only a transcription consistency check.

## C4 — multi-task optimizer comparison

The paper compares UW, GradNorm, CAGrad, RotoGrad, FAMO, MMPareto, STGU, and
CAME-Grad on a common RRG backbone.

Independent target: rerun all optimizers with fixed data, model selection,
seeds, and evaluation.

## C5 — stage ablations

The paper removes each of S1, S2, and S3 to attribute gains to the three-stage
design.

Independent target: execute the exact ablation protocol on the declared
backbones and datasets. A toy invariant test does not validate clinical
ablation effects.
