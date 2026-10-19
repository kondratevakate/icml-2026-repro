# SwitchCraft Claim Decomposition

Paper: *SwitchCraft: A Programmatic Framework for Designing State-Switching
Proteins* (`YtqaHnqv8c`, arXiv `2605.31236v1`).
Official code audited at commit
`67ffc0d155112624b557ea14dc55dbb9ac703933`.
Date: 2026-07-28.

## Decision

The high-throughput target is **4 strong points from C1/C2**, plus at most
**1 toy point from a bounded GPU primitive run**. The previous six-point
forecast is not defensible from the released bundle alone: the paper-scale C3
claim is much larger than the supplied task templates, while C4/C5 require an
unreleased or prohibitively large biosensor workflow.

| Claim | Full verdict route | Current limitation | Target |
|---|---|---|---:|
| C1: multistate sequence optimization and four-stage refinement | Audit Algorithm 1 and the exact 30/100/100/10 schedule, then complete one real 240-step CUDA optimization with a saved trajectory. | Static schedule audit is local; real Boltz gradients still require GPU. | 2 |
| C2: Motif, Binding, and Conformational Change losses | Independently evaluate Eqs. (1), (2), and (5), check gradients, and execute them inside a real design. | Core formulas run on CPU, but two literal code-paper divergences must be disclosed. | 2, subject to judge preflight |
| C3: six functional primitives | Reproduce all six task families with fresh structures and the paper's success criteria, including controls. | Full allostery alone is 24 motifs x 5 ligands x 2 directions x 100 designs. Several supplied YAMLs are generic templates, and no ligand-modification YAML is supplied. | 1 toy from bounded subset; do not claim full |
| C4: 44 cpGFP biosensor candidates | Recreate the 13,858-design screen, 89 prefiltered switches, cpGFP insertion, and final 44 passing designs. | No end-to-end cpGFP workflow or generated artifacts are released. | 0 |
| C5: SAM unquenching mechanism | Recreate the selected SAM switch, cpGFP insertion, chromophore-contact geometry, and compare with the natural nicotine sensor mechanism. | It is a single selected in-silico interpretation and depends on the missing C4 pipeline. | 0 |

## C1: Four-Stage Protocol

Paper anchors: Section 3.2, Algorithm 1, Appendix A.1.
Code anchors: `designer.py::MultistateDesigner.optimize`,
`designer.py::get_restype_from_logits`, and
`utils/mydesign_utils.py::Annealer`.

The official source has four stages totaling 240 updates:

| Stage | Steps | Learning rate | beta | gamma | Effective tau |
|---:|---:|---:|---:|---:|---:|
| 1 | 30 | 0.2 | 0 | 1 | 0.5 |
| 2 | 100 | 0.1 | 0 | 0 to 1 | 0.5 |
| 3 | 100 | 0.1 | 0 | 1 | 0.5 to 0.005 |
| 4 | 10 | 0.1 | 1 | 1 | 0.005 |

The code exposes temperatures `1.0` to `0.01`, but multiplies logits by two
before softmax. Its effective temperatures therefore match the paper's
`0.5` to `0.005`. It also implements the two inter-stage assignments
`z <- z_pseudo` and `z <- 2z`.

Full acceptance requires one real CUDA run with:

- all 240 optimization iterations completed;
- a finite nonzero loss trajectory and finite sequence gradients;
- five final Boltz diffusion samples for each state;
- the exact command, commit, GPU, package versions, runtime, and output hashes.

Local status: `audit_schedule.py` passed. It found all four stages, 240 total
updates, both inter-stage assignments, and exact effective paper
hyperparameters.

## C2: Loss Audit and Divergences

The CPU audit independently checks the released classes against the equations:

- `MotifLoss` and `AntiMotifLoss`;
- `LigandContactLoss`;
- `ConfChangeLoss`, including nonzero finite gradients.

Two differences must be disclosed:

1. `MotifLoss` only includes motif pairs with target distance below 22
   Angstrom, whereas literal Equation (1) sums all distinct motif pairs.
2. `AntiLigandContactLoss` defaults to `-1 * BindingLoss`, whereas Section 3.1
   states `-0.5 * BindingLoss`.

These do not automatically falsify the anchored C2 wording, which names the
three positive loss components, but they prevent an unqualified statement that
the released code implements every printed loss literally.

Local status: `audit_losses.py` passed all three positive-component checks with
zero numerical error on seeded CPU tensors. The conformational-change loss
produced a finite nonzero combined gradient norm of `0.1116180962`. The
22-Angstrom motif cutoff changed the audit example from the literal Equation
(1) value `206.3610` to the released-code value `73.2231`; the default
anti-binding ratio was `-1.0` rather than the printed `-0.5`.

## C3: Honest GPU Scope

The supplied task files are not all paper-exact:

- positive/negative allostery provide a concrete `1prw` plus OQO example;
- motif switching uses unresolved `motif1`/`motif2` placeholders;
- induced binding uses generic OQO/FAD ligands and length 100, while the paper
  uses a 50-aa design, a Top7 peptide fragment, and calcium;
- ligand discrimination uses OQO/FAD and length 100, while the paper uses
  OQO/calcium and length 50;
- no ligand-modification YAML is supplied.

Therefore the first GPU run is a feasibility smoke only. A defensible bounded
C3 toy reproduction should run both positive and negative allostery for a
paper-covered motif/ligand pair, generate five samples per state, apply the
Appendix A.2/B.1 criteria, and include a wrong-objective or no-AntiMotifLoss
control.

## Execution Order

1. Run the local C1 schedule and C2 numerical audits.
2. Build a pinned CUDA environment and download official Boltz/CCD assets.
3. Run one positive-allostery smoke design.
4. If the smoke is healthy, run positive, negative, and an ablation control
   with enough independent designs to avoid a selected-example claim.
5. Stop after C1/C2 plus bounded C3 evidence; do not enter the cpGFP screen.

## Updated Estimate

- C1/C2 local plus one GPU execution: 4 points in 2-4 hours.
- Bounded C3 toy extension: +1 point in 1-3 GPU hours.
- Realistic total: **4-5 points**, not six.
- C4/C5 are excluded from the throughput forecast.
