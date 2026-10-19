# Independent checks

## Outcome

Prepared score: **4 / 10**, one point below the frozen 5/10 forecast and
within its 3-7 range.

| Claim | Score | Evidence |
|---|---:|---|
| Online semantic-modality discovery | 1/2 | The official state contains the coherent 1/5/1/7/2 partition, but it was not regenerated from the unavailable processed prompt data and uses CRP alpha 2.0 rather than the reported 5.0. |
| Replay-free structure-aware continual learning | 2/2 | Independent unit checks verify modality-isolated LoRA behavior; the EWC state contains Fisher/parameter aggregates and identifiers, not examples. |
| Headline continual-segmentation performance | 0/2 | No processed splits, run-level outputs, seeds, or metric tables are released for the 73.3% Dice and 4.1% forgetting result. |
| Robustness and ablation evidence | 0/2 | Paper tables and figures are present, but no order, perturbation, ablation, merge, or baseline outputs are released. |
| Efficiency and clinical breadth | 1/2 | Code and dataset documentation cover the architecture and 16 tasks, but the 8.6M comparison was not independently recomputed and no clinical predictions are released. |

## C1: modality discovery

Safe loading (`torch.load(..., weights_only=True)`) of the immutable official
`modality_state.pth` gives:

- 16 task identifiers;
- five modality identifiers;
- group sizes 1, 5, 1, 7, and 2;
- assignments matching cardiac ultrasound, polyps, dermoscopy, chest X-ray,
  and breast ultrasound under the released interleaved order.

The state records `alpha=2.0`. The paper's implementation section and the
current `scripts/main.py` both specify `alpha=5.0`. The checkpoint is therefore
evidence for the released five-way partition, but not a matching artifact for
the reported configuration.

The checkpoint also yields intra similarity 0.9133 (sample SD 0.0808, n=8) and
inter similarity 0.4200 (sample SD 0.1213, n=54). Those differ from the
appendix's approximate 0.94/0.05 and 0.51/0.10 values.

### Proposition check

Appendix Proposition 1 does not establish its claimed zero clustering error.
Two Gaussian distributions with fixed nonzero variance have overlapping
support, so better estimates of their parameters do not remove per-task Bayes
error. Using the checkpoint statistics and the threshold printed in the
paper's proof gives a positive sum of tail probabilities of approximately
0.0327 despite satisfying the stated separation condition. The proof's bound
is constant in `t`, so the assertion that it tends to zero exponentially does
not follow.

## C2: LoRA and EWC

An independent CPU test constructs the released `DynamicModalityLoRALinear`,
changes only modality 0, and verifies that modality 1 retains its independent
zero-initialized update and that switching back restores modality 0's output.
Welford updates were also checked against direct sample statistics.

The 69,470,834-byte official EWC state:

- hashes to the pinned Hub LFS digest;
- contains all five modality states and the same task mapping;
- contains Fisher tensors, anchor parameters, counts, and identifiers;
- contains no images, masks, prompts, or replay examples.

There is a narrower implementation caveat. Task adapters and segmentation
heads are shared within a modality by assigning the same module object to
multiple task keys. Their Fisher names nevertheless include the current task
id. On a subsequent task, earlier task-specific names no longer match
`get_trainable_param_names`, so those shared task components receive no
cross-task EWC penalty; modality LoRA and enhancer names do match and are
regularized. This does not negate the released replay-free LoRA/EWC mechanism,
but it narrows the scope of the paper's all-parameter description.

## C3-C5: empirical evidence and environment

The official repository has training and inference code but no JSON, CSV, NPY,
or NPZ run-level results. The Hub checkpoint release has weights and state only,
without `checkpoint_info.json`, processed dataset splits, predictions, or
baseline artifacts. Consequently the headline uncertainty, order robustness,
prompt perturbations, ablations, and Fisher-merge table cannot be audited
independently.

The published `requirements.txt` is a raw Conda environment export rather than
a portable pip lock. A pip dry-run fails at line 4 because
`anaconda-anon-usage` has no matching PyPI distribution; the file also includes
`conda`, `libmambapy`, Conda solver packages, and CUDA 13 packages.

## Commands

```powershell
cd medcrp_cl
python audit_medcrp_cl.py
python -m unittest discover -s tests -v
python -m compileall -q official/repo/scripts
python -m pip install --dry-run --no-deps -r official/repo/requirements.txt
python repro_medcrp_cl/build_logbook.py
python ..\validate_icml_logbook.py
```

The pip dry-run is expected to fail and is recorded as a release finding. All
nine deterministic audit tests pass.
