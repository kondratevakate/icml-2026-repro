# MedMamba implementation-conformance reproduction

This CPU-only bundle audits six claims from *MedMamba: Multi-View State Space
Models with Adaptive Graph Learning for Medical Time Series Classification*
(arXiv `2605.24961v1`, OpenReview `qPqJH0heR0`).

## Result

Five central architecture claims are falsified in the released implementation:

- MCE returns `B x L x D`, not channel-wise `B x L x C x D`, and its
  convolutions are not depthwise;
- the first differential sample equals the input instead of zero;
- the spectral weight has shape `D`, so it cannot select frequency bins;
- the learned adjacency is static, input-independent, and disconnected from
  classification outputs and gradients;
- every Mamba call scans the time axis, while the paper's SGM scans channels
  and performs graph diffusion.

The five-dataset empirical claim is inconclusive because no fresh training was
performed. Prepared score: `10/12`.

## Run

Check out the author repository at commit
`418da50664338bc1d766394ee9c231496ab4de97` under `../official/code` and
extract the arXiv source under `../official/source`, then run:

```bash
python audit_claims.py \
  --official-code-root ../official/code \
  --paper-source-root ../official/source \
  --output evidence/claims_audit.json
python -m unittest discover -s tests -v
```

`audit_claims.py` injects an identity Mamba module only to run tensor-shape,
dependency, invariance, and gradient audits without CUDA. It does not use that
substitute to assess accuracy or the behavior of the Mamba kernel.

## Integrity

The bundle contains no author code, paper source, model weights, datasets, or
patient records. Exact source hashes and the pinned author commit are recorded
in `evidence/claims_audit.json`.

No leaderboard entries or peer reproductions were inspected.

