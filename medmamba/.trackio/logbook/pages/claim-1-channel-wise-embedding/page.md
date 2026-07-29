# Claim 1: channel-wise embedding


---
<!-- trackio-cell
{"type": "markdown", "id": "cell_a4424c697784", "created_at": "2026-07-29T16:40:28+00:00", "title": "Claim 1: channel-wise embedding"}
-->
**Verdict - FALSIFIED IN RELEASED IMPLEMENTATION.**

The paper declares channel-wise `T x C x D` embeddings and depthwise
convolutions. The pinned code returns
`[2, 16, 12]` (`B x L x D`) and all three convolution
groups are `[1, 1, 1]` for `4` input
channels. The channel-token axis no longer exists after MCE.


---
<!-- trackio-cell
{"type": "code", "id": "cell_f79ecbf09659", "created_at": "2026-07-29T16:40:28+00:00", "title": "C1 machine-readable evidence", "language": "python"}
-->
````output
{
  "conv_groups": [
    1,
    1,
    1
  ],
  "id": "C1",
  "input_channels": 4,
  "observed_embedding_shape": [
    2,
    16,
    12
  ],
  "observed_rank": 3,
  "paper_anchor_present": true,
  "paper_declared_rank": 4,
  "reason": "The channel axis is collapsed into BxLxD and convolutions use groups=1.",
  "score": 2,
  "verdict": "FALSIFIED_IN_RELEASED_IMPLEMENTATION"
}
````
