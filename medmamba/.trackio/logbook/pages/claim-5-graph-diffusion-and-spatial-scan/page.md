# Claim 5: graph diffusion and spatial scan


---
<!-- trackio-cell
{"type": "markdown", "id": "cell_465bc2017487", "created_at": "2026-07-29T16:40:28+00:00", "title": "Claim 5: graph diffusion and spatial scan"}
-->
**Verdict - FALSIFIED IN RELEASED IMPLEMENTATION.**

The paper specifies normalized adjacency diffusion and a spatial SSM over
channels at each time. All traced Mamba inputs are
`[[2, 16, 12], [2, 16, 12], [2, 16, 12], [2, 16, 12], [2, 16, 12]]`: sequence axis 16 is time, while the input has
4 channels. The block computes adjacency but never uses it in the returned
tensor; its nominal graph branch is a linear projection plus temporal Conv1d.


---
<!-- trackio-cell
{"type": "code", "id": "cell_a1e89c87871a", "created_at": "2026-07-29T16:40:28+00:00", "title": "C5 machine-readable evidence", "language": "python"}
-->
````output
{
  "all_mamba_sequence_axes_equal_time": true,
  "id": "C5",
  "input_channel_count": 4,
  "input_time_length": 16,
  "logits_shape": [
    2,
    3
  ],
  "mamba_input_shapes": [
    [
      2,
      16,
      12
    ],
    [
      2,
      16,
      12
    ],
    [
      2,
      16,
      12
    ],
    [
      2,
      16,
      12
    ],
    [
      2,
      16,
      12
    ]
  ],
  "paper_anchor_present": true,
  "reason": "All released Mamba calls scan L=time; adjacency is computed but not used in the output path.",
  "score": 2,
  "verdict": "FALSIFIED_IN_RELEASED_IMPLEMENTATION"
}
````
