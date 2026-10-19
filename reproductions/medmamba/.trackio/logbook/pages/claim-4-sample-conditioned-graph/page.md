# Claim 4: sample-conditioned graph


---
<!-- trackio-cell
{"type": "markdown", "id": "cell_e8f7b8f96564", "created_at": "2026-07-29T16:40:28+00:00", "title": "Claim 4: sample-conditioned graph"}
-->
**Verdict - FALSIFIED IN RELEASED IMPLEMENTATION.**

Two strongly different inputs produce identical adjacency
(`max |delta|=0.0`). Mutating graph
parameters changes adjacency by `0.502037`
but changes the returned representation by exactly
`0.0`.

Backpropagating a classification-output proxy gives these graph gradients:
`{"dynamic_gate.0.bias": null, "dynamic_gate.0.weight": null, "nodevec1": null, "nodevec2": null}`. Every value is null. Structure
penalties can optimize this static graph, but classification cannot.


---
<!-- trackio-cell
{"type": "code", "id": "cell_79f71af86258", "created_at": "2026-07-29T16:40:28+00:00", "title": "C4 machine-readable evidence", "language": "python"}
-->
````output
{
  "adjacency_input_change_max_abs": 0.0,
  "adjacency_parameter_mutation_max_abs": 0.5020374059677124,
  "classification_proxy_graph_gradients": {
    "dynamic_gate.0.bias": null,
    "dynamic_gate.0.weight": null,
    "nodevec1": null,
    "nodevec2": null
  },
  "id": "C4",
  "output_parameter_mutation_max_abs": 0.0,
  "paper_anchor_present": true,
  "reason": "The learner ignores x and adjacency is disconnected from the returned representation.",
  "score": 2,
  "verdict": "FALSIFIED_IN_RELEASED_IMPLEMENTATION"
}
````
