"""verify_claim6.py — Section 6.2 (HH-RLHF / Alpaca-7B Pareto front).

This script does NOT attempt a toy substitute. It records the concrete, checked reasons why the
claim cannot be reproduced within this task's CPU/time budget, and emits results/claim6.json with
verdict "inconclusive".

Required to reproduce Section 6.2 as stated in the paper:
  1. a reproduced Alpaca-7B policy (dai2024safe variant) trained with (non-linear) GRPO via TRL,
     with M=8 sampled responses per prompt, on hh-rlhf prompts;
  2. TWO reward models: Qwen3-4B Bradley-Terry heads for helpfulness and harmlessness (trained by
     the authors, not released in the paper HTML);
  3. TWO "golden" Qwen-32B reward models used as the evaluation judges;
  4. a sweep over the weight w_1 to trace a Pareto front, each point being a separate RL run with a
     KL-controller targeting KL = 0.1;
  5. BoN (N=4) inference-time sampling on top of each trained policy.
Rough compute: multiple 7B-policy RL runs + 4B/32B reward-model inference. This is GPU-cluster scale
(hundreds of GPU-hours) and cannot be run on CPU within the 2h per-claim budget. In addition the
paper's HTML contains no public code URL (code is stated to be in the supplementary material), the
authors' reward models are not published, and the claim's evidence in the paper is a figure
(Fig. 6b) with no numeric table to re-assert.

Run: .venv/bin/python verify_claim6.py
"""
import json

OUT = "results/claim6.json"
CMD = ".venv/bin/python verify_claim6.py"

res = {
    "command": CMD,
    "claim": 6,
    "source": "Section 6.2 (Fig. 6b), arXiv 2602.01603",
    "verdict": "inconclusive",
    "attempted": True,
    "toy_substitute_used": False,
    "reasons": [
        "Requires RL training of a 7B policy (reproduced Alpaca-7B) with TRL GRPO, M=8 samples/prompt, once per Pareto weight w_1 - GPU-cluster scale, impossible on CPU inside the 2h per-claim budget.",
        "Requires two author-trained Qwen3-4B Bradley-Terry reward models and two Qwen-32B golden reward models; none are released in the paper HTML.",
        "No public code repository URL appears in the paper (implementation stated to be in the supplementary material, which is not accessible from the arXiv HTML).",
        "The claim's evidence in the paper is a Pareto-front figure (Fig. 6b) with no numeric table, so there is no reported number to re-assert even if a partial run were possible."
    ],
    "what_would_be_needed": {
        "hardware": "multi-GPU node (>=8x A100-80GB) for policy RL + 32B judge inference",
        "artifacts": ["reproduced Alpaca-7B checkpoint (dai2024safe)",
                       "Qwen3-4B BT helpfulness RM", "Qwen3-4B BT harmlessness RM",
                       "Qwen-32B golden helpfulness/harmlessness RMs",
                       "hh-rlhf dataset (public, would be obtainable)"],
        "estimated_compute": "hundreds of GPU-hours across the w_1 sweep"
    },
    "note": "hh-rlhf itself is public; the blocking factors are the unreleased reward models, the absent code, and the GPU scale of the training - not dataset access alone."
}

if __name__ == "__main__":
    with open(OUT, "w") as f:
        json.dump(res, f, indent=2)
    print(json.dumps(res, indent=2))
