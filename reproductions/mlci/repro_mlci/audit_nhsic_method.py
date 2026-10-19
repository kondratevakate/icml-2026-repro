#!/usr/bin/env python3
"""Audit the Section 5 DeepSets and weighted nHSIC construction."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch import nn


class DeepSetScore(nn.Module):
    def __init__(self, vocabulary_size: int, width: int = 128) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocabulary_size, width)
        self.phi = nn.Sequential(
            nn.Linear(width, width),
            nn.ReLU(),
            nn.Linear(width, width),
            nn.ReLU(),
        )
        self.pool_norm = nn.LayerNorm(2 * width)
        self.rho = nn.Sequential(
            nn.Linear(2 * width, width),
            nn.ReLU(),
            nn.Linear(width, width),
            nn.ReLU(),
            nn.Linear(width, 1),
        )

    def forward(
        self, tokens: torch.Tensor, valid: torch.Tensor
    ) -> torch.Tensor:
        features = self.phi(self.embedding(tokens))
        mask = valid.unsqueeze(-1)
        count = mask.sum(dim=1).clamp_min(1)
        mean = (features * mask).sum(dim=1) / count
        max_values = features.masked_fill(~mask, -torch.inf).amax(dim=1)
        pooled = self.pool_norm(torch.cat([mean, max_values], dim=1))
        return self.rho(pooled).squeeze(-1)


def centered(matrix: torch.Tensor) -> torch.Tensor:
    size = matrix.shape[0]
    h = torch.eye(size, dtype=matrix.dtype)
    h -= torch.ones((size, size), dtype=matrix.dtype) / size
    return h @ matrix @ h


def normalized_hsic(
    scores: torch.Tensor, labels: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    distances = (scores[:, None] - scores[None, :]).abs()
    positive = distances.detach()[distances.detach() > 0]
    bandwidth = positive.median().clamp_min(1e-3)
    k = torch.exp(-(distances**2) / (2 * bandwidth**2))
    l = (labels[:, None] == labels[None, :]).to(scores.dtype)
    kc = centered(k)
    lc = centered(l)
    denominator = kc.norm().clamp_min(1e-4) * lc.norm()
    value = (kc * lc).sum() / denominator
    return value, lc


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="results/nhsic_method.json")
    args = parser.parse_args()

    torch.manual_seed(260617450)
    batch, length, tasks = 48, 13, 4
    tokens = torch.randint(2, 80, (batch, length))
    valid = torch.zeros((batch, length), dtype=torch.bool)
    lengths = torch.randint(2, length + 1, (batch,))
    for row, row_length in enumerate(lengths.tolist()):
        valid[row, :row_length] = True
    tokens[~valid] = 0

    severity = torch.randn(batch)
    labels = torch.stack(
        [
            severity > 0.1,
            severity.square() > 0.7,
            severity + 0.4 * torch.randn(batch) > -0.2,
            torch.sin(1.7 * severity) > 0,
        ],
        dim=1,
    ).to(torch.int64)

    model = DeepSetScore(vocabulary_size=80)
    scores = model(tokens, valid)

    permutations = torch.stack([torch.randperm(length) for _ in range(batch)])
    shuffled_tokens = tokens.gather(1, permutations)
    shuffled_valid = valid.gather(1, permutations)
    shuffled_scores = model(shuffled_tokens, shuffled_valid)
    permutation_error = (scores - shuffled_scores).abs().max().item()

    changed_padding = tokens.clone()
    changed_padding[~valid] = torch.randint(
        2, 80, (int((~valid).sum().item()),)
    )
    padding_scores = model(changed_padding, valid)
    padding_error = (scores - padding_scores).abs().max().item()

    values = []
    rank_one_errors = []
    for task in range(tasks):
        value, lc = normalized_hsic(scores, labels[:, task])
        values.append(value)
        centered_label = labels[:, task].to(scores.dtype)
        centered_label -= centered_label.mean()
        expected = 2 * torch.outer(centered_label, centered_label)
        rank_one_errors.append((lc - expected).abs().max().item())

    strengths = torch.stack([value.detach() for value in values])
    h_max = strengths.max()
    weights = torch.pow(h_max / strengths.clamp_min(0.02), 0.25)
    weights = weights.clamp(min=1 / 3, max=3)
    weights = tasks * weights / weights.sum()
    objective = torch.sum(weights * torch.stack(values))
    (-objective).backward()
    gradient_norm = torch.sqrt(
        sum(
            parameter.grad.square().sum()
            for parameter in model.parameters()
            if parameter.grad is not None
        )
    ).item()

    sign_values = [
        normalized_hsic(-scores.detach(), labels[:, task])[0].item()
        for task in range(tasks)
    ]
    sign_error = max(
        abs(value.detach().item() - flipped)
        for value, flipped in zip(values, sign_values)
    )

    checks = {
        "permutation_invariant": permutation_error < 1e-6,
        "padding_is_masked": padding_error < 1e-6,
        "centered_delta_kernel_rank_one": max(rank_one_errors) < 1e-6,
        "all_nhsic_values_are_finite_and_bounded": all(
            torch.isfinite(value) and -1e-7 <= value.item() <= 1 + 1e-7
            for value in values
        ),
        "inverse_strength_weights_are_positive": bool((weights > 0).all()),
        "objective_backpropagates": gradient_norm > 1e-8,
        "rbf_objective_is_sign_invariant": sign_error < 1e-6,
    }
    result = {
        "paper_anchor": "Section 5, DeepSets encoder and weighted nHSIC",
        "scope": (
            "Independent construction audit on diagnostic inputs; "
            "not a MIMIC empirical reproduction."
        ),
        "batch_size": batch,
        "embedding_dimension": 128,
        "tasks": tasks,
        "protocol": {
            "nhsic_floor": 1e-4,
            "weight_epsilon": 0.02,
            "weight_exponent": 0.25,
            "weight_clip_factor": 3.0,
        },
        "nhsic_values": [value.detach().item() for value in values],
        "task_weights": weights.tolist(),
        "weighted_objective": objective.detach().item(),
        "permutation_max_error": permutation_error,
        "padding_max_error": padding_error,
        "delta_rank_one_max_error": max(rank_one_errors),
        "score_sign_invariance_max_error": sign_error,
        "parameter_gradient_norm": gradient_norm,
        "checks": checks,
        "status": "passed" if all(checks.values()) else "failed",
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    if result["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
