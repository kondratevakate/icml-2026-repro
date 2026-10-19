#!/usr/bin/env python3
"""Independent implementation-conformance audits for MedMamba.

The fake Mamba module preserves tensors and records shapes. It is not used to
test model quality; it isolates shape, dependency, and gradient claims in the
released MedMamba integration code on a CPU-only machine.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn


class ShapeRecordingMamba(nn.Module):
    seen_shapes: list[list[int]] = []

    def __init__(self, d_model: int, **_: object) -> None:
        super().__init__()
        self.d_model = d_model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        type(self).seen_shapes.append(list(x.shape))
        return x


def install_fake_mamba() -> None:
    module = types.ModuleType("mamba_ssm")
    module.Mamba = ShapeRecordingMamba
    sys.modules["mamba_ssm"] = module


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def max_abs(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a - b).abs().max().item())


def load_official(code_root: Path):
    install_fake_mamba()
    sys.path.insert(0, str(code_root.resolve()))
    for name in (
        "layers.DiffMamba_Layer",
        "layers.SpatialMamba_Layer",
        "models.MedMamba",
    ):
        sys.modules.pop(name, None)
    diff = importlib.import_module("layers.DiffMamba_Layer")
    spatial = importlib.import_module("layers.SpatialMamba_Layer")
    model = importlib.import_module("models.MedMamba")
    return diff, spatial, model


def run_audit(code_root: Path, source_root: Path) -> dict[str, object]:
    torch.manual_seed(20260729)
    diff, spatial, medmamba = load_official(code_root)

    paper_method = (source_root / "method.tex").read_text(encoding="utf-8")
    paper_full = (source_root / "example_paper.tex").read_text(
        encoding="utf-8", errors="replace"
    )
    paper_experiments = (source_root / "exp.tex").read_text(
        encoding="utf-8", errors="replace"
    )

    embedding = medmamba.MultiScaleEmbedding(
        enc_in=4, d_model=12, kernel_sizes=[3, 5, 7], dropout=0.0
    ).eval()
    sample = torch.randn(2, 16, 4)
    embedded = embedding(sample)
    conv_groups = [layer.groups for layer in embedding.convs]

    diff_block = diff.DiffSSMBlock(d_model=12, dropout=0.0).eval()
    nonzero = torch.arange(1, 1 + 2 * 5 * 12, dtype=torch.float32).reshape(2, 5, 12)
    observed_diff = diff_block._compute_diff(nonzero)
    declared_diff = torch.cat(
        [torch.zeros_like(nonzero[:, :1]), nonzero[:, 1:] - nonzero[:, :-1]],
        dim=1,
    )

    frequency = diff.FrequencyBranch(d_model=12, dropout=0.0)

    learner = spatial.AdaptiveGraphLearner(enc_in=4, node_dim=3)
    graph_input_a = torch.randn(2, 16, 12)
    graph_input_b = torch.randn(2, 16, 12) * 7 + 13
    adjacency_a = learner(graph_input_a)
    adjacency_b = learner(graph_input_b)

    block = spatial.SpatialGraphMambaBlock(
        enc_in=4, d_model=12, node_dim=3, dropout=0.0
    ).eval()
    block_input = torch.randn(2, 16, 12)
    output_before, adj_before = block(block_input)
    with torch.no_grad():
        block.graph_learner.nodevec1.add_(10 * torch.randn_like(block.graph_learner.nodevec1))
        block.graph_learner.nodevec2.add_(10 * torch.randn_like(block.graph_learner.nodevec2))
    output_after, adj_after = block(block_input)

    block.zero_grad(set_to_none=True)
    classification_proxy, _ = block(block_input)
    classification_proxy.square().mean().backward()
    graph_gradients = {
        name: None if parameter.grad is None else float(parameter.grad.abs().max().item())
        for name, parameter in block.graph_learner.named_parameters()
    }

    ShapeRecordingMamba.seen_shapes.clear()
    configs = SimpleNamespace(
        task_name="classification",
        enc_in=4,
        d_model=12,
        d_ff=24,
        e_layers=1,
        num_class=3,
        dropout=0.0,
        d_state=4,
        d_conv=2,
        expand=1,
        nodedim=3,
    )
    full_model = medmamba.Model(configs).eval()
    logits, _ = full_model(sample, None, None, None)
    mamba_shapes = list(ShapeRecordingMamba.seen_shapes)

    code_files = {
        name: sha256(code_root / relative)
        for name, relative in {
            "MedMamba.py": Path("models/MedMamba.py"),
            "DiffMamba_Layer.py": Path("layers/DiffMamba_Layer.py"),
            "SpatialMamba_Layer.py": Path("layers/SpatialMamba_Layer.py"),
            "exp_classification.py": Path("exp/exp_classification.py"),
        }.items()
    }

    claims = [
        {
            "id": "C1",
            "verdict": "FALSIFIED_IN_RELEASED_IMPLEMENTATION",
            "score": 2,
            "paper_anchor_present": (
                r"\mathbb{R}^{T\times C\times D}" in paper_method
            ),
            "observed_embedding_shape": list(embedded.shape),
            "paper_declared_rank": 4,
            "observed_rank": embedded.ndim,
            "conv_groups": conv_groups,
            "input_channels": 4,
            "reason": "The channel axis is collapsed into BxLxD and convolutions use groups=1.",
        },
        {
            "id": "C2",
            "verdict": "FALSIFIED_IN_RELEASED_IMPLEMENTATION",
            "score": 2,
            "paper_anchor_present": r"\big[\mathbf{0};" in paper_method,
            "observed_first_step_max_abs": float(observed_diff[:, 0].abs().max().item()),
            "declared_first_step_max_abs": float(declared_diff[:, 0].abs().max().item()),
            "operator_max_abs_difference": max_abs(observed_diff, declared_diff),
            "observed_first_step_equals_input": bool(
                torch.equal(observed_diff[:, 0], nonzero[:, 0])
            ),
            "reason": "Left zero-padding followed by x - padded[:-1] makes delta[0]=x[0], not zero.",
        },
        {
            "id": "C3",
            "verdict": "FALSIFIED_IN_RELEASED_IMPLEMENTATION",
            "score": 2,
            "paper_anchor_present": "frequency" in paper_method.lower(),
            "real_weight_shape": list(frequency.freq_weight_real.shape),
            "imag_weight_shape": list(frequency.freq_weight_imag.shape),
            "fft_bins_for_probe": sample.shape[1] // 2 + 1,
            "has_frequency_axis": frequency.freq_weight_real.ndim >= 2,
            "reason": "The learned complex gain has shape D and is broadcast identically across all FFT bins.",
        },
        {
            "id": "C4",
            "verdict": "FALSIFIED_IN_RELEASED_IMPLEMENTATION",
            "score": 2,
            "paper_anchor_present": "sample-conditioned" in paper_method,
            "adjacency_input_change_max_abs": max_abs(adjacency_a, adjacency_b),
            "adjacency_parameter_mutation_max_abs": max_abs(adj_before, adj_after),
            "output_parameter_mutation_max_abs": max_abs(output_before, output_after),
            "classification_proxy_graph_gradients": graph_gradients,
            "reason": "The learner ignores x and adjacency is disconnected from the returned representation.",
        },
        {
            "id": "C5",
            "verdict": "FALSIFIED_IN_RELEASED_IMPLEMENTATION",
            "score": 2,
            "paper_anchor_present": (
                r"\widetilde{\mathbf{A}}\,\mathbf{Z}" in paper_method
                and "over channels" in paper_method
            ),
            "mamba_input_shapes": mamba_shapes,
            "input_time_length": 16,
            "input_channel_count": 4,
            "all_mamba_sequence_axes_equal_time": all(
                shape[1] == 16 for shape in mamba_shapes
            ),
            "logits_shape": list(logits.shape),
            "reason": "All released Mamba calls scan L=time; adjacency is computed but not used in the output path.",
        },
        {
            "id": "C6",
            "verdict": "INCONCLUSIVE_NOT_EXECUTED",
            "score": 0,
            "paper_anchor_present": (
                "20 top-1 and 3 top-2 results out of 30 entries"
                in paper_full + paper_experiments
            ),
            "released_dataset_scripts": ["scripts/APAVA_Subject.sh"],
            "required_datasets": ["ADFTD", "APAVA", "PTB", "PTB-XL", "TDBRAIN"],
            "reason": "No fresh five-dataset, five-seed execution was performed.",
        },
    ]

    return {
        "paper": {
            "title": "MedMamba: Multi-View State Space Models with Adaptive Graph Learning for Medical Time Series Classification",
            "forum_id": "qPqJH0heR0",
            "arxiv": "2605.24961v1",
        },
        "official_commit": "418da50664338bc1d766394ee9c231496ab4de97",
        "code_sha256": code_files,
        "paper_source_sha256": {
            name: sha256(source_root / name)
            for name in ("example_paper.tex", "method.tex", "exp.tex")
        },
        "audit_seed": 20260729,
        "scope": "Implementation conformance; fake Mamba is used only for shape and dependency tracing.",
        "claims": claims,
        "prepared_score": sum(int(claim["score"]) for claim in claims),
        "maximum_score": 12,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--official-code-root", type=Path, required=True)
    parser.add_argument("--paper-source-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    evidence = run_audit(args.official_code_root, args.paper_source_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(evidence, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        f"Prepared {evidence['prepared_score']}/{evidence['maximum_score']} "
        f"points; evidence={args.output}"
    )


if __name__ == "__main__":
    main()
