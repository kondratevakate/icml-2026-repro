#!/usr/bin/env python3
"""Independent numerical audit of SwitchCraft losses against Eqs. 1, 2, and 5."""

from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.util
import json
import sys
import types
from pathlib import Path

import numpy as np
import torch


def load_official_losses(official_repo: Path):
    """Load losses.py while isolating it from Boltz's unrelated dependencies."""

    utility_path = official_repo / "utils" / "mydesign_utils.py"
    utility_tree = ast.parse(utility_path.read_text(encoding="utf-8"))
    required_helpers = {
        "get_mid_points",
        "get_con_loss",
        "_get_con_loss",
        "min_k",
        "_get_helix_loss",
        "_get_sheet_loss",
    }
    helper_nodes = [
        node
        for node in utility_tree.body
        if isinstance(node, ast.FunctionDef) and node.name in required_helpers
    ]
    found = {node.name for node in helper_nodes}
    if found != required_helpers:
        raise AssertionError(f"missing utility helpers: {required_helpers - found}")

    helper_module = types.ModuleType("utils.mydesign_utils")
    helper_module.__dict__["torch"] = torch
    exec(
        compile(
            ast.Module(body=helper_nodes, type_ignores=[]),
            str(utility_path),
            "exec",
        ),
        helper_module.__dict__,
    )
    utils_package = types.ModuleType("utils")
    utils_package.__path__ = []
    utils_package.mydesign_utils = helper_module

    previous_utils = sys.modules.get("utils")
    previous_helper = sys.modules.get("utils.mydesign_utils")
    sys.modules["utils"] = utils_package
    sys.modules["utils.mydesign_utils"] = helper_module
    try:
        losses_path = official_repo / "losses.py"
        spec = importlib.util.spec_from_file_location(
            "switchcraft_official_losses", losses_path
        )
        if spec is None or spec.loader is None:
            raise RuntimeError(f"cannot load {losses_path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        if previous_utils is None:
            sys.modules.pop("utils", None)
        else:
            sys.modules["utils"] = previous_utils
        if previous_helper is None:
            sys.modules.pop("utils.mydesign_utils", None)
        else:
            sys.modules["utils.mydesign_utils"] = previous_helper


def midpoint_bins(device: torch.device) -> torch.Tensor:
    boundaries = torch.linspace(2.0, 22.0, 63, device=device)
    expanded = torch.cat(
        (
            torch.tensor([1.0], device=device),
            boundaries,
            torch.tensor([27.0], device=device),
        )
    )
    return (expanded[:-1] + expanded[1:]) / 2


def independent_min_k(
    values: torch.Tensor, k: int, mask: torch.Tensor
) -> torch.Tensor:
    masked = torch.where(mask.bool(), values, torch.nan)
    ordered = torch.sort(masked, dim=-1).values
    include = (
        torch.arange(ordered.shape[-1], device=ordered.device) < k
    ) & ~torch.isnan(ordered)
    return torch.where(include, ordered, 0.0).sum(-1) / (
        include.sum(-1) + 1e-8
    )


def independent_binding_loss(
    logits: torch.Tensor,
    protein_mask: torch.Tensor,
    ligand_mask: torch.Tensor,
    num_positions: int,
) -> torch.Tensor:
    bins = midpoint_bins(logits.device) < 20.0
    probabilities = torch.softmax(logits, dim=-1)
    truncated = torch.softmax(logits - 1e7 * (~bins), dim=-1)
    contact_entropy = -(
        truncated * torch.log_softmax(logits, dim=-1)
    ).sum(-1)
    top_two_per_position = independent_min_k(
        contact_entropy,
        2,
        ligand_mask,
    )
    return independent_min_k(
        top_two_per_position,
        num_positions,
        protein_mask,
    )


def run_audit(official_repo: Path, seed: int) -> dict[str, object]:
    official = load_official_losses(official_repo)

    torch.manual_seed(seed)
    dtype = torch.float64
    device = torch.device("cpu")
    bins = midpoint_bins(device).to(dtype)

    motif_positions = np.asarray(
        [[0.0, 0.0, 0.0], [5.0, 0.0, 0.0], [30.0, 0.0, 0.0]]
    )
    motif_mask = np.asarray([True, True, True])
    motif = {"cb_pos": motif_positions, "motif_mask": motif_mask}
    motif_logits = torch.randn((1, 3, 3, 64), dtype=dtype)

    official_motif = official.MotifLoss(motif).evaluate(
        {"pdistogram": motif_logits}, device
    )
    distances = torch.from_numpy(
        np.linalg.norm(
            motif_positions[:, None, :] - motif_positions[None, :, :], axis=-1
        )
    ).to(dtype)
    code_mask = (distances > 1e-3) & (distances < 22.0)
    expected_pair_error = (
        torch.softmax(motif_logits, dim=-1)
        * (bins - distances[..., None]) ** 2
    ).sum(-1)
    independent_code_motif = (
        expected_pair_error * code_mask
    ).sum() / code_mask.sum()
    motif_error = float(abs(official_motif - independent_code_motif))
    if motif_error > 1e-9:
        raise AssertionError(f"MotifLoss mismatch: {motif_error}")

    literal_eq1_mask = ~torch.eye(3, dtype=torch.bool)
    literal_eq1 = (
        expected_pair_error * literal_eq1_mask
    ).sum() / literal_eq1_mask.sum()
    anti_motif = official.AntiMotifLoss(motif).evaluate(
        {"pdistogram": motif_logits}, device
    )
    anti_motif_error = float(abs(anti_motif + 0.5 * official_motif))
    if anti_motif_error > 1e-9:
        raise AssertionError(f"AntiMotifLoss coefficient mismatch: {anti_motif_error}")

    total_tokens = 5
    binding_logits = torch.randn(
        (1, total_tokens, total_tokens, 64), dtype=dtype
    )
    asym_id = torch.tensor([[0, 0, 0, 1, 1]])
    binding_dict = {
        "asym_id": asym_id,
        "pdistogram": binding_logits,
    }
    binding = official.LigandContactLoss().evaluate(
        binding_dict,
        device,
        {"num_optimizing_binder_pos": 3},
    )
    independent_binding = independent_binding_loss(
        binding_logits,
        protein_mask=asym_id == 0,
        ligand_mask=asym_id != 0,
        num_positions=3,
    )
    binding_error = float(torch.max(abs(binding - independent_binding)))
    if binding_error > 1e-9:
        raise AssertionError(f"BindingLoss mismatch: {binding_error}")

    anti_binding = official.AntiLigandContactLoss().evaluate(
        binding_dict,
        device,
        {"num_optimizing_binder_pos": 3},
    )
    anti_binding_ratio = float((anti_binding / binding).item())

    state0 = torch.randn((1, 4, 4, 64), dtype=dtype, requires_grad=True)
    state1 = torch.randn((1, 4, 4, 64), dtype=dtype, requires_grad=True)
    mol_type = torch.zeros((1, 4), dtype=torch.long)
    strength = 2.5
    conformational = official.ConfChangeLoss(strength=strength).evaluate(
        [
            {"mol_type": mol_type, "pdistogram": state0},
            {"mol_type": mol_type, "pdistogram": state1},
        ],
        device,
    )
    p = torch.softmax(state0, dim=-1).clamp(min=1e-6)
    q = torch.softmax(state1, dim=-1).clamp(min=1e-6)
    mixture = (0.5 * (p + q)).clamp(min=1e-6)
    jsd = 0.5 * (
        p * (p.log() - mixture.log()) + q * (q.log() - mixture.log())
    ).sum(-1)
    independent_conformational = -strength * jsd.max(-1).values.mean()
    conformational_error = float(
        abs(conformational - independent_conformational).detach()
    )
    if conformational_error > 1e-9:
        raise AssertionError(
            f"ConformationalChangeLoss mismatch: {conformational_error}"
        )
    conformational.backward()
    gradient_norm = float(state0.grad.norm() + state1.grad.norm())
    if not np.isfinite(gradient_norm) or gradient_norm <= 0:
        raise AssertionError("conformational loss did not propagate finite gradients")

    return {
        "status": "passed_with_divergences",
        "seed": seed,
        "official_repo": str(official_repo.resolve()),
        "source_sha256": {
            "losses.py": hashlib.sha256(
                (official_repo / "losses.py").read_bytes()
            ).hexdigest(),
            "utils/mydesign_utils.py": hashlib.sha256(
                (official_repo / "utils" / "mydesign_utils.py").read_bytes()
            ).hexdigest(),
        },
        "torch_version": torch.__version__,
        "equation_checks": {
            "motif_code_rule_absolute_error": motif_error,
            "anti_motif_coefficient_absolute_error": anti_motif_error,
            "binding_absolute_error": binding_error,
            "conformational_change_absolute_error": conformational_error,
            "conformational_gradient_norm": gradient_norm,
        },
        "code_paper_divergences": {
            "motif_loss": {
                "official_code_value": float(official_motif),
                "literal_equation_1_value": float(literal_eq1),
                "reason": (
                    "Official code excludes motif pairs at distances >=22 A; "
                    "Equation 1 sums all distinct motif pairs."
                ),
            },
            "anti_binding": {
                "official_default_ratio_to_binding": anti_binding_ratio,
                "paper_ratio_to_binding": -0.5,
                "reason": (
                    "AntiLigandContactLoss defaults to -1 times the contact "
                    "loss, while Section 3.1 states -0.5."
                ),
            },
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--official-repo", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=260531236)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = run_audit(args.official_repo, args.seed)
    rendered = json.dumps(report, indent=2)
    print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
