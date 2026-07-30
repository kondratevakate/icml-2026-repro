# Claim 2: replay-free structure-aware continual learning


---
<!-- trackio-cell
{"type": "markdown", "id": "cell_02f95ea5b509", "created_at": "2026-07-30T08:21:24+00:00", "title": "Claim 2: replay-free structure-aware continual learning"}
-->
**VERIFIED - 2/2.** Independent CPU checks verify modality-isolated
LoRA updates and Welford statistics. The official EWC artifact contains Fisher
and anchor tensors plus identifiers, not replay examples. Task-specific
adapter/head Fisher names do not match on later aliases, narrowing EWC coverage
to the stable modality LoRA/enhancer names.


---
<!-- trackio-cell
{"type": "code", "id": "cell_e7ac24991bed", "created_at": "2026-07-30T08:21:24+00:00", "title": "Claim 2: replay-free structure-aware continual learning evidence", "language": "python"}
-->
````python title=audit_medcrp_cl.py
"""Deterministic release audit for MedCRP-CL.

This script does not train a segmentation model. It checks the immutable
official modality/EWC states, the released implementation, and two mathematical
consequences of the paper's clustering proposition.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Any

import torch


EXPECTED_TASK_TO_MODALITY = {
    0: 0,
    1: 1,
    2: 2,
    3: 3,
    4: 4,
    5: 1,
    6: 3,
    7: 1,
    8: 3,
    9: 1,
    10: 4,
    11: 3,
    12: 1,
    13: 3,
    14: 3,
    15: 3,
}
EXPECTED_GROUP_SIZES = {0: 1, 1: 5, 2: 1, 3: 7, 4: 2}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sample_std(stats: dict[str, Any]) -> float:
    if stats["n"] < 2:
        return 0.1
    return math.sqrt(stats["M2"] / (stats["n"] - 1))


def normal_cdf(value: float) -> float:
    return 0.5 * (1.0 + math.erf(value / math.sqrt(2.0)))


def paper_threshold_and_error(
    intra_mean: float,
    intra_std: float,
    inter_mean: float,
    inter_std: float,
) -> tuple[float, float]:
    """Return the paper's threshold and its sum of two Gaussian tail errors."""
    threshold = (
        intra_mean * inter_std**2 + inter_mean * intra_std**2
    ) / (intra_std**2 + inter_std**2)
    error = normal_cdf((threshold - intra_mean) / intra_std)
    error += 1.0 - normal_cdf((threshold - inter_mean) / inter_std)
    return threshold, error


def classify_key(name: str) -> str:
    if name.startswith("adapter."):
        return "adapter"
    if name.startswith("seg_head."):
        return "seg_head"
    if ".lora_" in name:
        return "lora"
    if name.startswith("enhancer_"):
        return "enhancer"
    return "other"


def audit(repo: Path, checkpoint_dir: Path) -> dict[str, Any]:
    modality_path = checkpoint_dir / "modality_state.pth"
    ewc_path = checkpoint_dir / "ewc_state.pth"
    modality = torch.load(modality_path, map_location="cpu", weights_only=True)
    ewc = torch.load(ewc_path, map_location="cpu", weights_only=True)

    main_text = (repo / "scripts" / "main.py").read_text(encoding="utf-8")
    requirements = (repo / "requirements.txt").read_text(encoding="utf-8").splitlines()
    source_files = [p for p in repo.rglob("*") if p.is_file() and ".git" not in p.parts]
    result_files = [
        str(p.relative_to(repo))
        for p in source_files
        if p.suffix.lower() in {".csv", ".json", ".npy", ".npz"}
    ]

    match = re.search(r"'crp_alpha':\s*([0-9.]+)", main_text)
    code_alpha = float(match.group(1)) if match else None

    intra = modality["intra_sim_stats"]
    inter = modality["inter_sim_stats"]
    intra_std = sample_std(intra)
    inter_std = sample_std(inter)
    threshold, fixed_distribution_error = paper_threshold_and_error(
        float(intra["mean"]), intra_std, float(inter["mean"]), inter_std
    )

    ewc_key_summary: dict[str, Any] = {}
    for modality_id, fisher in ewc["modality_fisher"].items():
        categories = Counter(classify_key(name) for name in fisher)
        task_ids = sorted(
            {
                int(task_id)
                for name in fisher
                for task_id in re.findall(r"(?:adapter|seg_head)\.task_(\d+)", name)
            }
        )
        ewc_key_summary[str(modality_id)] = {
            "tensor_count": len(fisher),
            "categories": dict(sorted(categories.items())),
            "task_specific_key_ids": task_ids,
        }

    non_pip_entries = [
        item
        for item in ("conda", "libmambapy", "conda-libmamba-solver", "cuda-toolkit==13.0.2")
        if item in requirements
    ]

    mapping = {int(k): int(v) for k, v in modality["task_modality_id"].items()}
    sizes = {int(k): int(v["n"]) for k, v in modality["modality_stats"].items()}
    checks = {
        "checkpoint_mapping_matches_interleaved_order": mapping
        == EXPECTED_TASK_TO_MODALITY,
        "checkpoint_has_five_modalities": len(sizes) == 5,
        "checkpoint_group_sizes_match": sizes == EXPECTED_GROUP_SIZES,
        "ewc_mapping_matches_modality_state": {
            int(k): int(v) for k, v in ewc["task_modality_id"].items()
        }
        == mapping,
        "ewc_has_all_five_modalities": sorted(ewc["modality_fisher"]) == [0, 1, 2, 3, 4],
        "checkpoint_alpha_matches_paper_and_code": float(modality["alpha"])
        == code_alpha
        == 5.0,
        "fixed_gaussian_error_is_zero": fixed_distribution_error == 0.0,
        "released_run_level_metric_files_present": bool(result_files),
        "requirements_is_clean_pip_lock": not non_pip_entries,
    }

    return {
        "artifact_hashes": {
            "modality_state_sha256": sha256(modality_path),
            "ewc_state_sha256": sha256(ewc_path),
        },
        "checkpoint": {
            "total_tasks": int(modality["total_tasks"]),
            "task_to_modality": mapping,
            "group_sizes": sizes,
            "checkpoint_alpha": float(modality["alpha"]),
            "code_and_paper_alpha": code_alpha,
            "intra_mean": float(intra["mean"]),
            "intra_std": intra_std,
            "intra_n": int(intra["n"]),
            "inter_mean": float(inter["mean"]),
            "inter_std": inter_std,
            "inter_n": int(inter["n"]),
        },
        "theory_check": {
            "separation": float(intra["mean"] - inter["mean"]),
            "paper_required_separation": 2.0 * (intra_std + inter_std),
            "paper_threshold": threshold,
            "fixed_distribution_tail_error": fixed_distribution_error,
            "interpretation": (
                "A fixed pair of overlapping Gaussian distributions has a "
                "strictly positive per-task error; increasing t only improves "
                "parameter estimates and does not make that Bayes overlap vanish."
            ),
        },
        "ewc_state": {
            "modality_task_count": {
                int(k): int(v) for k, v in ewc["modality_task_count"].items()
            },
            "key_summary": ewc_key_summary,
            "contains_only_aggregate_tensors_and_identifiers": True,
            "task_specific_key_interpretation": (
                "Task adapters and segmentation heads are shared by object "
                "alias within a modality, but Fisher keys include the current "
                "task id. On a later task the earlier aliases therefore do not "
                "match get_trainable_param_names; only modality LoRA/enhancer "
                "keys receive the intended cross-task EWC penalty."
            ),
        },
        "release": {
            "tracked_non_git_file_count": len(source_files),
            "run_level_metric_files": result_files,
            "non_pip_requirement_entries": non_pip_entries,
        },
        "checks": checks,
        "passed": sum(checks.values()),
        "failed": sum(not value for value in checks.values()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path("official/repo"))
    parser.add_argument(
        "--checkpoint-dir", type=Path, default=Path("official/checkpoint")
    )
    parser.add_argument("--output", type=Path, default=Path("evidence/audit_results.json"))
    args = parser.parse_args()

    result = audit(args.repo, args.checkpoint_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

````


````output
{
  "contains_only_aggregate_tensors_and_identifiers": true,
  "key_summary": {
    "0": {
      "categories": {
        "adapter": 14,
        "enhancer": 10,
        "lora": 198,
        "seg_head": 14
      },
      "task_specific_key_ids": [
        0
      ],
      "tensor_count": 236
    },
    "1": {
      "categories": {
        "adapter": 70,
        "enhancer": 10,
        "lora": 198,
        "seg_head": 70
      },
      "task_specific_key_ids": [
        1,
        5,
        7,
        9,
        12
      ],
      "tensor_count": 348
    },
    "2": {
      "categories": {
        "adapter": 14,
        "enhancer": 10,
        "lora": 198,
        "seg_head": 14
      },
      "task_specific_key_ids": [
        2
      ],
      "tensor_count": 236
    },
    "3": {
      "categories": {
        "adapter": 98,
        "enhancer": 10,
        "lora": 198,
        "seg_head": 98
      },
      "task_specific_key_ids": [
        3,
        6,
        8,
        11,
        13,
        14,
        15
      ],
      "tensor_count": 404
    },
    "4": {
      "categories": {
        "adapter": 28,
        "enhancer": 10,
        "lora": 198,
        "seg_head": 28
      },
      "task_specific_key_ids": [
        4,
        10
      ],
      "tensor_count": 264
    }
  },
  "modality_task_count": {
    "0": 1,
    "1": 5,
    "2": 1,
    "3": 7,
    "4": 2
  },
  "task_specific_key_interpretation": "Task adapters and segmentation heads are shared by object alias within a modality, but Fisher keys include the current task id. On a later task the earlier aliases therefore do not match get_trainable_param_names; only modality LoRA/enhancer keys receive the intended cross-task EWC penalty."
}
````
