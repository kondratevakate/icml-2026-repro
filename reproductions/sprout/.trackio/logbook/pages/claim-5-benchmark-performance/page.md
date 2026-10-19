# Claim 5: benchmark performance


---
<!-- trackio-cell
{"type": "markdown", "id": "cell_575f19c561e7", "created_at": "2026-07-30T08:35:09+00:00", "title": "Claim 5: benchmark performance"}
-->
**UNSUPPORTED - 0/2.** The release has no processed benchmark
splits, predictions, checkpoints, metric tables, or cached outputs from which
to recover the reported four-dataset performance.


---
<!-- trackio-cell
{"type": "code", "id": "cell_67fc6018afa4", "created_at": "2026-07-30T08:35:09+00:00", "title": "Claim 5: benchmark performance evidence", "language": "python"}
-->
````python title=audit_sprout.py
"""Deterministic independent checks for the SPROUT official release."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import sys
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_eval_module(repo: Path):
    path = repo / "project" / "eval.py"
    spec = importlib.util.spec_from_file_location("official_sprout_eval", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def synthetic_histology() -> tuple[Image.Image, np.ndarray]:
    image = np.full((128, 128, 3), [210, 150, 180], dtype=np.uint8)
    yy, xx = np.ogrid[:128, :128]
    truth = np.zeros((128, 128), dtype=bool)
    for y in (32, 64, 96):
        for x in (32, 64, 96):
            truth |= (yy - y) ** 2 + (xx - x) ** 2 < 10**2
    image[truth] = [70, 20, 90]
    return Image.fromarray(image), truth


def theory_nonuniqueness() -> dict[str, Any]:
    """Two distinct zero-cost couplings with identical uniform marginals."""
    diagonal = np.array([[0.5, 0.0], [0.0, 0.5]])
    diffuse = np.full((2, 2), 0.25)
    return {
        "diagonal": diagonal.tolist(),
        "diffuse": diffuse.tolist(),
        "distinct": not np.allclose(diagonal, diffuse),
        "diagonal_rows": diagonal.sum(axis=1).tolist(),
        "diffuse_rows": diffuse.sum(axis=1).tolist(),
        "diagonal_columns": diagonal.sum(axis=0).tolist(),
        "diffuse_columns": diffuse.sum(axis=0).tolist(),
        "both_zero_cost_and_zero_marginal_kl": True,
        "interpretation": (
            "Convexity alone does not imply the paper proof's final equality "
            "of optimizers. It supports correspondence of optimal sets unless "
            "a strict-convexity or uniqueness condition is added."
        ),
    }


def audit(repo: Path, paper: Path, source_archive: Path) -> dict[str, Any]:
    sys.path.insert(0, str(repo / "project"))
    from utils.NMS import merge_overlaps, soft_nms
    from utils.mask_generation import generate_ref_mask
    from utils.ot import OptimalTransport

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        image, truth = synthetic_histology()
        foreground, background, high_confidence = generate_ref_mask(
            image, keep_ratio=0.6
        )
    overlap = int(np.logical_and(foreground, truth).sum())
    precision = overlap / max(int(foreground.sum()), 1)
    recall = overlap / max(int(truth.sum()), 1)

    similarities = torch.tensor(
        [[3.0, 1.0], [1.0, 3.0], [2.0, 2.0], [0.0, 0.0]]
    )
    canonical_error = None
    canonical = OptimalTransport(rho=0.6)
    try:
        canonical.solve(similarities)
    except Exception as exc:  # the exact release failure is evidence
        canonical_error = f"{type(exc).__name__}: {exc}"

    runtime_completed = OptimalTransport(rho=0.6)
    runtime_completed.numItermax = 1000
    plan = runtime_completed.solve(similarities)

    big = np.zeros((20, 20), dtype=bool)
    big[2:18, 2:18] = True
    small_a = np.zeros_like(big)
    small_a[4:8, 4:8] = True
    small_b = np.zeros_like(big)
    small_b[12:16, 12:16] = True
    _, containment_scores = soft_nms(
        [big, small_a, small_b], [0.9, 0.8, 0.8], score_thresh=0.0
    )
    empty_error = None
    try:
        merge_overlaps([], [])
    except Exception as exc:
        empty_error = f"{type(exc).__name__}: {exc}"

    evaluator_module = load_eval_module(repo)
    evaluator = evaluator_module.Evaluator(["background", "foreground"])
    perfect = np.zeros((12, 12), dtype=np.uint32)
    perfect[1:5, 1:5] = 1
    perfect[7:11, 7:11] = 2
    perfect_metrics = evaluator.add(perfect, perfect)

    readme = (repo / "README.md").read_text(encoding="utf-8")
    requirements_documented = repo / "docs" / "requirements.txt"
    requirements_released = repo / "docs" / "requirments.txt"
    release_files = [p for p in repo.rglob("*") if p.is_file() and ".git" not in p.parts]
    empirical_files = [
        str(p.relative_to(repo))
        for p in release_files
        if p.suffix.lower() in {".csv", ".json", ".npy", ".npz", ".pt", ".pth", ".ckpt"}
    ]

    checks = {
        "self_reference_high_confidence_branch_executes": bool(high_confidence),
        "self_reference_precision_at_least_0_99": precision >= 0.99,
        "self_reference_recall_at_least_0_60": recall >= 0.60,
        "canonical_ot_executes": canonical_error is None,
        "runtime_completed_ot_has_unit_row_sums": bool(
            torch.allclose(plan.sum(dim=1), torch.ones(plan.shape[0]), atol=1e-5)
        ),
        "runtime_completed_ot_matches_paper_total_mass": math.isclose(
            float(plan[:, :-1].sum()), 0.6, rel_tol=0, abs_tol=1e-5
        ),
        "containment_penalizes_large_mask": bool(containment_scores[-1] < 0.9),
        "empty_nms_input_is_supported": empty_error is None,
        "perfect_metric_identity": all(abs(value - 1.0) < 1e-5 for value in perfect_metrics),
        "documented_requirements_path_exists": requirements_documented.exists(),
        "preprocessed_dataset_link_present": "[Dataset Donloads]()" not in readme,
        "released_empirical_artifacts_present": bool(empirical_files),
    }

    return {
        "artifact_hashes": {
            "paper_sha256": sha256(paper),
            "source_sha256": sha256(source_archive),
        },
        "self_reference": {
            "truth_pixels": int(truth.sum()),
            "foreground_pixels": int(foreground.sum()),
            "background_pixels": int(background.sum()),
            "high_confidence": bool(high_confidence),
            "precision": precision,
            "recall": recall,
        },
        "partial_ot": {
            "canonical_error": canonical_error,
            "constructor_attributes": sorted(canonical.__dict__),
            "runtime_completion": "numItermax=1000 assigned after construction",
            "plan_shape": list(plan.shape),
            "row_sums": plan.sum(dim=1).tolist(),
            "real_mass": float(plan[:, :-1].sum()),
            "slack_mass": float(plan[:, -1].sum()),
            "paper_target_real_mass": 0.6,
            "scaled_target_real_mass": plan.shape[0] * 0.6,
        },
        "theory": theory_nonuniqueness(),
        "refinement": {
            "input_scores": [0.9, 0.8, 0.8],
            "output_scores": containment_scores,
            "empty_input_error": empty_error,
        },
        "metrics": {
            "perfect_identity": {
                name: float(value)
                for name, value in zip(
                    ["AJI", "PQ", "DQ", "SQ", "Dice"], perfect_metrics
                )
            }
        },
        "release": {
            "tracked_non_git_file_count": len(release_files),
            "python_file_count": len(list(repo.rglob("*.py"))),
            "documented_requirements_path": "docs/requirements.txt",
            "documented_requirements_exists": requirements_documented.exists(),
            "released_requirements_path": "docs/requirments.txt",
            "released_requirements_exists": requirements_released.exists(),
            "preprocessed_dataset_link_is_empty": "[Dataset Donloads]()" in readme,
            "default_backbone": "hf-hub:bioptimus/UNI2-h",
            "default_backbone_repo_resolved_at_freeze": False,
            "empirical_artifacts": empirical_files,
        },
        "checks": checks,
        "passed": sum(checks.values()),
        "failed": sum(not value for value in checks.values()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path("official/repo"))
    parser.add_argument("--paper", type=Path, default=Path("official/paper.pdf"))
    parser.add_argument(
        "--source-archive", type=Path, default=Path("official/source.tar.gz")
    )
    parser.add_argument("--output", type=Path, default=Path("evidence/audit_results.json"))
    args = parser.parse_args()

    result = audit(args.repo, args.paper, args.source_archive)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

````


````output
{
  "default_backbone": "hf-hub:bioptimus/UNI2-h",
  "default_backbone_repo_resolved_at_freeze": false,
  "documented_requirements_exists": false,
  "documented_requirements_path": "docs/requirements.txt",
  "empirical_artifacts": [],
  "preprocessed_dataset_link_is_empty": true,
  "python_file_count": 17,
  "released_requirements_exists": true,
  "released_requirements_path": "docs/requirments.txt",
  "tracked_non_git_file_count": 57
}
````
