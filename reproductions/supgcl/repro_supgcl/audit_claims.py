#!/usr/bin/env python3
"""Independent finite-distribution audit of SupGCL Theorem 1 and Corollary 1."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

import numpy as np


CODE_COMMIT = "a384eeaeffaabfd140335e19a0633ab400ba3bcb"
ARCHIVE_SHA256 = "20d8c3ef3ce61e816e88817d2cccabddd7b0fc8e8d3b0c3fa2ba1436b5e55ec8"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def softmax(values: np.ndarray, axis: int = -1) -> np.ndarray:
    shifted = values - np.max(values, axis=axis, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.sum(exp, axis=axis, keepdims=True)


def kl(p: np.ndarray, q: np.ndarray) -> float:
    return float(np.sum(p * (np.log(p) - np.log(q))))


def normalized(values: np.ndarray, axis: int = -1) -> np.ndarray:
    return values / np.sum(values, axis=axis, keepdims=True)


def theorem_one_probe(seed: int) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    augmentations, nodes = 4, 5
    p_aug = normalized(rng.uniform(0.1, 2.0, (augmentations, augmentations)))
    q_aug = normalized(rng.uniform(0.1, 2.0, (augmentations, augmentations)))
    p_node = normalized(rng.uniform(0.1, 2.0, (nodes, nodes)))
    q_node = normalized(
        rng.uniform(0.1, 2.0, (augmentations, augmentations, nodes, nodes))
    )

    joint_p = np.empty((augmentations, augmentations, nodes, nodes))
    joint_q = np.empty_like(joint_p)
    for a in range(augmentations):
        for b in range(augmentations):
            for i in range(nodes):
                joint_p[a, b, i] = (
                    p_aug[a, b] * p_node[i] / (augmentations * nodes)
                )
                joint_q[a, b, i] = (
                    q_aug[a, b] * q_node[a, b, i] / (augmentations * nodes)
                )

    lhs = kl(joint_p, joint_q)
    augmentation_kl = sum(
        kl(p_aug[a], q_aug[a]) for a in range(augmentations)
    ) / augmentations
    expected_node_kl = 0.0
    for a in range(augmentations):
        for b in range(augmentations):
            node_kl = sum(
                kl(p_node[i], q_node[a, b, i]) for i in range(nodes)
            ) / nodes
            expected_node_kl += p_aug[a, b] * node_kl / augmentations
    rhs = augmentation_kl + expected_node_kl
    return {"lhs": lhs, "rhs": rhs, "absolute_error": abs(lhs - rhs)}


def corollary_probe(seed: int) -> dict:
    rng = np.random.default_rng(seed)
    augmentations, nodes = 5, 4
    teacher_similarity = rng.normal(size=(augmentations, augmentations))
    student_similarity = rng.normal(size=(augmentations, augmentations))
    p_node = normalized(rng.uniform(0.1, 2.0, (nodes, nodes)))
    q_node = normalized(
        rng.uniform(0.1, 2.0, (augmentations, augmentations, nodes, nodes))
    )
    node_losses = np.empty((augmentations, augmentations))
    for a in range(augmentations):
        for b in range(augmentations):
            node_losses[a, b] = sum(
                kl(p_node[i], q_node[a, b, i]) for i in range(nodes)
            ) / nodes
    limiting_node_loss = float(np.mean(node_losses))

    temperatures = np.asarray([1.0, 10.0, 100.0, 1_000.0, 10_000.0, 100_000.0])
    records = []
    for tau in temperatures:
        p_aug = softmax(teacher_similarity / tau)
        q_aug = softmax(student_similarity / tau)
        weighted_node = float(np.sum(p_aug * node_losses) / augmentations)
        augmentation_kl = sum(
            kl(p_aug[a], q_aug[a]) for a in range(augmentations)
        ) / augmentations
        total = weighted_node + augmentation_kl
        uniform = np.full_like(p_aug, 1.0 / augmentations)
        records.append(
            {
                "tau": float(tau),
                "loss": total,
                "distance_to_node_limit": abs(total - limiting_node_loss),
                "max_teacher_distance_to_uniform": float(
                    np.max(np.abs(p_aug - uniform))
                ),
                "max_student_distance_to_uniform": float(
                    np.max(np.abs(q_aug - uniform))
                ),
                "augmentation_kl": augmentation_kl,
            }
        )
    return {"limiting_node_loss": limiting_node_loss, "records": records}


def require_patterns(text: str, patterns: dict[str, str], label: str) -> dict[str, bool]:
    found = {
        name: re.search(pattern, text, flags=re.DOTALL) is not None
        for name, pattern in patterns.items()
    }
    missing = [name for name, present in found.items() if not present]
    if missing:
        raise AssertionError(f"Missing {label} anchors: {missing}")
    return found


def audit(code_root: Path, paper_source: Path) -> dict:
    methodology_path = paper_source / "04Methodology.tex"
    proof_path = paper_source / "A01ProofOfTheorem.tex"
    sampler_path = code_root / "Pretrain" / "SupGCL" / "samplers.py"
    main_path = code_root / "Pretrain" / "SupGCL" / "main.py"
    loss_path = code_root / "Pretrain" / "SupGCL" / "losses.py"

    methodology = methodology_path.read_text(encoding="utf-8")
    proof = proof_path.read_text(encoding="utf-8")
    sampler = sampler_path.read_text(encoding="utf-8")
    main = main_path.read_text(encoding="utf-8")
    losses = loss_path.read_text(encoding="utf-8")

    paper_anchors = require_patterns(
        methodology + "\n" + proof,
        {
            "factorization_assumption": r"p_\\phi\(i,j,a,b\)\s*=\s*p\(i,j\)p_\\phi\(a,b\)",
            "kl_decomposition": r"D_.*?KL.*?p_\\phi\s*\(a,b\).*?Loss.*?Aug",
            "corollary_limit": r"lim_.*?tau.*?infty.*?Loss.*?SupGCL.*?Loss.*?Node",
            "uniform_limit": r"p_\\phi\(b\\|a\)\s*\\rightarrow\s*\\mathrm\{U\}",
        },
        "paper",
    )
    code_anchors = require_patterns(
        sampler + "\n" + main + "\n" + losses,
        {
            "lincs_graphs": r"lincs_graphs",
            "knockdown_metadata": r"knockdown_metadata",
            "kd_gene_mapping": r"kd_gene_to_sample_ids",
            "teacher_sampler": r"TeacherSampler",
            "augmentation_kl": r"def loss_aug",
            "node_infonce": r"def info_nce_loss",
        },
        "official-code",
    )

    theorem_trials = [theorem_one_probe(seed) for seed in range(100)]
    theorem_max_error = max(item["absolute_error"] for item in theorem_trials)
    if theorem_max_error > 1e-12:
        raise AssertionError("Independent KL chain-rule audit failed")

    corollary_trials = [corollary_probe(seed) for seed in range(25)]
    final_distances = [
        item["records"][-1]["distance_to_node_limit"] for item in corollary_trials
    ]
    final_aug_kls = [
        item["records"][-1]["augmentation_kl"] for item in corollary_trials
    ]
    monotonic_failures = 0
    for item in corollary_trials:
        uniform_errors = [
            max(
                row["max_teacher_distance_to_uniform"],
                row["max_student_distance_to_uniform"],
            )
            for row in item["records"]
        ]
        monotonic_failures += sum(
            later >= earlier
            for earlier, later in zip(uniform_errors, uniform_errors[1:])
        )
    if monotonic_failures:
        raise AssertionError("Augmentation distributions did not approach uniform")
    if max(final_distances) > 1e-4 or max(final_aug_kls) > 1e-8:
        raise AssertionError("Finite-temperature corollary probe did not converge")

    not_executed = [
        {
            "id": cid,
            "verdict": "NOT_EXECUTED",
            "score": 0,
            "reason": (
                "Requires the 4.3 GB official graph archive, GPU pretraining, "
                "and repeated downstream evaluation."
            ),
        }
        for cid in ("C3", "C4", "C5", "C6")
    ]
    return {
        "provenance": {
            "official_code_commit": CODE_COMMIT,
            "arxiv_source_archive_sha256": ARCHIVE_SHA256,
            "methodology_tex_sha256": sha256(methodology_path),
            "proof_tex_sha256": sha256(proof_path),
            "sampler_sha256": sha256(sampler_path),
            "losses_sha256": sha256(loss_path),
        },
        "claims": [
            {
                "id": "C1",
                "verdict": "VERIFIED",
                "score": 2,
                "paper_anchors": paper_anchors,
                "official_code_anchors": code_anchors,
                "independent_probability_tables": len(theorem_trials),
                "max_chain_rule_absolute_error": theorem_max_error,
                "finding": (
                    "The KL decomposition holds independently, and pinned "
                    "source connects LINCS knockdown graphs to the sampler."
                ),
            },
            {
                "id": "C2",
                "verdict": "VERIFIED",
                "score": 2,
                "independent_probability_tables": len(corollary_trials),
                "temperatures": corollary_trials[0]["records"],
                "max_final_distance_to_node_limit": max(final_distances),
                "max_final_augmentation_kl": max(final_aug_kls),
                "uniform_convergence_monotonic_failures": monotonic_failures,
                "finding": (
                    "As tau_a grows, p and q become uniform, augmentation KL "
                    "vanishes, and the objective reaches the uniform node-loss average."
                ),
            },
            *not_executed,
        ],
        "implementation_scope_note": (
            "The theorem audit does not assert that the released sampled loss "
            "is an exact unbiased estimator of the full mathematical objective."
        ),
        "prepared_score": 4,
        "maximum_score": 12,
        "leaderboard_or_peer_evidence_used": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--code-root", type=Path, required=True)
    parser.add_argument("--paper-source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = audit(args.code_root, args.paper_source)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
