#!/usr/bin/env python3
"""Run the official cached ROCP evaluations in parallel with resumable outputs."""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np


ALPHAS = [0.001, 0.005, 0.01, 0.02, 0.03, 0.05, 0.1]
COVID_LOSSES = {
    "lambda0": [
        [0, 8, 8, 6],
        [10, 0, 7, 3],
        [10, 7, 0, 2],
        [9, 6, 6, 0],
    ],
    "lambda1": [
        [0, 8, 8, 6],
        [100, 0, 70, 3],
        [100, 70, 0, 2],
        [90, 60, 60, 0],
    ],
}
SCORE_METHODS = ["LAS", "APS", "SOCOP"]


def configure_official_imports(official_repo: Path) -> None:
    os.environ.setdefault("MPLBACKEND", "Agg")
    resolved = str(official_repo.resolve())
    if resolved not in sys.path:
        sys.path.insert(0, resolved)


def jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def run_covid_seed(
    official_repo: str,
    seed_path: str,
    seed: int,
    variant: str,
) -> dict[str, Any]:
    configure_official_imports(Path(official_repo))
    from evaluation import evaluate_seed

    started = time.perf_counter()
    with np.load(seed_path) as data:
        result = evaluate_seed(
            data["cal_probs"],
            data["cal_labels"],
            data["test_probs"],
            data["test_labels"],
            COVID_LOSSES[variant],
            ALPHAS,
            [1, 2, 3],
        )
        shapes = {
            key: list(data[key].shape)
            for key in ("cal_probs", "cal_labels", "test_probs", "test_labels")
        }
    return {
        "dataset": "covid",
        "variant": variant,
        "seed": seed,
        "source": str(Path(seed_path).resolve()),
        "input_shapes": shapes,
        "elapsed_seconds": time.perf_counter() - started,
        "result": result,
    }


def run_bdd_split(
    official_repo: str,
    npz_path: str,
    seed: int,
    phi_frac: float,
    cal_frac: float,
    eps: float,
) -> dict[str, Any]:
    configure_official_imports(Path(official_repo))
    from evaluation import evaluate_seed
    from evaluation_bdd import (
        apply_isotonic,
        build_fx,
        build_loss_matrix,
        encode_labels,
        fit_isotonic_models,
        make_splits,
        top1_accuracy,
    )

    started = time.perf_counter()
    with np.load(npz_path) as data:
        scores = data["scores"]
        true_bits = data["true_bits"]
    (
        _,
        scores_phi,
        bits_phi,
        scores_cal,
        bits_cal,
        scores_test,
        bits_test,
    ) = next(make_splits(scores, true_bits, phi_frac, cal_frac, [seed]))
    models = fit_isotonic_models(scores_phi, bits_phi, eps=eps)
    cal_probs = build_fx(apply_isotonic(models, scores_cal, eps=eps)).astype(
        np.float32
    )
    test_probs = build_fx(apply_isotonic(models, scores_test, eps=eps)).astype(
        np.float32
    )
    cal_labels = encode_labels(bits_cal)
    test_labels = encode_labels(bits_test)
    result = evaluate_seed(
        cal_probs,
        cal_labels,
        test_probs,
        test_labels,
        build_loss_matrix(),
        ALPHAS,
        [1, 2, 3, 4, 5, 6, 7],
        bad_action_threshold=60.0,
    )
    return {
        "dataset": "bdd",
        "variant": "default",
        "seed": seed,
        "source": str(Path(npz_path).resolve()),
        "input_shapes": {
            "phi_scores": list(scores_phi.shape),
            "cal_probs": list(cal_probs.shape),
            "test_probs": list(test_probs.shape),
        },
        "top1": {
            "calibration": top1_accuracy(cal_probs, cal_labels),
            "test": top1_accuracy(test_probs, test_labels),
        },
        "elapsed_seconds": time.perf_counter() - started,
        "result": result,
    }


def mean_se(rows: list[Any]) -> dict[str, Any]:
    values = np.asarray(rows, dtype=float)
    mean = np.mean(values, axis=0)
    if len(values) > 1:
        se = np.std(values, axis=0, ddof=1) / math.sqrt(len(values))
    else:
        se = np.zeros_like(mean)
    return {"mean": jsonable(mean), "se": jsonable(se)}


def aggregate(records: list[dict[str, Any]]) -> dict[str, Any]:
    results = [record["result"] for record in records]
    summary: dict[str, Any] = {
        "dataset": records[0]["dataset"],
        "variant": records[0]["variant"],
        "seeds": [record["seed"] for record in records],
        "alpha_list": ALPHAS,
        "elapsed_seconds": {
            "sum_worker_time": sum(record["elapsed_seconds"] for record in records),
            "mean_per_seed": float(
                np.mean([record["elapsed_seconds"] for record in records])
            ),
        },
        "metrics": {
            "worst_case_risk": {},
            "realized_loss": {},
            "miscoverage": {},
        },
    }
    for metric in ("worst_case_risk", "realized_loss"):
        for method in ("ROCP", "RAC"):
            summary["metrics"][metric][method] = mean_se(
                [row[metric][method] for row in results]
            )
        for scorer in SCORE_METHODS:
            for rule in ("a_ROCP", "a_RAC"):
                key = f"{scorer}_{rule}"
                source = f"{metric}_scores"
                summary["metrics"][metric][key] = mean_se(
                    [row[source][scorer][rule] for row in results]
                )
    for method in ("ROCP", "RAC", *SCORE_METHODS):
        summary["metrics"]["miscoverage"][method] = mean_se(
            [row["miscoverage"][method] for row in results]
        )
    summary["metrics"]["best_response_realized_loss"] = mean_se(
        [row["best_realized_loss"] for row in results]
    )

    critical_key = (
        "critical_bad_action"
        if records[0]["dataset"] == "bdd"
        else "critical_mistake"
    )
    labels = sorted(
        int(label) for label in results[0][critical_key]["rocp"].keys()
    )
    critical = {}
    for method in ("best", "rocp", "rac"):
        critical[method] = {
            str(label): mean_se(
                [
                    row[critical_key][method].get(
                        label, row[critical_key][method].get(str(label))
                    )
                    for row in results
                ]
            )
            for label in labels
        }
    summary["metrics"][critical_key] = critical

    rocp_loss = np.asarray(
        summary["metrics"]["realized_loss"]["ROCP"]["mean"], dtype=float
    )
    rac_loss = np.asarray(
        summary["metrics"]["realized_loss"]["RAC"]["mean"], dtype=float
    )
    rocp_risk = np.asarray(
        summary["metrics"]["worst_case_risk"]["ROCP"]["mean"], dtype=float
    )
    rac_risk = np.asarray(
        summary["metrics"]["worst_case_risk"]["RAC"]["mean"], dtype=float
    )
    summary["rocp_minus_rac"] = {
        "realized_loss": jsonable(rocp_loss - rac_loss),
        "worst_case_risk": jsonable(rocp_risk - rac_risk),
    }
    return summary


def load_record(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def git_revision(repo: Path) -> str:
    completed = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--official-repo", type=Path, required=True)
    parser.add_argument("--dataset", choices=("covid", "bdd"), required=True)
    parser.add_argument(
        "--variant", choices=("lambda0", "lambda1"), default="lambda0"
    )
    parser.add_argument("--seeds", type=int, nargs="+")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--output-dir", type=Path, default=Path("results/cached"))
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--phi-frac", type=float, default=0.2)
    parser.add_argument("--cal-frac", type=float, default=0.4)
    parser.add_argument("--eps", type=float, default=1e-3)
    args = parser.parse_args()

    official_repo = args.official_repo.resolve()
    seeds = sorted(set(args.seeds)) if args.seeds else list(range(23, 43))
    variant = args.variant if args.dataset == "covid" else "default"
    run_dir = args.output_dir / args.dataset / variant
    run_dir.mkdir(parents=True, exist_ok=True)

    records: dict[int, dict[str, Any]] = {}
    pending = []
    for seed in seeds:
        output = run_dir / f"seed_{seed}.json"
        if output.exists() and not args.force:
            records[seed] = load_record(output)
        else:
            pending.append(seed)

    started = time.perf_counter()
    with ProcessPoolExecutor(max_workers=max(1, args.workers)) as executor:
        futures = {}
        for seed in pending:
            if args.dataset == "covid":
                source = (
                    official_repo
                    / "results"
                    / "Covid_data"
                    / "probabilities"
                    / f"seed_{seed}.npz"
                )
                future = executor.submit(
                    run_covid_seed,
                    str(official_repo),
                    str(source),
                    seed,
                    args.variant,
                )
            else:
                source = official_repo / "results" / "BDD" / "raw_bits.npz"
                future = executor.submit(
                    run_bdd_split,
                    str(official_repo),
                    str(source),
                    seed,
                    args.phi_frac,
                    args.cal_frac,
                    args.eps,
                )
            futures[future] = seed

        for future in as_completed(futures):
            seed = futures[future]
            record = jsonable(future.result())
            output = run_dir / f"seed_{seed}.json"
            output.write_text(
                json.dumps(record, indent=2) + "\n", encoding="utf-8"
            )
            records[seed] = record
            print(
                f"completed {args.dataset}/{variant} seed {seed} "
                f"in {record['elapsed_seconds']:.1f}s",
                flush=True,
            )

    ordered = [records[seed] for seed in seeds]
    summary = aggregate(ordered)
    summary["elapsed_seconds"]["wall_for_this_invocation"] = (
        time.perf_counter() - started
    )
    summary["official_repo"] = str(official_repo)
    summary["official_git_revision"] = git_revision(official_repo)
    summary_path = run_dir / "summary.json"
    summary_path.write_text(
        json.dumps(jsonable(summary), indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(jsonable(summary), indent=2))


if __name__ == "__main__":
    main()
