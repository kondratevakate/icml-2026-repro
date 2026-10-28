"""Deterministic audit of the official released Table 2 and Table 4 CSVs."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import statistics
from collections import defaultdict
from pathlib import Path


OFFICIAL_COMMIT = "39a99dcad92205a15d93f2c5fec40c76540abf1c"
EXPECTED_HASHES = {
    "classifier_results.csv": "070257695f54bd4f2c0d2a8911a765ad49582dc776ed6b235723d4cd43a1230b",
    "classification_results.csv": "e6261f6f8c26b5af98bf0b6297e38460829f46086e58d9773d7e415e55871b83",
}


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def table2(rows: list[dict[str, str]]) -> dict:
    metrics = ("ERT_L1_miscoverage", "ERT_brier_score", "ERT_logloss")
    maxima: dict[tuple[str, int], dict[str, float]] = {}
    cleaned = []
    for row in rows:
        item = dict(row)
        for metric in metrics:
            value = float(item[metric]) if item[metric] else 0.0
            item[metric] = max(0.0, value)
        item["experiment"] = int(item["experiment"])
        cleaned.append(item)
        key = (item["dataset"], item["experiment"])
        maxima.setdefault(key, {metric: 0.0 for metric in metrics})
        for metric in metrics:
            maxima[key][metric] = max(maxima[key][metric], item[metric])

    relative = defaultdict(lambda: defaultdict(list))
    for item in cleaned:
        key = (item["dataset"], item["experiment"])
        for metric in metrics:
            denominator = maxima[key][metric]
            value = 0.0 if denominator == 0.0 else 100.0 * item[metric] / denominator
            relative[(item["method"], item["experiment"])][metric].append(value)

    by_method = defaultdict(lambda: defaultdict(list))
    for (method, _experiment), values in relative.items():
        for metric in metrics:
            by_method[method][metric].append(statistics.mean(values[metric]))

    summary = {
        method: {
            metric: statistics.mean(experiment_means)
            for metric, experiment_means in metric_values.items()
        }
        for method, metric_values in by_method.items()
    }
    lightgbm = summary["CheapBetterLGBMClassifier"]["ERT_L1_miscoverage"]
    partition = summary["PartitionWise"]["ERT_L1_miscoverage"]
    return {
        "rows": len(rows),
        "datasets": len({row["dataset"] for row in rows}),
        "experiments": len({int(row["experiment"]) for row in rows}),
        "methods": len({row["method"] for row in rows}),
        "lightgbm_relative_l1_power_percent": lightgbm,
        "partitionwise_relative_l1_power_percent": partition,
        "power_ratio": lightgbm / partition,
        "matches_official_notebook": (
            abs(lightgbm - 68.929005) < 1e-5
            and abs(partition - 33.670548) < 1e-5
        ),
    }


def table4(rows: list[dict[str, str]]) -> dict:
    groups = defaultdict(list)
    maximum_additive_residual = 0.0
    for row in rows:
        total = float(row["ERT_logloss"])
        under = float(row["ERT_underconfident_logloss"])
        over = float(row["ERT_overconfident_logloss"])
        maximum_additive_residual = max(
            maximum_additive_residual, abs(total - under - over)
        )
        groups[(row["dataset"], row["method"])].append((under, over))

    means = {}
    divergent_groups = 0
    for (dataset, method), values in sorted(groups.items()):
        under_mean = statistics.mean(item[0] for item in values)
        over_mean = statistics.mean(item[1] for item in values)
        difference = abs(under_mean - over_mean)
        if difference > 0.01:
            divergent_groups += 1
        means[f"{dataset}/{method}"] = {
            "kl_under_mean": under_mean,
            "kl_over_mean": over_mean,
            "absolute_difference": difference,
            "repetitions": len(values),
        }

    return {
        "rows": len(rows),
        "datasets": sorted({row["dataset"] for row in rows}),
        "methods": sorted({row["method"] for row in rows}),
        "groups": means,
        "groups_with_absolute_difference_above_0_01": divergent_groups,
        "maximum_additive_identity_residual": maximum_additive_residual,
    }


def run(input_dir: Path) -> dict:
    classifier_path = input_dir / "classifier_results.csv"
    classification_path = input_dir / "classification_results.csv"
    hashes = {
        classifier_path.name: file_hash(classifier_path),
        classification_path.name: file_hash(classification_path),
    }
    result2 = table2(read_rows(classifier_path))
    result4 = table4(read_rows(classification_path))
    passed = (
        hashes == EXPECTED_HASHES
        and result2["matches_official_notebook"]
        and result2["power_ratio"] > 2.0
        and result4["rows"] == 80
        and len(result4["datasets"]) == 4
        and result4["groups_with_absolute_difference_above_0_01"] == 8
        and result4["maximum_additive_identity_residual"] < 1e-6
    )
    return {
        "status": "passed" if passed else "failed",
        "official_commit": OFFICIAL_COMMIT,
        "sha256": hashes,
        "table2_classifier_power": result2,
        "table4_classification_decomposition": result4,
        "interpretation": (
            "The official released rows independently reproduce the notebook's "
            "LightGBM versus PartitionWise relative-power aggregation and show "
            "distinct directional KL components in every dataset/method group."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, default=Path("official_results"))
    parser.add_argument("--output", type=Path, default=Path("results/released_results.json"))
    args = parser.parse_args()
    result = run(args.input_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    if result["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
