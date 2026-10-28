#!/usr/bin/env python3
"""Streaming audit of the published SP-Mind annotation ground truth."""

from __future__ import annotations

import argparse
import collections
import csv
import hashlib
import json
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
CANDIDATE_DIR = SCRIPT_DIR.parent


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def audit_file(path: Path) -> dict:
    cluster_labels: dict[str, collections.Counter[str]] = {}
    annotations: collections.Counter[str] = collections.Counter()
    cell_ids: set[str] = set()
    duplicate_cell_ids = 0
    rows = 0

    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        columns = reader.fieldnames or []
        expected = {"cellLabel", "cluster", "Annotation"}
        missing = sorted(expected - set(columns))
        for record in reader:
            rows += 1
            cell_id = record.get("cellLabel", "")
            if cell_id in cell_ids:
                duplicate_cell_ids += 1
            else:
                cell_ids.add(cell_id)
            cluster = record.get("cluster", "")
            annotation = record.get("Annotation", "")
            cluster_labels.setdefault(cluster, collections.Counter())[annotation] += 1
            annotations[annotation] += 1

    mixed = {
        cluster: dict(sorted(labels.items()))
        for cluster, labels in cluster_labels.items()
        if len(labels) > 1
    }
    return {
        "file": path.name,
        "bytes": path.stat().st_size,
        "sha256": sha256(path),
        "columns": columns,
        "missing_required_columns": missing,
        "rows": rows,
        "unique_cell_ids": len(cell_ids),
        "duplicate_cell_id_rows": duplicate_cell_ids,
        "cluster_count": len(cluster_labels),
        "annotation_label_count": len(annotations),
        "annotation_counts": dict(sorted(annotations.items())),
        "mixed_ground_truth_cluster_count": len(mixed),
        "mixed_ground_truth_clusters": mixed,
        "passed_schema": not missing,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=Path,
        default=CANDIDATE_DIR / "official" / "data-minimal" / "annotation" / "gt",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=SCRIPT_DIR / "evidence" / "annotation_ground_truth_audit.json",
    )
    args = parser.parse_args()

    files = [audit_file(path) for path in sorted(args.root.glob("*.csv"))]
    result = {
        "file_count": len(files),
        "total_rows": sum(item["rows"] for item in files),
        "all_schemas_pass": all(item["passed_schema"] for item in files),
        "files": files,
        "interpretation": (
            "Ground truth is available and structurally usable. Agent prediction "
            "files are still required to reproduce the reported CyteOnto scores."
        ),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["all_schemas_pass"] and files else 1


if __name__ == "__main__":
    raise SystemExit(main())
