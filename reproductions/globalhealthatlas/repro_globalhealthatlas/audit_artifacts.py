from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import struct
import sys
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
OFFICIAL = ROOT / "official"
REPO = OFFICIAL / "repo"
RESULTS = REPO / "Experimental Results"
BENCHMARK = RESULTS / "Benchmark_evaluation_results"
DATASET_RESULTS = RESULTS / "Dataset_evaluation_results"

DIMS = (
    "Accuracy",
    "Reasoning",
    "Completeness",
    "Consensus Alignment",
    "Terminology Norms",
    "Insightfulness",
)

DETAILED_TO_TABLE = {
    "Qwen2_5_7Binstruct_detailed.csv": "qwen2.5_7binstruct",
    "Qwen2_5_72binstruct_detailed.csv": "qwen2.5_72binstruct",
    "deepseek_distill_qwen_32b_detailed.csv": "deepseek-r1-distill-qwen-32b",
    "deepseek_distill_qwen_8b_detailed.csv": "deepSeek_r1_distill_qwen_8B",
    "deepseek_r1_detailed.csv": "deepseek_r1",
    "deepseek_v3_2_detailed.csv": "deepSeek-v3.2",
    "claude_sonnet_thinking_detailed.csv": "claude_sonnet_thinking",
    "Qwen3_8B_detailed.csv": "Qwen3_8B",
    "Qwen3_32B_detailed.csv": "qwen3_32b",
    "gemini_3_flash_preview_thinking_detailed.csv": "gemini-3-flash-preview-thinking",
    "qwq_32B_detailed.csv": "qwq_32B",
    "kimi_k2_thinking_detailed.csv": "kimi-k2-thinking",
    "glm_4_7_detailed.csv": "glm-4.7",
    "chatGPT_detailed.csv": "chatGPT_22243",
    "grok_3_detailed.csv": "grok-3-mini",
    "Qwen3_8B100_detailed.csv": "Qwen3_8b 100%微调",
}

EXPECTED_TRANSFER = [
    ("Qwen-4b base", "MMLU-Pro", 6.762),
    ("Qwen-4b base", "GPQA", 4.948),
    ("Qwen-4b SFT", "MMLU-Pro", 7.043),
    ("Qwen-4b SFT", "GPQA", 5.342),
    ("Qwen-8b base", "MMLU-Pro", 7.139),
    ("Qwen-8b base", "GPQA", 5.342),
    ("Qwen-8b SFT", "MMLU-Pro", 7.405),
    ("Qwen-8b SFT", "GPQA", 5.904),
    ("Qwen-14b base", "MMLU-Pro", 7.426),
    ("Qwen-14b base", "GPQA", 5.277),
    ("Qwen-14b SFT", "MMLU-Pro", 7.309),
    ("Qwen-14b SFT", "GPQA", 5.723),
]

EXPECTED_ROBUSTNESS = {
    "qwen3-8b 原始": {
        "Public_Benchmark1000": 0.8541,
        "paraphrase_score": 0.8391,
        "Public_Benchmark_Noisy_score": 0.7726,
        "cross_lingual_score": 0.8168,
    },
    "qwen3-8b 100%微调": {
        "Public_Benchmark1000": 0.8730,
        "paraphrase_score": 0.8600,
        "Public_Benchmark_Noisy_score": 0.7734,
        "cross_lingual_score": 0.8440,
    },
    "ds-r1-distill-qwen": {
        "Public_Benchmark1000": 0.4555,
        "paraphrase_score": 0.4940,
        "Public_Benchmark_Noisy_score": 0.4000,
        "cross_lingual_score": 0.6152,
    },
    "claude": {
        "Public_Benchmark1000": 0.6469,
        "paraphrase_score": 0.5800,
        "Public_Benchmark_Noisy_score": 0.6458,
        "cross_lingual_score": 0.6589,
    },
    "Qwen3-14B": {
        "Public_Benchmark1000": 0.8330,
        "paraphrase_score": 0.8270,
        "Public_Benchmark_Noisy_score": 0.7710,
        "cross_lingual_score": 0.8190,
    },
    "llama3.1-8B": {
        "Public_Benchmark1000": 0.7089,
        "paraphrase_score": 0.7053,
        "Public_Benchmark_Noisy_score": 0.5000,
        "cross_lingual_score": 0.7309,
    },
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def finite_float(value: str) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def mean(values: list[float]) -> float:
    return sum(values) / len(values)


def safetensors_summary(path: Path) -> dict[str, Any]:
    with path.open("rb") as handle:
        header_size = struct.unpack("<Q", handle.read(8))[0]
        header = json.loads(handle.read(header_size))
    tensors = {key: value for key, value in header.items() if key != "__metadata__"}
    dtypes: dict[str, int] = defaultdict(int)
    parameter_count = 0
    max_end = 0
    for tensor in tensors.values():
        dtypes[tensor["dtype"]] += 1
        parameter_count += math.prod(tensor["shape"])
        max_end = max(max_end, tensor["data_offsets"][1])
    return {
        "bytes": path.stat().st_size,
        "sha256": sha256(path),
        "header_bytes": header_size,
        "tensor_count": len(tensors),
        "parameter_count": parameter_count,
        "dtype_counts": dict(sorted(dtypes.items())),
        "payload_extent_matches_file": 8 + header_size + max_end == path.stat().st_size,
    }


def load_benchmark_2() -> dict[str, dict[str, float]]:
    with (BENCHMARK / "benchmark_2.csv").open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    result: dict[str, dict[str, float]] = {}
    for row in rows:
        if not row["Model"]:
            continue
        result[row["Model"]] = {
            "sc": float(row["SC"]),
            "qa": float(row["QA"]),
            "overall": float(row["Total Score"]),
        }
    return result


def reconstruct_detailed(path: Path) -> dict[str, Any]:
    accum = defaultdict(lambda: {"count": 0, "sum": defaultdict(float), "weight": defaultdict(int)})
    current_label = ""
    with path.open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            if row["label"]:
                current_label = row["label"]
            label = current_label
            count = int(float(row["count"]))
            accum[label]["count"] += count
            for dim in DIMS:
                value = finite_float(row.get(dim, ""))
                if value is not None:
                    accum[label]["sum"][dim] += value * count
                    accum[label]["weight"][dim] += count

    labels: dict[str, dict[str, Any]] = {}
    for label, data in accum.items():
        all_dim_means = {
            dim: data["sum"][dim] / data["weight"][dim]
            for dim in DIMS
            if data["weight"][dim]
        }
        # The released aggregate table marks Reasoning as unavailable ("-")
        # for non-reasoning models. Their detailed exports contain only zeros
        # or a tiny residual mean below 0.01, so reproduce that documented
        # aggregation convention instead of treating the unavailable field as
        # a real zero score.
        dim_means = {
            dim: value
            for dim, value in all_dim_means.items()
            if not (dim == "Reasoning" and value < 0.01)
        }
        labels[label] = {
            "count": data["count"],
            "dimension_means": dim_means,
            "omitted_dimensions": sorted(set(all_dim_means) - set(dim_means)),
            "mean": mean(list(dim_means.values())),
        }
    qa = labels["Question-Answer"]
    sc = labels["Single-Choice"]
    total_count = qa["count"] + sc["count"]
    return {
        "labels": labels,
        "overall": (qa["mean"] * qa["count"] + sc["mean"] * sc["count"]) / total_count,
        "count": total_count,
    }


def benchmark_reconstruction() -> dict[str, Any]:
    expected = load_benchmark_2()
    comparisons = []
    for filename, table_name in DETAILED_TO_TABLE.items():
        reconstructed = reconstruct_detailed(BENCHMARK / filename)
        target = expected[table_name]
        errors = {
            "qa": abs(reconstructed["labels"]["Question-Answer"]["mean"] - target["qa"]),
            "sc": abs(reconstructed["labels"]["Single-Choice"]["mean"] - target["sc"]),
            "overall": abs(reconstructed["overall"] - target["overall"]),
        }
        comparisons.append(
            {
                "file": filename,
                "table_model": table_name,
                "reconstructed": reconstructed,
                "table": target,
                "absolute_errors": errors,
            }
        )
    return {
        "models_compared": len(comparisons),
        "max_score_abs_error": max(
            max(item["absolute_errors"][key] for key in ("qa", "sc", "overall"))
            for item in comparisons
        ),
        "rows_matching_paper_rounding": sum(
            max(item["absolute_errors"].values()) < 5e-4 for item in comparisons
        ),
        "comparisons": comparisons,
    }


def incremental_reconstruction() -> dict[str, Any]:
    path = DATASET_RESULTS / "incremental_experiment.csv"
    with path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.reader(handle))
    results = []
    for row in rows[2:]:
        if not row or not row[0]:
            continue
        qa = mean([float(value) for value in row[1:7]])
        sc = mean([float(value) for value in row[7:13]])
        qa_count, sc_count = int(row[13]), int(row[14])
        overall = (qa * qa_count + sc * sc_count) / (qa_count + sc_count)
        results.append(
            {
                "model": row[0],
                "recomputed": overall,
                "released": float(row[15]),
                "absolute_error": abs(overall - float(row[15])),
            }
        )
    by_family: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in results:
        family = item["model"].split("b", 1)[0]
        by_family[family].append(item)
    return {
        "rows": len(results),
        "max_abs_error": max(item["absolute_error"] for item in results),
        "all_sft_above_base": {
            family: all(item["released"] > values[0]["released"] for item in values[1:])
            for family, values in by_family.items()
        },
        "rows_detail": results,
    }


def transfer_reconstruction() -> dict[str, Any]:
    path = DATASET_RESULTS / "transfer_experiment.csv"
    with path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.reader(handle))[2:]
    comparisons = []
    current_model = ""
    for index, row in enumerate(rows):
        if not row:
            continue
        if row[0]:
            current_model = row[0]
        score = mean([float(value) for value in row[2:8]])
        label, dataset, expected = EXPECTED_TRANSFER[index]
        comparisons.append(
            {
                "released_model_label": current_model,
                "paper_role": label,
                "dataset": dataset,
                "recomputed": score,
                "paper": expected,
                "absolute_error": abs(score - expected),
                "valid_count": int(row[8]),
            }
        )
    return {
        "rows": len(comparisons),
        "max_abs_error_after_paper_rounding": max(item["absolute_error"] for item in comparisons),
        "rows_matching_paper_rounding": sum(
            item["absolute_error"] < 5e-4 for item in comparisons
        ),
        "duplicate_qwen14_base_label": comparisons[-1]["released_model_label"]
        == comparisons[-3]["released_model_label"],
        "comparisons": comparisons,
    }


def robustness_reconstruction() -> dict[str, Any]:
    path = DATASET_RESULTS / "robustness_experiment.csv"
    with path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    current_model = ""
    comparisons = []
    for row in rows:
        if row["Unnamed: 0"]:
            current_model = row["Unnamed: 0"]
        value = float(row["准确率"])
        expected = EXPECTED_ROBUSTNESS[current_model][row["dataset"]]
        comparisons.append(
            {
                "model": current_model,
                "setting": row["dataset"],
                "released": value,
                "paper": expected,
                "absolute_error": abs(value - expected),
                "valid_count": int(row["有效数据"]),
            }
        )
    return {
        "rows": len(comparisons),
        "max_abs_error": max(item["absolute_error"] for item in comparisons),
        "valid_count_range": [
            min(item["valid_count"] for item in comparisons),
            max(item["valid_count"] for item in comparisons),
        ],
        "comparisons": comparisons,
    }


def leakage_reconstruction() -> dict[str, Any]:
    path = DATASET_RESULTS / "data_leakage_experiment.csv"
    with path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    nonzero = [
        {
            "model": row["Model"],
            "prop1": float(row["prop1"]),
            "prop2": finite_float(row["prop2"]),
        }
        for row in rows
        if float(row["prop1"]) > 0
    ]
    return {
        "rows": len(rows),
        "nonzero_rows": nonzero,
        "max_prop1": max(float(row["prop1"]) for row in rows),
        "all_10gram_zero": all(
            float(row["prop1"]) == 0 for row in rows if row["Model"].endswith("-10")
        ),
    }


def lightweight_code_smoke() -> dict[str, Any]:
    sys.path.insert(0, str(REPO))
    from src.core.prompt_builder import build_prompt
    from src.utils.data_handler import load_input_data, save_output_data

    sample = {
        "domain": "Vaccination and Immunization",
        "label": "Question-Answer",
        "question": "What is population immunity?",
        "answer": "Reference",
        "complexCOT": "Reference reasoning",
        "llm_complexCOT": "Candidate reasoning",
        "llm_answer": "Candidate answer",
    }
    prompt = build_prompt(sample)
    placeholders_remaining = [token for token in ("{{domain}}", "{{question}}", "{{answer}}") if token in prompt]
    with tempfile.TemporaryDirectory() as temp_dir:
        output = Path(temp_dir) / "roundtrip.json"
        save_output_data([sample], str(output))
        roundtrip = load_input_data(str(output))
        stale_temp_exists = output.with_suffix(output.suffix + ".tmp").exists()
    return {
        "prompt_length": len(prompt),
        "required_values_present": all(str(value) in prompt for value in sample.values()),
        "placeholders_remaining": placeholders_remaining,
        "atomic_json_roundtrip": roundtrip == [sample],
        "stale_temp_exists": stale_temp_exists,
    }


def training_metadata() -> dict[str, Any]:
    evaluator_config = json.loads((OFFICIAL / "evaluator" / "adapter_config.json").read_text())
    model_config = json.loads((OFFICIAL / "public_model" / "adapter_config.json").read_text())
    evaluator_results = json.loads((OFFICIAL / "evaluator" / "all_results.json").read_text())
    model_results = json.loads((OFFICIAL / "public_model" / "all_results.json").read_text())
    training_script = (REPO / "training" / "train_lora.sh").read_text(encoding="utf-8")
    return {
        "evaluator_adapter_config": evaluator_config,
        "public_model_adapter_config": model_config,
        "evaluator_train_results": evaluator_results,
        "public_model_train_results": model_results,
        "released_script": {
            "mentions_psychology_dataset": 'DATASET_NAME="distill_psychology-10k-r1"' in training_script,
            "rank_16": "LORA_RANK=16" in training_script,
            "epochs_1": "NUM_TRAIN_EPOCHS=1.0" in training_script,
            "max_samples_5400": "MAX_SAMPLES=5400" in training_script,
        },
        "paper_public_model_config_matches_adapter": {
            "rank_8": model_config["r"] == 8,
            "alpha_16": model_config["lora_alpha"] == 16,
            "dropout_0": model_config["lora_dropout"] == 0.0,
            "seven_target_modules": len(model_config["target_modules"]) == 7,
            "two_epochs": abs(model_results["epoch"] - 2.0) < 1e-9,
        },
    }


def build_audit() -> dict[str, Any]:
    csv_files = sorted(RESULTS.rglob("*.csv"))
    adapters = {
        "evaluator": safetensors_summary(OFFICIAL / "evaluator" / "adapter_model.safetensors"),
        "public_model": safetensors_summary(OFFICIAL / "public_model" / "adapter_model.safetensors"),
    }
    adapters["evaluator"]["hub_lfs_sha256"] = (
        "60556b253446fe762431b63054f3964277732f651af311e6059327f22a0b991c"
    )
    adapters["public_model"]["hub_lfs_sha256"] = (
        "056422c0659ccf527a1ff10c01a97d275380276d0cfe1afd78355d1076e9134d"
    )
    for item in adapters.values():
        item["hub_checksum_matches"] = item["sha256"] == item["hub_lfs_sha256"]

    return {
        "prepared_score": 6,
        "score_denominator": 12,
        "claims": [
            {"id": "C1", "score": 1, "verdict": "PARTIAL_TEST_SET_COVERAGE_CORPUS_ABSENT"},
            {"id": "C2", "score": 1, "verdict": "PARTIAL_AGGREGATES_ONLY_RAW_QC_ROWS_ABSENT"},
            {"id": "C3", "score": 0, "verdict": "UNSUPPORTED_VALID_ADAPTER_BUT_GOLD_AND_STABILITY_ROWS_ABSENT"},
            {"id": "C4", "score": 2, "verdict": "SUPPORTED_DETAILED_CSV_RECONSTRUCTION"},
            {"id": "C5", "score": 1, "verdict": "PARTIAL_ARITHMETIC_RECONSTRUCTION_WITH_TRANSFER_MISMATCH"},
            {"id": "C6", "score": 1, "verdict": "PARTIAL_CACHED_TABLES_EXACT_RAW_INPUTS_ABSENT"},
        ],
        "paper": {
            "arxiv": "2602.00491v3",
            "pdf_sha256": sha256(OFFICIAL / "paper.pdf"),
            "source_sha256": sha256(OFFICIAL / "source.tar.gz"),
        },
        "artifact_availability": {
            "github_revision": "23edda8517ed95e3a3db4fda0fc0fc53546532cb",
            "result_csv_count": len(csv_files),
            "result_csv_bytes": sum(path.stat().st_size for path in csv_files),
            "exact_name_hf_dataset_search_result_count": len(
                json.loads((OFFICIAL / "hf_dataset_search.json").read_text())
            ),
            "raw_corpus_present_in_repo": any(
                path.suffix.lower() in {".json", ".jsonl", ".parquet", ".arrow"}
                and "result" not in path.name.lower()
                for path in REPO.rglob("*")
                if path.is_file()
            ),
            "expert_gold_or_repeated_run_matrices_present": False,
        },
        "adapters": adapters,
        "benchmark_reconstruction": benchmark_reconstruction(),
        "incremental_reconstruction": incremental_reconstruction(),
        "transfer_reconstruction": transfer_reconstruction(),
        "robustness_reconstruction": robustness_reconstruction(),
        "leakage_reconstruction": leakage_reconstruction(),
        "training_metadata": training_metadata(),
        "lightweight_code_smoke": lightweight_code_smoke(),
        "limits": [
            "The 280,210 row corpus and 35,500 source-document inventory were not released in the linked artifacts.",
            "The 14,010 expert quality-audit rows were not released.",
            "The 100 expert/evaluator agreement rows and ten-run stability matrices were not released.",
            "Cached result tables lack raw prompts, predictions, perturbations, and generation seeds.",
            "Full inference needs Qwen3-8B plus CUDA/vLLM and was not attempted on this CPU host.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parent / "evidence" / "artifact_audit.json",
    )
    args = parser.parse_args()
    audit = build_audit()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({
        "output": str(args.output),
        "benchmark_models": audit["benchmark_reconstruction"]["models_compared"],
        "benchmark_max_error": audit["benchmark_reconstruction"]["max_score_abs_error"],
        "incremental_max_error": audit["incremental_reconstruction"]["max_abs_error"],
        "transfer_max_error": audit["transfer_reconstruction"]["max_abs_error_after_paper_rounding"],
        "robustness_max_error": audit["robustness_reconstruction"]["max_abs_error"],
        "adapter_checksums": {
            key: value["hub_checksum_matches"] for key, value in audit["adapters"].items()
        },
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
