# Claim 4: CRC-CODEX quantification


---
<!-- trackio-cell
{"type": "markdown", "id": "cell_0de32efcad51", "created_at": "2026-07-29T17:06:36+00:00", "title": "Claim 4: CRC-CODEX quantification"}
-->
**Verdict - INCONCLUSIVE, MISSING INPUTS AND OUTPUTS.**

The released evaluator implements the five metric families, but the official
dataset does not include the CRC-CODEX evaluation inputs, generated
quantification outputs, or complete ground-truth bundle needed to recompute
the reported 0.953/0.902/0.727/0.368/0.513 values.


---
<!-- trackio-cell
{"type": "code", "id": "cell_b41f7101fb2b", "created_at": "2026-07-29T17:06:36+00:00", "title": "C4 verdict record", "language": "python"}
-->
````python title=audit_artifacts.py
#!/usr/bin/env python3
"""Independent, network-free audit of the published SP-Mind artifacts.

This script deliberately does not import SP-Mind, call an LLM, invoke the
author's benchmark generator, or copy counts from the paper. It derives the
benchmark and tool inventories from the frozen JSONL/Python artifacts.
"""

from __future__ import annotations

import argparse
import ast
import collections
import hashlib
import json
import re
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
CANDIDATE_DIR = SCRIPT_DIR.parent
REQUIRED_KEYS = {
    "id",
    "query",
    "tier",
    "category",
    "stages",
    "num_stages",
    "context_specific",
}
EXPECTED_TIERS = {"basic", "intermediate", "advanced", "challenging"}
EXPECTED_STAGE_MODULES = {
    "illumination": "basic_illumination",
    "registration": "registration",
    "background_subtraction": "background_subtraction",
    "dearray": "unetcoreograph",
    "probability_mapping": "segmentation_unmicst",
    "segmentation": "segmentation_s3segmenter",
    "quantification": "quantification",
    "clustering": "clustering",
}
EXPECTED_PLACEHOLDERS = {
    "input_dir",
    "output_dir",
    "markers_csv",
    "illum_profiles_dir",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
        if not isinstance(value, dict):
            raise ValueError(f"{path}:{line_number}: record is not an object")
        records.append(value)
    return records


def assigned_literal(tree: ast.Module, variable: str) -> Any:
    for node in tree.body:
        value = None
        if isinstance(node, ast.Assign):
            if any(isinstance(target, ast.Name) and target.id == variable for target in node.targets):
                value = node.value
        elif isinstance(node, ast.AnnAssign):
            if isinstance(node.target, ast.Name) and node.target.id == variable:
                value = node.value
        if value is not None:
            return ast.literal_eval(value)
    raise ValueError(f"no literal assignment to {variable!r}")


def function_names(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    return {
        node.name
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }


def audit_benchmark(path: Path) -> dict[str, Any]:
    records = read_jsonl(path)
    errors: list[str] = []
    warnings: list[str] = []

    for index, record in enumerate(records, start=1):
        keys = set(record)
        if keys != REQUIRED_KEYS:
            errors.append(
                f"record {index}: schema keys differ; "
                f"missing={sorted(REQUIRED_KEYS - keys)}, extra={sorted(keys - REQUIRED_KEYS)}"
            )
        if record.get("tier") not in EXPECTED_TIERS:
            errors.append(f"record {index}: unknown tier {record.get('tier')!r}")
        stages = record.get("stages")
        if not isinstance(stages, list) or not all(
            isinstance(stage, str) for stage in stages
        ):
            errors.append(f"record {index}: stages is not a list of strings")
        elif record.get("num_stages") != len(stages):
            errors.append(
                f"record {index}: num_stages={record.get('num_stages')} "
                f"but len(stages)={len(stages)}"
            )
        if not isinstance(record.get("context_specific"), bool):
            errors.append(f"record {index}: context_specific is not Boolean")

    ids = [record.get("id") for record in records]
    queries = [record.get("query") for record in records]
    normalized_queries = [
        re.sub(r"\s+", " ", str(query)).strip().casefold() for query in queries
    ]
    duplicate_ids = sorted(
        value for value, count in collections.Counter(ids).items() if count > 1
    )
    duplicate_queries = sorted(
        value
        for value, count in collections.Counter(normalized_queries).items()
        if count > 1
    )
    if duplicate_ids:
        errors.append(f"duplicate IDs: {duplicate_ids}")
    if duplicate_queries:
        errors.append(f"duplicate normalized queries: {duplicate_queries}")

    tier_counts = collections.Counter(record["tier"] for record in records)
    category_counts = collections.Counter(record["category"] for record in records)
    stage_counts = collections.Counter(
        stage for record in records for stage in record["stages"]
    )
    num_stage_counts = collections.Counter(record["num_stages"] for record in records)
    placeholder_counts = collections.Counter(
        placeholder
        for record in records
        for placeholder in re.findall(r"\{([^{}]+)\}", record["query"])
    )
    no_placeholder_ids = [
        record["id"]
        for record in records
        if not re.search(r"\{[^{}]+\}", record["query"])
    ]
    unknown_placeholders = sorted(set(placeholder_counts) - EXPECTED_PLACEHOLDERS)
    if no_placeholder_ids:
        warnings.append(f"queries without path placeholders: {no_placeholder_ids}")
    if unknown_placeholders:
        warnings.append(f"unrecognized placeholders: {unknown_placeholders}")

    tier_stage_rule_violations = []
    expected_lengths = {"basic": {1}, "intermediate": {2}, "advanced": {3}}
    for record in records:
        allowed = expected_lengths.get(record["tier"])
        if allowed is not None and record["num_stages"] not in allowed:
            tier_stage_rule_violations.append(record["id"])
        if record["tier"] == "challenging" and record["num_stages"] <= 3:
            tier_stage_rule_violations.append(record["id"])
    if tier_stage_rule_violations:
        errors.append(
            f"tier/stage-count rule violations: {sorted(set(tier_stage_rule_violations))}"
        )

    unusual_categories = sorted(
        category for category in category_counts if category != category.upper()
    )
    if unusual_categories:
        warnings.append(
            "category labels are not consistently uppercase: "
            + ", ".join(unusual_categories)
        )

    return {
        "path": path.name,
        "sha256": sha256(path),
        "record_count": len(records),
        "tier_counts": dict(sorted(tier_counts.items())),
        "category_count": len(category_counts),
        "category_counts": dict(sorted(category_counts.items())),
        "stage_count": len(stage_counts),
        "stage_invocation_count": sum(stage_counts.values()),
        "stage_counts": dict(sorted(stage_counts.items())),
        "num_stages_counts": {
            str(key): value for key, value in sorted(num_stage_counts.items())
        },
        "context_specific_counts": {
            str(key).lower(): value
            for key, value in sorted(
                collections.Counter(
                    record["context_specific"] for record in records
                ).items()
            )
        },
        "placeholder_counts": dict(sorted(placeholder_counts.items())),
        "unique_ids": len(set(ids)),
        "unique_normalized_queries": len(set(normalized_queries)),
        "errors": errors,
        "warnings": warnings,
        "passed": not errors,
    }


def audit_tools(code_root: Path) -> dict[str, Any]:
    description_root = code_root / "spmind" / "tool" / "tool_description"
    tool_root = code_root / "spmind" / "tool"
    utils_path = code_root / "spmind" / "utils.py"
    errors: list[str] = []

    utils_tree = ast.parse(utils_path.read_text(encoding="utf-8"), filename=str(utils_path))
    fields: list[str] | None = None
    for node in ast.walk(utils_tree):
        if isinstance(node, ast.FunctionDef) and node.name == "read_module2api":
            for child in node.body:
                if isinstance(child, ast.Assign) and any(
                    isinstance(target, ast.Name) and target.id == "fields"
                    for target in child.targets
                ):
                    fields = ast.literal_eval(child.value)
    if fields is None:
        raise ValueError("could not extract read_module2api fields")

    tools: list[dict[str, str]] = []
    for module in fields:
        description_path = description_root / f"{module}.py"
        implementation_path = tool_root / f"{module}.py"
        tree = ast.parse(
            description_path.read_text(encoding="utf-8"),
            filename=str(description_path),
        )
        descriptions = assigned_literal(tree, "description")
        implementations = function_names(implementation_path)
        for description in descriptions:
            name = description["name"]
            tools.append({"module": module, "name": name})
            if name not in implementations:
                errors.append(
                    f"description {module}.{name} has no top-level implementation"
                )

    names = [tool["name"] for tool in tools]
    duplicates = sorted(
        name for name, count in collections.Counter(names).items() if count > 1
    )
    if duplicates:
        errors.append(f"duplicate registered tool names: {duplicates}")

    benchmark_modules_missing = sorted(
        set(EXPECTED_STAGE_MODULES.values()) - set(fields)
    )
    if benchmark_modules_missing:
        errors.append(
            "benchmark-stage modules missing from registry: "
            + ", ".join(benchmark_modules_missing)
        )

    skill_paths = sorted(
        path.relative_to(code_root).as_posix()
        for path in (code_root / "spmind" / "skills").rglob("*.md")
    )
    agent_source = (code_root / "spmind" / "agent" / "agent.py").read_text(
        encoding="utf-8"
    )
    architecture_evidence = {
        "claude_agent_options": "ClaudeAgentOptions" in agent_source,
        "sdk_query_stream": bool(re.search(r"async\s+for\s+message\s+in\s+query\(", agent_source)),
        "system_prompt_requests_stepwise_tool_execution": all(
            token in agent_source
            for token in (
                "Analyze the user's biomedical task",
                "Explore and understand the data first",
                "Write and execute Python code using SP-Mind tools",
            )
        ),
        "explicit_locally_implemented_react_loop": False,
    }

    return {
        "registry_modules": fields,
        "registered_tool_count": len(tools),
        "registered_tools": tools,
        "unique_tool_names": len(set(names)),
        "all_descriptions_have_implementations": not any(
            "no top-level implementation" in error for error in errors
        ),
        "benchmark_stage_to_module": EXPECTED_STAGE_MODULES,
        "benchmark_stage_modules_present": not benchmark_modules_missing,
        "skill_document_count": len(skill_paths),
        "skill_documents": skill_paths,
        "architecture_evidence": architecture_evidence,
        "errors": errors,
        "passed": not errors,
    }


def audit_source(source_root: Path) -> dict[str, Any]:
    experiments = source_root / "resources" / "sections" / "experiments.tex"
    spbench = source_root / "resources" / "sections" / "spbench.tex"
    text = experiments.read_text(encoding="utf-8") + "\n" + spbench.read_text(
        encoding="utf-8"
    )
    tokens = {
        "spbench_overall_68_9": r"\textbf{68.9}" in text,
        "crc_pearson_0_953": r"\textbf{0.953" in text,
        "annotation_average_0_681": r"\textbf{0.681}" in text,
        "spbench_102": "102" in text,
        "spbench_18_categories": "18 categories" in text,
        "spbench_8_stages": "8 stages" in text,
    }
    return {
        "experiments_tex_sha256": sha256(experiments),
        "spbench_tex_sha256": sha256(spbench),
        "published_value_tokens": tokens,
        "note": "Source concordance only; this is not an experimental rerun.",
        "passed": all(tokens.values()),
    }


def render_markdown(result: dict[str, Any]) -> str:
    benchmark = result["benchmark"]
    tools = result["tools"]
    source = result["source"]
    overall = result["passed"]
    lines = [
        "# SP-Mind independent artifact audit",
        "",
        f"Overall deterministic audit: **{'PASS' if overall else 'FAIL'}**",
        "",
        "## SP-Bench manifest",
        "",
        f"- Records: {benchmark['record_count']}",
        f"- Tiers: `{json.dumps(benchmark['tier_counts'], sort_keys=True)}`",
        f"- Categories: {benchmark['category_count']}",
        f"- Stages: {benchmark['stage_count']}",
        f"- Total stage invocations: {benchmark['stage_invocation_count']}",
        f"- Unique IDs / queries: {benchmark['unique_ids']} / "
        f"{benchmark['unique_normalized_queries']}",
        f"- Result: **{'PASS' if benchmark['passed'] else 'FAIL'}**",
        "",
        "## Agent/tool artifact",
        "",
        f"- Registered tool descriptions: {tools['registered_tool_count']}",
        f"- Unique tool names: {tools['unique_tool_names']}",
        f"- Skill documents: {tools['skill_document_count']}",
        f"- All eight benchmark stage modules present: "
        f"`{tools['benchmark_stage_modules_present']}`",
        f"- Description-to-implementation checks: "
        f"`{tools['all_descriptions_have_implementations']}`",
        f"- Result: **{'PASS' if tools['passed'] else 'FAIL'}**",
        "",
        "The local code delegates the iterative agent loop to Claude Agent SDK. "
        "It contains a stepwise tool-use system prompt, but not a separately "
        "implemented local ReAct state machine.",
        "",
        "## Paper-source concordance",
        "",
        f"- Published-value tokens found: `{all(source['published_value_tokens'].values())}`",
        "- This confirms the frozen source matches the reported values; it does "
        "not reproduce those values.",
        "",
    ]
    warnings = benchmark["warnings"]
    if warnings:
        lines.extend(["## Warnings", ""])
        lines.extend(f"- {warning}" for warning in warnings)
        lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--code-root",
        type=Path,
        default=CANDIDATE_DIR / "official" / "code",
    )
    parser.add_argument(
        "--source-root",
        type=Path,
        default=CANDIDATE_DIR / "official" / "arxiv-source",
    )
    parser.add_argument(
        "--json-out",
        type=Path,
        default=SCRIPT_DIR / "evidence" / "audit_results.json",
    )
    parser.add_argument(
        "--markdown-out",
        type=Path,
        default=SCRIPT_DIR / "evidence" / "audit_results.md",
    )
    args = parser.parse_args()

    result = {
        "audit_version": 1,
        "provenance": {
            "official_commit": "d5b889c3649fbdfd1489e5b2f60fc52a0d3ddbc6",
            "official_file_sha256": {
                "spmind/agent/agent.py": sha256(
                    args.code_root / "spmind" / "agent" / "agent.py"
                ),
                "spmind/utils.py": sha256(
                    args.code_root / "spmind" / "utils.py"
                ),
                "experiments/run_annotation_eval.py": sha256(
                    args.code_root / "experiments" / "run_annotation_eval.py"
                ),
            },
        },
        "benchmark": audit_benchmark(
            args.code_root / "benchmark" / "sp_bench.jsonl"
        ),
        "tools": audit_tools(args.code_root),
        "source": audit_source(args.source_root),
    }
    result["passed"] = all(
        result[section]["passed"] for section in ("benchmark", "tools", "source")
    )

    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.markdown_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    args.markdown_out.write_text(render_markdown(result), encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

````


````output
{
  "id": "C4",
  "score": 0,
  "title": "CRC-CODEX quantification",
  "verdict": "INCONCLUSIVE_MISSING_INPUTS_AND_OUTPUTS"
}
````
