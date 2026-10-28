#!/usr/bin/env python3
"""Independent static, schema, and dependency-light state-machine audit."""

from __future__ import annotations

import hashlib
import importlib
import json
import re
import sys
import tempfile
import types
from pathlib import Path
from typing import Any

import pandas as pd


HERE = Path(__file__).resolve().parent
CANDIDATE = HERE.parent
CODE = CANDIDATE / "official" / "code"
TEX = CANDIDATE / "official" / "arxiv-source" / "example_paper.tex"
EVIDENCE = HERE / "evidence" / "artifact_audit.json"
DATA = CODE / "data" / "example" / "data_case"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def audit_release_data() -> dict[str, Any]:
    sft_path = DATA / "biomedparser-SFT.json"
    rl_path = DATA / "biomedparser-RL.json"
    train_path = DATA / "train" / "train.parquet"
    test_path = DATA / "train" / "test.parquet"
    sft = json.loads(sft_path.read_text(encoding="utf-8"))
    rl = json.loads(rl_path.read_text(encoding="utf-8"))
    train = pd.read_parquet(train_path)
    test = pd.read_parquet(test_path)

    referenced: list[str] = []
    for row in sft:
        referenced.extend(row.get("images", []))
    for row in rl:
        referenced.extend(row.get("image_path", []))
        mask = row.get("mask_path", [])
        referenced.extend(mask if isinstance(mask, list) else [mask])

    parquet_paths: list[str] = []
    for frame in (train, test):
        for value in frame["images"]:
            parquet_paths.extend(str(item["image"]) for item in value)

    return {
        "sft_sha256": sha256(sft_path),
        "rl_sha256": sha256(rl_path),
        "train_parquet_sha256": sha256(train_path),
        "test_parquet_sha256": sha256(test_path),
        "sft_records": len(sft),
        "rl_records": len(rl),
        "train_rows": len(train),
        "test_rows": len(test),
        "train_test_parquet_identical": sha256(train_path) == sha256(test_path),
        "duplicate_rl_question_ids": len(rl)
        - len({row.get("question_id") for row in rl}),
        "rl_mask_path_is_list": isinstance(rl[1].get("mask_path"), list),
        "referenced_paths": len(referenced),
        "referenced_paths_existing_as_written": sum(
            (CODE / path).exists() for path in referenced
        ),
        "parquet_paths": parquet_paths,
        "parquet_paths_existing": sum(Path(path).exists() for path in parquet_paths),
        "complete_64k_corpus_present": len(sft) + len(rl) >= 64_000,
    }


def audit_code_spec() -> dict[str, Any]:
    env = (
        CODE
        / "agent_system"
        / "environments"
        / "env_package"
        / "medical_agent"
        / "envs.py"
    ).read_text(encoding="utf-8")
    prompt = (
        CODE / "agent_system" / "environments" / "prompts" / "medical_agent.py"
    ).read_text(encoding="utf-8")
    preprocess = (
        CODE / "examples" / "data_preprocess" / "prepare_Ophiuchus_dataset.py"
    ).read_text(encoding="utf-8")
    atrl = (CODE / "examples" / "grpo_trainer" / "run_ATRL.sh").read_text(
        encoding="utf-8"
    )
    paper = TEX.read_text(encoding="utf-8")
    sft = (CODE / "scripts" / "SFT" / "cold-start" / "train_cold_start.sh").is_file()
    reflect = (
        CODE
        / "scripts"
        / "SFT"
        / "reflective_rejection_fine_tuning"
        / "train_reflective.sh"
    ).is_file()

    kl_values = re.findall(
        r"actor_rollout_ref\.actor\.use_kl_loss=(True|False)", atrl
    )
    checks = {
        "three_tools_declared": all(
            name in prompt for name in ("Zoom-in", "SAM2", "BiomedParse")
        ),
        "three_tool_handlers": all(
            name in env
            for name in ("_handle_zoom_in", "_handle_sam2", "_handle_biomedparse")
        ),
        "format_answer_tool_rewards": all(
            name in env
            for name in (
                "_compute_format_reward",
                "_compute_answer_reward",
                "_compute_tool_reward",
            )
        ),
        "paper_duplicate_tool_early_stop": (
            "duplicates a previously executed one" in paper
        ),
        "release_duplicate_tool_early_stop": bool(
            re.search(r"duplicate|previously executed", env, re.I)
        ),
        "paper_special_tool_call_tokens": r"\texttt{<tool\_call>}" in paper,
        "release_parser_accepts_action_name_only": (
            r"Action\s*:\s*([^\r\n]+)" in env
        ),
        "preprocess_contains_invalid_placeholder_expression": (
            "primary_image = your/path/to(image_paths[0])" in preprocess
        ),
        "cold_start_script_present": sft,
        "reflection_script_present": reflect,
        "atrl_script_present": "env.env_name=medical_agent" in atrl,
        "paper_grpo_without_kl": "without the KL penalty term" in paper,
        "atrl_kl_assignments": kl_values,
        "atrl_effective_kl_loss_enabled": bool(kl_values)
        and kl_values[-1] == "True",
        "atrl_kl_coefficient_001": "actor_rollout_ref.actor.kl_loss_coef=0.01" in atrl,
        "atrl_uses_placeholder_data_path": (
            "/your/path/to/SFT+RL/data_case/train/train.parquet" in atrl
        ),
    }
    return {"checks": checks}


def _worker_class():
    if "gym" not in sys.modules:
        gym = types.ModuleType("gym")
        gym.Env = object
        sys.modules["gym"] = gym
    if "ray" not in sys.modules:
        sys.modules["ray"] = types.ModuleType("ray")
    sys.path.insert(0, str(CODE))
    try:
        module = importlib.import_module(
            "agent_system.environments.env_package.medical_agent.envs"
        )
    finally:
        sys.path.pop(0)
    return module.MedicalAgentWorker


def audit_state_machine() -> dict[str, Any]:
    worker_class = _worker_class()
    image = (
        DATA / "Images" / "case1" / "liver_case_original_image_1.png"
    ).resolve()
    case = {
        "question": "Based on this CT image, is there an abnormality?",
        "question_type": "multiple choice",
        "options": ["A. pancreas tumor", "B. none", "C. liver tumor"],
        "answer": "C",
        "data_type": "vqa",
        "source": "BioMedParse",
        "image_path": [str(image)],
        "max_steps": 6,
    }
    sft_action = (
        "<think>inspect</think> Action: <tool_call>Zoom-in\n"
        '```json\n[{"index":1,"bbox_2d":[100,100,200,200]}]\n```</tool_call>'
    )
    action = (
        "<think>inspect</think> Action: Zoom-in\n"
        '```json\n[{"index":1,"bbox_2d":[100,100,200,200]}]\n```'
    )
    final = "<think>final</think> Action: Answer\n<answer>C</answer>"
    with tempfile.TemporaryDirectory() as output:
        worker = worker_class(0, {"output_dir": output})
        worker.reset(case)
        sft_parsed = worker._parse_tool_invocations(sft_action)
        canonical_parsed = worker._parse_tool_invocations(action)
        _, _, first_reward, first_done, _ = worker.step(action)
        _, _, duplicate_reward, duplicate_done, _ = worker.step(action)
        _, _, final_reward, final_done, final_info = worker.step(final)

    segmentation = dict(case)
    segmentation["data_type"] = "seg"
    segmentation["mask_path"] = [
        str((DATA / "Images" / "case1" / "liver_case_GTmask_image_1.png").resolve())
    ]
    mask_error = None
    with tempfile.TemporaryDirectory() as output:
        try:
            worker_class(0, {"output_dir": output}).reset(segmentation)
        except Exception as exc:  # expected release-schema failure
            mask_error = f"{type(exc).__name__}: {exc}"

    return {
        "sft_style_tool_call_parsed": bool(sft_parsed),
        "canonical_action_tool_call_parsed": bool(canonical_parsed),
        "first_zoom_reward": first_reward,
        "first_zoom_done": first_done,
        "duplicate_zoom_reward": duplicate_reward,
        "duplicate_zoom_done": duplicate_done,
        "duplicate_zoom_early_stopped": duplicate_done,
        "final_reward": final_reward,
        "final_done": final_done,
        "final_won": final_info["won"],
        "released_list_mask_reset_error": mask_error,
    }


def build_evidence() -> dict[str, Any]:
    data = audit_release_data()
    code = audit_code_spec()
    state = audit_state_machine()
    checks = code["checks"]
    c1_core = (
        checks["three_tools_declared"]
        and checks["three_tool_handlers"]
        and checks["format_answer_tool_rewards"]
        and state["canonical_action_tool_call_parsed"]
        and state["final_won"]
    )
    c1_full = (
        c1_core
        and state["sft_style_tool_call_parsed"]
        and state["duplicate_zoom_early_stopped"]
        and state["released_list_mask_reset_error"] is None
    )
    c1 = 2 if c1_full else 1 if c1_core else 0
    c2_core = all(
        checks[key]
        for key in (
            "cold_start_script_present",
            "reflection_script_present",
            "atrl_script_present",
            "format_answer_tool_rewards",
        )
    )
    c2_full = (
        c2_core
        and not checks["preprocess_contains_invalid_placeholder_expression"]
        and not checks["atrl_effective_kl_loss_enabled"]
        and not checks["atrl_uses_placeholder_data_path"]
    )
    c2 = 2 if c2_full else 1 if c2_core else 0
    return {
        "audit_version": 1,
        "official_commit": "256023386c4f3241cdecebc80e00b96c61126128",
        "release_data": data,
        "code_and_spec": code,
        "state_machine": state,
        "claims": {
            "C1": {"points": c1, "verdict": "PARTIAL_TOOL_LOOP_FORMAT_AND_EARLY_STOP_MISMATCH"},
            "C2": {"points": c2, "verdict": "PARTIAL_THREE_STAGES_PRESENT_RUN_CONFIG_MISMATCH"},
            "C3": {"points": 0, "verdict": "INCONCLUSIVE_64K_CORPUS_UNRELEASED"},
            "C4": {"points": 0, "verdict": "INCONCLUSIVE_CHECKPOINT_EVALUATION_UNRELEASED"},
            "C5": {"points": 0, "verdict": "INCONCLUSIVE_CHECKPOINT_ABLATIONS_UNRELEASED"},
        },
        "prepared_score": c1 + c2,
        "score_denominator": 10,
        "passed": c1_core and c2_core,
    }


def main() -> int:
    evidence = build_evidence()
    EVIDENCE.parent.mkdir(parents=True, exist_ok=True)
    EVIDENCE.write_text(
        json.dumps(evidence, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(evidence, indent=2, sort_keys=True))
    return 0 if evidence["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
