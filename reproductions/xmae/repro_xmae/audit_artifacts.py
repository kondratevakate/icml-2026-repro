#!/usr/bin/env python3
"""Independent release audit for xMAE.

This audit deliberately separates paper-level empirical claims from checks that
can be performed with the synthetic HDF5 file and placeholder checkpoint in the
official repository.
"""

from __future__ import annotations

import hashlib
import json
import sys
from collections import OrderedDict
from pathlib import Path
from typing import Any

import h5py
import pandas as pd
import torch
import yaml


HERE = Path(__file__).resolve().parent
CANDIDATE = HERE.parent
CODE = CANDIDATE / "official" / "code"
TEX = CANDIDATE / "official" / "arxiv-source"
EVIDENCE = HERE / "evidence" / "artifact_audit.json"

CHECKPOINT = CODE / "xmae_weights_permute.pth"
H5 = CODE / "pvc_10s_synth.h5"
METADATA = CODE / "pvc_10s_synth_metadata.csv"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def notebook_sources(path: Path) -> str:
    notebook = json.loads(path.read_text(encoding="utf-8"))
    return "\n".join(
        "".join(cell.get("source", []))
        for cell in notebook.get("cells", [])
        if cell.get("cell_type") == "code"
    )


def model_config() -> dict[str, Any]:
    return {
        "sampling_freq": 100,
        "seg_len": 10,
        "source": "ppg+ecg",
        "model_params": {
            "latent_dim": 256,
            "d_model": 256,
            "nhead": 8,
            "depth_ecg": 1,
            "depth_ppg": 2,
            "depth_bridge": 1,
            "stem_ch": 32,
            "dropout": 0.1,
            "use_cross_bridge": True,
            "patch_len": 40,
        },
    }


def load_model_and_checkpoint():
    sys.path.insert(0, str(CODE))
    try:
        from utils.model_arch.xmae import build_model_from_cfg
    finally:
        sys.path.pop(0)

    model = build_model_from_cfg(model_config())
    checkpoint = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    state = OrderedDict(
        (key[len("model.") :], value)
        for key, value in checkpoint.items()
        if key.startswith("model.")
    )
    incompatible = model.load_state_dict(state, strict=True)
    return model, checkpoint, state, incompatible


def audit_checkpoint() -> dict[str, Any]:
    model, checkpoint, state, incompatible = load_model_and_checkpoint()
    running_vars = {
        key: value
        for key, value in checkpoint.items()
        if key.endswith("running_var") and torch.is_tensor(value)
    }
    negative_running_var_values = sum(
        int((value < 0).sum().item()) for value in running_vars.values()
    )
    affected_running_var_tensors = sum(
        bool((value < 0).any().item()) for value in running_vars.values()
    )
    nonfinite_checkpoint_values = sum(
        int((~torch.isfinite(value)).sum().item())
        for value in checkpoint.values()
        if torch.is_tensor(value)
    )

    torch.manual_seed(123)
    ppg = torch.randn(2, 1000)
    ecg = torch.randn(2, 1000)
    n_patches = 25
    ids_keep_ecg = torch.arange(5).view(1, -1).expand(2, -1)
    ids_sorted = torch.cat(
        [
            ids_keep_ecg,
            torch.arange(5, n_patches).view(1, -1).expand(2, -1),
        ],
        dim=1,
    )
    ids_restore_ecg = torch.argsort(ids_sorted, dim=1)

    model.train()
    with torch.no_grad():
        train_outputs = model(
            ppg=ppg,
            ecg=ecg,
            ids_keep_ecg=ids_keep_ecg,
            ids_restore_ecg=ids_restore_ecg,
        )
    train_embedding_finite = bool(
        torch.isfinite(train_outputs["ppg_embedding"]).all().item()
    )
    train_reconstruction_finite = bool(
        torch.isfinite(train_outputs["ecg_reconstructed"]).all().item()
    )

    model.eval()
    with torch.no_grad():
        eval_outputs = model(ppg=ppg, ecg=None)
    eval_embedding = eval_outputs["ppg_embedding"]
    eval_embedding_finite = bool(torch.isfinite(eval_embedding).all().item())

    reverse_bridge_numel = sum(
        parameter.numel() for parameter in model.bridger.ppg2ecg.parameters()
    )
    return {
        "file": CHECKPOINT.name,
        "sha256": sha256(CHECKPOINT),
        "checkpoint_keys": len(checkpoint),
        "state_keys_after_model_prefix_strip": len(state),
        "strict_load_passed": not incompatible.missing_keys
        and not incompatible.unexpected_keys,
        "missing_keys": incompatible.missing_keys,
        "unexpected_keys": incompatible.unexpected_keys,
        "nonfinite_checkpoint_values": nonfinite_checkpoint_values,
        "running_var_tensors": len(running_vars),
        "affected_negative_running_var_tensors": affected_running_var_tensors,
        "negative_running_var_values": negative_running_var_values,
        "train_mode_embedding_finite": train_embedding_finite,
        "train_mode_reconstruction_finite": train_reconstruction_finite,
        "eval_mode_embedding_finite": eval_embedding_finite,
        "eval_mode_embedding_nan_count": int(torch.isnan(eval_embedding).sum().item()),
        "reverse_bridge_parameters_declared_but_unused": reverse_bridge_numel,
    }


def audit_synthetic_data() -> dict[str, Any]:
    metadata = pd.read_csv(METADATA)
    with h5py.File(H5, "r") as handle:
        ppg = handle["ppg"][:]
        labels = handle["labels"][:]
        patient_ids = handle["patient_ids"][:]
        noise_indices = handle["label_noise_indices"][:]

    aligned = (
        len(ppg)
        == len(labels)
        == len(patient_ids)
        == len(metadata)
        == 3000
    )
    metadata_labels_match_h5 = bool(
        (metadata["label_noisy"].to_numpy() == labels).all()
    )
    metadata_patients_match_h5 = bool(
        (metadata["patient_id"].to_numpy() == patient_ids).all()
    )
    metadata_noise_match_h5 = set(
        metadata.index[metadata["label_flipped"] == 1].tolist()
    ) == set(int(value) for value in noise_indices)

    return {
        "h5_sha256": sha256(H5),
        "metadata_sha256": sha256(METADATA),
        "ppg_shape": list(ppg.shape),
        "ppg_dtype": str(ppg.dtype),
        "label_distribution": {
            str(int(key.item())): int(value.item())
            for key, value in zip(*torch.unique(torch.from_numpy(labels), return_counts=True))
        },
        "subjects": int(len(set(int(value) for value in patient_ids))),
        "label_noise_rows": int(len(noise_indices)),
        "rows_aligned": aligned,
        "metadata_labels_match_h5": metadata_labels_match_h5,
        "metadata_patients_match_h5": metadata_patients_match_h5,
        "metadata_noise_indices_match_h5": metadata_noise_match_h5,
        "passed": aligned
        and metadata_labels_match_h5
        and metadata_patients_match_h5
        and metadata_noise_match_h5,
    }


def audit_code_and_spec() -> dict[str, Any]:
    config = yaml.safe_load((CODE / "cfg" / "xmae.yaml").read_text(encoding="utf-8"))
    model_source = (CODE / "utils" / "model_arch" / "xmae.py").read_text(
        encoding="utf-8"
    )
    trainer_source = (CODE / "utils" / "helper_trainer.py").read_text(
        encoding="utf-8"
    )
    simple_notebook = notebook_sources(CODE / "eval_0_simple_example.ipynb")
    pvc_notebook = notebook_sources(CODE / "eval_1_pvc.ipynb")
    paper_method = (TEX / "03_method.tex").read_text(encoding="utf-8")
    paper_training = (TEX / "appendix" / "training_details.tex").read_text(
        encoding="utf-8"
    )
    repro_statement = (TEX / "appendix" / "reproduce.tex").read_text(
        encoding="utf-8"
    )
    loss = config["loss_weights"]

    checks = {
        "paper_asymmetric_cross_reconstruction": (
            "PPG is always visible" in paper_method
            and "ECG is masked" in paper_method
        ),
        "paper_curriculum_80_to_90": (
            "M_0 = 80\\%" in paper_training
            and "M_{\\max} = 90\\%" in paper_training
        ),
        "code_directional_ecg_query_ppg_kv": (
            "upd = blk(t_ecg_hat, t_ppg)" in model_source
        ),
        "code_ecg_reconstruction": (
            'out["ecg_reconstructed"]' in model_source
            and "self.ecg_path.decode_full_wave" in model_source
        ),
        "config_ppg_unmasked": float(loss["mask_ratio_ppg"]) == 0.0,
        "config_ecg_mask_starts_80": float(loss["mask_ratio_ecg"]) == 0.8,
        "config_curriculum_enabled": bool(loss["use_cl"]),
        "trainer_declares_loss_ladder": 'schedule="ladder_loss"' in trainer_source,
        "trainer_has_ladder_promotion_update": (
            "_ladder_idx +=" in trainer_source
            or "_ladder_idx =" in trainer_source.replace("_ladder_idx = 0", "")
        ),
        "reverse_bridge_declared": "self.ppg2ecg = nn.ModuleList" in model_source,
        "reverse_bridge_called": "for blk in self.ppg2ecg" in model_source,
        "simple_notebook_loads_checkpoint": (
            "build_and_load_xmae(BACKBONE_CKPT)" in simple_notebook
        ),
        "simple_notebook_uses_true_ecg_tensor": (
            "ecg = ecg_signal2.to(device)" in simple_notebook
        ),
        "pvc_notebook_loads_checkpoint": (
            "build_and_load_xmae(BACKBONE_CKPT)" in pvc_notebook
            or "build_and_load_xmae(\"xmae_weights_permute.pth\")" in pvc_notebook
        ),
        "paper_says_real_weights_unreleased": (
            "unable to release the pretrained weights" in repro_statement
        ),
    }
    method_core_present = all(
        checks[key]
        for key in (
            "paper_asymmetric_cross_reconstruction",
            "paper_curriculum_80_to_90",
            "code_directional_ecg_query_ppg_kv",
            "code_ecg_reconstruction",
            "config_ppg_unmasked",
            "config_ecg_mask_starts_80",
            "trainer_declares_loss_ladder",
        )
    )
    method_matches_paper = (
        method_core_present
        and checks["config_curriculum_enabled"]
        and checks["trainer_has_ladder_promotion_update"]
    )
    return {
        "checks": checks,
        "method_core_present": method_core_present,
        "method_matches_paper": method_matches_paper,
        "canonical_mask_ratio_ppg": float(loss["mask_ratio_ppg"]),
        "canonical_mask_ratio_ecg": float(loss["mask_ratio_ecg"]),
        "canonical_use_cl": bool(loss["use_cl"]),
    }


def build_evidence() -> dict[str, Any]:
    checkpoint = audit_checkpoint()
    synthetic = audit_synthetic_data()
    code = audit_code_and_spec()
    c1 = 2 if code["method_matches_paper"] else 1 if code["method_core_present"] else 0
    released_example_core = (
        synthetic["passed"]
        and checkpoint["strict_load_passed"]
        and checkpoint["train_mode_embedding_finite"]
        and checkpoint["train_mode_reconstruction_finite"]
    )
    released_example_valid = (
        released_example_core
        and checkpoint["eval_mode_embedding_finite"]
        and code["checks"]["simple_notebook_uses_true_ecg_tensor"]
        and code["checks"]["pvc_notebook_loads_checkpoint"]
    )
    c2 = 2 if released_example_valid else 1 if released_example_core else 0
    return {
        "audit_version": 1,
        "official_commit": "7a0d24e1409f21e5ea34ae78c67c98706be0f930",
        "checkpoint": checkpoint,
        "synthetic_data": synthetic,
        "code_and_spec": code,
        "claims": {
            "C1": {
                "points": c1,
                "verdict": "PARTIAL_CORE_PRESENT_CURRICULUM_CONFIG_AND_UPDATE_MISMATCH",
            },
            "C2": {
                "points": c2,
                "verdict": "PARTIAL_ARTIFACTS_RUN_ONLY_AS_SMOKE_NOT_VALID_EVALUATION",
            },
            "C3": {"points": 0, "verdict": "INCONCLUSIVE_REAL_WEIGHTS_DATA_UNRELEASED"},
            "C4": {"points": 0, "verdict": "INCONCLUSIVE_COHORT_NOT_AVAILABLE"},
            "C5": {"points": 0, "verdict": "INCONCLUSIVE_REAL_WEIGHTS_DATA_UNRELEASED"},
            "C6": {"points": 0, "verdict": "INCONCLUSIVE_EVALUATION_STUDIES_UNRELEASED"},
        },
        "prepared_score": c1 + c2,
        "score_denominator": 12,
        "passed": synthetic["passed"]
        and checkpoint["strict_load_passed"]
        and code["method_core_present"],
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
