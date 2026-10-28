# Claim 3: evaluator validity and stability


---
<!-- trackio-cell
{"type": "markdown", "id": "cell_6aa081301588", "created_at": "2026-07-30T06:50:26+00:00", "title": "Claim 3: evaluator validity and stability"}
-->
**UNSUPPORTED — 0/2.** The evaluator adapter is an intact
288-tensor safetensors payload and matches
Hub LFS SHA-256. Artifact integrity does not reproduce ICC 0.9735 or stability:
the 100 expert/evaluator pairs and ten-run matrices were not released. The
published training script also names a psychology dataset and one epoch, not a
faithful paper recipe.


---
<!-- trackio-cell
{"type": "code", "id": "cell_bdc5df0d958c", "created_at": "2026-07-30T06:50:26+00:00", "title": "Claim 3: evaluator validity and stability evidence", "language": "python"}
-->
````output
{
  "adapter": {
    "bytes": 61380432,
    "sha256": "60556b253446fe762431b63054f3964277732f651af311e6059327f22a0b991c",
    "header_bytes": 38728,
    "tensor_count": 288,
    "parameter_count": 15335424,
    "dtype_counts": {
      "F32": 288
    },
    "payload_extent_matches_file": true,
    "hub_lfs_sha256": "60556b253446fe762431b63054f3964277732f651af311e6059327f22a0b991c",
    "hub_checksum_matches": true
  },
  "training": {
    "evaluator_adapter_config": {
      "alpha_pattern": {},
      "auto_mapping": null,
      "base_model_name_or_path": "/root/Qwen3-8B",
      "bias": "none",
      "corda_config": null,
      "eva_config": null,
      "exclude_modules": null,
      "fan_in_fan_out": false,
      "inference_mode": true,
      "init_lora_weights": true,
      "layer_replication": null,
      "layers_pattern": null,
      "layers_to_transform": null,
      "loftq_config": {},
      "lora_alpha": 32,
      "lora_bias": false,
      "lora_dropout": 0.0,
      "megatron_config": null,
      "megatron_core": "megatron.core",
      "modules_to_save": null,
      "peft_type": "LORA",
      "r": 16,
      "rank_pattern": {},
      "revision": null,
      "target_modules": [
        "self_attn.q_proj",
        "self_attn.o_proj",
        "self_attn.v_proj",
        "self_attn.k_proj"
      ],
      "task_type": "CAUSAL_LM",
      "trainable_token_indices": null,
      "use_dora": false,
      "use_rslora": false
    },
    "public_model_adapter_config": {
      "alpha_pattern": {},
      "auto_mapping": null,
      "base_model_name_or_path": "/root/autodl-tmp/LLM/Qwen3-8B",
      "bias": "none",
      "corda_config": null,
      "eva_config": null,
      "exclude_modules": null,
      "fan_in_fan_out": false,
      "inference_mode": true,
      "init_lora_weights": true,
      "layer_replication": null,
      "layers_pattern": null,
      "layers_to_transform": null,
      "loftq_config": {},
      "lora_alpha": 16,
      "lora_bias": false,
      "lora_dropout": 0.0,
      "megatron_config": null,
      "megatron_core": "megatron.core",
      "modules_to_save": null,
      "peft_type": "LORA",
      "qalora_group_size": 16,
      "r": 8,
      "rank_pattern": {},
      "revision": null,
      "target_modules": [
        "o_proj",
        "v_proj",
        "up_proj",
        "down_proj",
        "k_proj",
        "gate_proj",
        "q_proj"
      ],
      "target_parameters": null,
      "task_type": "CAUSAL_LM",
      "trainable_token_indices": null,
      "use_dora": false,
      "use_qalora": false,
      "use_rslora": false
    },
    "evaluator_train_results": {
      "epoch": 1.996218805142425,
      "total_flos": 1.404804282412155e+18,
      "train_loss": 0.6424942979908953,
      "train_runtime": 29125.2365,
      "train_samples_per_second": 0.272,
      "train_steps_per_second": 0.034
    },
    "public_model_train_results": {
      "epoch": 2.0,
      "total_flos": 1.277030840488113e+19,
      "train_loss": 0.8469410036818174,
      "train_runtime": 209948.1178,
      "train_samples_per_second": 2.311,
      "train_steps_per_second": 0.144
    },
    "released_script": {
      "mentions_psychology_dataset": true,
      "rank_16": true,
      "epochs_1": true,
      "max_samples_5400": true
    },
    "paper_public_model_config_matches_adapter": {
      "rank_8": true,
      "alpha_16": true,
      "dropout_0": true,
      "seven_target_modules": true,
      "two_epochs": true
    }
  }
}
````
