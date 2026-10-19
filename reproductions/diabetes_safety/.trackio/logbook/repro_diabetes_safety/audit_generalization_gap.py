#!/usr/bin/env python3
"""Scoped independent evaluation of one released safe-RL checkpoint."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import numpy as np
import torch
from torch import nn

import glucosim  # noqa: F401
from glucosim import gym_env as gym


def build_actor(config: dict, state_dict: dict[str, torch.Tensor]) -> nn.Sequential:
    actor_config = config["model_cfgs"]["actor"]
    hidden_sizes = actor_config["hidden_sizes"]
    activation_name = actor_config["activation"]
    activation_type = {"tanh": nn.Tanh, "relu": nn.ReLU}[activation_name]
    obs_dim = int(state_dict["logits_net.0.weight"].shape[1])
    output_dim = int(state_dict["logits_net.4.weight"].shape[0])

    sizes = [obs_dim, *hidden_sizes, output_dim]
    layers: list[nn.Module] = []
    for index, (input_size, output_size) in enumerate(zip(sizes, sizes[1:])):
        layers.append(nn.Linear(input_size, output_size))
        if index < len(sizes) - 2:
            layers.append(activation_type())
    actor = nn.Sequential(*layers)
    remapped = {
        key.removeprefix("logits_net."): value for key, value in state_dict.items()
    }
    actor.load_state_dict(remapped, strict=True)
    actor.eval()
    return actor


def normalize_observation(
    observation: np.ndarray, normalizer: dict[str, torch.Tensor]
) -> np.ndarray:
    count = float(normalizer["_count"].item())
    if count <= 0:
        return observation
    mean = normalizer["_mean"].detach().cpu().numpy()
    variance = normalizer["_var"].detach().cpu().numpy()
    return (observation - mean) / np.sqrt(variance + 1e-8)


def risk_index(glucose: np.ndarray) -> float:
    glucose = np.clip(glucose.astype(np.float64), 1e-6, None)
    transform = 1.509 * (np.log(glucose) ** 1.084 - 5.381)
    return float(np.mean(10.0 * transform**2))


def evaluate_patient(
    actor: nn.Sequential,
    normalizer: dict[str, torch.Tensor],
    patient_name: str,
    evaluation_seed: int,
) -> dict:
    torch.manual_seed(evaluation_seed)
    np.random.seed(evaluation_seed)
    env = gym.make(
        "t1d-v0",
        simulation_minutes=7 * 24 * 60,
        sample_time=5,
        patient_name=patient_name,
    )
    observation, _ = env.reset(seed=evaluation_seed)
    glucose = []
    rewards = []
    costs = []
    actions = []

    while True:
        raw = np.asarray(observation, dtype=np.float64)
        normalized = normalize_observation(raw, normalizer)
        with torch.no_grad():
            logits = actor(torch.as_tensor(normalized, dtype=torch.float32))
            bolus = torch.distributions.Categorical(logits=logits[:5]).sample()
            meal = torch.distributions.Categorical(logits=logits[5:]).sample()
        action = np.asarray([int(bolus.item()), int(meal.item())], dtype=np.int64)
        actions.append(action)
        observation, reward, cost, terminated, truncated, _ = env.step(action)
        glucose.append(float(np.asarray(observation, dtype=np.float64)[0]))
        rewards.append(float(reward))
        costs.append(float(cost))
        if terminated or truncated:
            break
    env.close()

    glucose_values = np.asarray(glucose)
    action_values = np.asarray(actions)
    return {
        "patient": patient_name,
        "steps": int(glucose_values.size),
        "tir_percent": float(
            100.0
            * np.mean((glucose_values >= 70.0) & (glucose_values <= 180.0))
        ),
        "risk_index": risk_index(glucose_values),
        "cv_percent": float(
            100.0 * np.std(glucose_values) / np.mean(glucose_values)
        ),
        "minimum_glucose": float(glucose_values.min()),
        "maximum_glucose": float(glucose_values.max()),
        "reward_sum": float(np.sum(rewards)),
        "cost_sum": float(np.sum(costs)),
        "action_sha256": hashlib.sha256(action_values.tobytes()).hexdigest(),
    }


def audit(checkpoint: Path, config_path: Path, evaluation_seed: int) -> dict:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    actor = build_actor(config, payload["pi"])
    normalizer = payload["obs_normalizer"]

    id_result = evaluate_patient(
        actor, normalizer, "adolescent#001", evaluation_seed
    )
    ood_results = [
        evaluate_patient(
            actor,
            normalizer,
            f"adolescent#{patient_index:03d}",
            evaluation_seed,
        )
        for patient_index in range(2, 11)
    ]
    ood_tir = float(np.mean([item["tir_percent"] for item in ood_results]))
    ood_risk = float(np.mean([item["risk_index"] for item in ood_results]))
    return {
        "scope": "CPO_T1D_ADOLESCENT_SEED0_ONE_7_DAY_EPISODE_PER_PATIENT",
        "evaluation_seed": evaluation_seed,
        "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        "metric_trace_contract": (
            "CORRECTED_POST_STEP_OBSERVATION_0_ONE_VALUE_PER_ACTION"
        ),
        "released_evaluator_effective_trace": (
            "PRE_STEP_OBSERVATION_0_BECAUSE_INFO_HAS_NO_CGM_KEY"
        ),
        "id": id_result,
        "ood": ood_results,
        "ood_mean_tir_percent": ood_tir,
        "ood_mean_risk_index": ood_risk,
        "tir_gap_ood_minus_id": ood_tir - id_result["tir_percent"],
        "risk_gap_ood_minus_id": ood_risk - id_result["risk_index"],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--evaluation-seed", type=int, default=20260729)
    args = parser.parse_args()

    report = audit(args.checkpoint, args.config, args.evaluation_seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
