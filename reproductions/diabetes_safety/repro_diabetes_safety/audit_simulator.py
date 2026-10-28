#!/usr/bin/env python3
"""Deterministic CPU execution audit for all released diabetes environments."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import numpy as np

import glucosim  # noqa: F401
from glucosim import gym_env as gym


ENVIRONMENTS = ("t1d-v0", "t2d-v0", "t2d_no_pump-v0")


def run_episode(env_id: str, seed: int, steps: int) -> dict:
    env = gym.make(
        env_id,
        simulation_minutes=24 * 60,
        sample_time=5,
        patient_name="adult#001",
    )
    observation, _ = env.reset(seed=seed)
    trajectory = []
    rewards = []
    costs = []
    action = np.asarray([0, 0], dtype=np.int64)

    for _ in range(steps):
        trajectory.append(np.asarray(observation, dtype=np.float64))
        observation, reward, cost, terminated, truncated, _ = env.step(action)
        rewards.append(float(reward))
        costs.append(float(cost))
        if terminated or truncated:
            break
    env.close()

    values = np.asarray(trajectory)
    digest = hashlib.sha256(values.tobytes()).hexdigest()
    return {
        "steps": int(values.shape[0]),
        "observation_dimension": int(values.shape[1]),
        "action_nvec": [5, 5],
        "trajectory_sha256": digest,
        "minimum_glucose": float(values[:, 0].min()),
        "maximum_glucose": float(values[:, 0].max()),
        "time_in_range_fraction": float(
            np.mean((values[:, 0] >= 70.0) & (values[:, 0] <= 180.0))
        ),
        "reward_sum": float(np.sum(rewards)),
        "cost_sum": float(np.sum(costs)),
    }


def audit(seed: int = 20260729, steps: int = 288) -> dict:
    environments = {}
    for env_id in ENVIRONMENTS:
        first = run_episode(env_id, seed, steps)
        second = run_episode(env_id, seed, steps)
        environments[env_id] = {
            **first,
            "repeat_sha256": second["trajectory_sha256"],
            "deterministic_repeat": first["trajectory_sha256"]
            == second["trajectory_sha256"],
        }

    return {
        "seed": seed,
        "requested_steps": steps,
        "environment_count": len(environments),
        "all_deterministic": all(
            item["deterministic_repeat"] for item in environments.values()
        ),
        "all_observation_dimensions_14": all(
            item["observation_dimension"] == 14 for item in environments.values()
        ),
        "environments": environments,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260729)
    parser.add_argument("--steps", type=int, default=288)
    args = parser.parse_args()

    report = audit(seed=args.seed, steps=args.steps)
    if not report["all_deterministic"]:
        raise AssertionError("A fixed seed did not reproduce an identical trajectory")
    if not report["all_observation_dimensions_14"]:
        raise AssertionError("Unexpected observation dimension")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
