"""Mutation control for Algorithm 1's out-of-fold evaluation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


TARGET = 0.9
EPS = 1e-6


def l2_ert(prediction: np.ndarray, cover: np.ndarray) -> float:
    h = np.clip(prediction, EPS, 1.0 - EPS)
    return float(np.mean((h - TARGET) ** 2 + (cover - h) * 2.0 * (h - TARGET)))


def folds(n: int, rng: np.random.Generator, k: int = 5):
    permutation = rng.permutation(n)
    for test_indices in np.array_split(permutation, k):
        mask = np.ones(n, dtype=bool)
        mask[test_indices] = False
        yield np.flatnonzero(mask), test_indices


def run(seed: int = 251211779, repetitions: int = 20, n: int = 2000) -> dict:
    rng = np.random.default_rng(seed)
    in_sample_false_positive = []
    crossfit_null = []
    crossfit_signal = []
    maximum_overlap = 0

    for _ in range(repetitions):
        x = rng.normal(size=(n, 8))
        null_cover = rng.binomial(1, TARGET, size=n).astype(float)

        # A memorizer predicts every training label exactly.
        in_sample_false_positive.append(l2_ert(null_cover, null_cover))

        null_prediction = np.empty(n)
        signal_prediction = np.empty(n)
        signal_probability = np.where(x[:, 0] >= 0.0, 1.0, 0.8)
        signal_cover = rng.binomial(1, signal_probability).astype(float)

        for train, test in folds(n, rng):
            maximum_overlap = max(
                maximum_overlap,
                int(np.intersect1d(train, test, assume_unique=True).size),
            )

            # Unique continuous inputs are unseen out of fold, so a pure
            # memorizer must fall back to the target predictor.
            null_prediction[test] = TARGET

            # A deliberately simple non-memorizing classifier estimates one
            # probability on either side of x_0 = 0.
            for positive_side in (False, True):
                train_side = (x[train, 0] >= 0.0) == positive_side
                test_side = (x[test, 0] >= 0.0) == positive_side
                estimate = float(signal_cover[train][train_side].mean())
                signal_prediction[test[test_side]] = estimate

        crossfit_null.append(l2_ert(null_prediction, null_cover))
        crossfit_signal.append(l2_ert(signal_prediction, signal_cover))

    result = {
        "status": "passed",
        "paper_anchor": "Algorithm 1",
        "seed": seed,
        "repetitions": repetitions,
        "sample_size": n,
        "folds": 5,
        "maximum_train_evaluation_overlap": maximum_overlap,
        "mean_in_sample_null_ert": float(np.mean(in_sample_false_positive)),
        "mean_crossfit_null_ert": float(np.mean(crossfit_null)),
        "mean_crossfit_signal_ert": float(np.mean(crossfit_signal)),
        "interpretation": (
            "A label memorizer creates a large false ERT signal in sample. "
            "Five-fold evaluation removes that signal while retaining a "
            "genuine conditional-coverage signal learned from held-out data."
        ),
    }
    if not (
        maximum_overlap == 0
        and result["mean_in_sample_null_ert"] > 0.07
        and abs(result["mean_crossfit_null_ert"]) < 1e-12
        and result["mean_crossfit_signal_ert"] > 0.005
    ):
        result["status"] = "failed"
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("results/algorithm1.json"))
    args = parser.parse_args()
    result = run()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    if result["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
