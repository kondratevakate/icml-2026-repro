from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import torch


HERE = Path(__file__).resolve()
REPRO = HERE.parents[1]
sys.path.insert(0, str(REPRO))

from audit_generalization_gap import (  # noqa: E402
    build_actor,
    normalize_observation,
    risk_index,
)


class GeneralizationComponentTests(unittest.TestCase):
    def test_risk_index_matches_direct_formula(self) -> None:
        glucose = np.asarray([70.0, 100.0, 180.0])
        expected = np.mean(
            10.0 * (1.509 * (np.log(glucose) ** 1.084 - 5.381)) ** 2
        )
        self.assertAlmostEqual(risk_index(glucose), float(expected), places=12)

    def test_normalizer_uses_checkpoint_statistics(self) -> None:
        state = {
            "_count": torch.tensor(10.0),
            "_mean": torch.tensor([1.0, 2.0]),
            "_var": torch.tensor([4.0, 9.0]),
        }
        normalized = normalize_observation(np.asarray([3.0, 5.0]), state)
        np.testing.assert_allclose(normalized, [1.0, 1.0], atol=1e-7)

    def test_actor_restores_released_state_shape(self) -> None:
        config = {
            "model_cfgs": {
                "actor": {"hidden_sizes": [4, 4], "activation": "tanh"}
            }
        }
        reference = torch.nn.Sequential(
            torch.nn.Linear(3, 4),
            torch.nn.Tanh(),
            torch.nn.Linear(4, 4),
            torch.nn.Tanh(),
            torch.nn.Linear(4, 10),
        )
        state = {
            f"logits_net.{key}": value for key, value in reference.state_dict().items()
        }
        actor = build_actor(config, state)
        self.assertEqual(tuple(actor(torch.zeros(3)).shape), (10,))


if __name__ == "__main__":
    unittest.main()
