"""Synthetic tests for analyze_e1.py: models recover a planted effect, dictionaries behave."""
import unittest

import numpy as np
import pandas as pd

import analyze_e1 as E


def synthetic(effect, seed=0, n_rep=60, n_pap=400):
    rng = np.random.default_rng(seed)
    skill = rng.normal(0, 0.15, n_rep)
    med = rng.random(n_pap) < 0.2
    diff = rng.normal(0, 0.1, n_pap)
    rows = []
    for p in range(n_pap):
        for r in rng.choice(n_rep, 2, replace=False):
            for _ in range(5):
                theory = int(rng.random() < 0.3)
                pr = np.clip(0.45 + skill[r] + diff[p] + effect * med[p] + 0.1 * theory, 0.01, 0.99)
                rows.append(dict(reproducer=f"r{r}", paper=f"p{p}", verified=int(rng.random() < pr),
                                 medical=int(med[p]), narrow=int(med[p]), theory=theory, discovery=1,
                                 organiser=0, verdict="verified", evidence=""))
    df = pd.DataFrame(rows)
    df["w_paper"] = 1.0 / df.groupby("paper")["verified"].transform("size")
    return df


class ModelTests(unittest.TestCase):
    def test_lpm_recovers_planted_effect(self):
        res = E.lpm(synthetic(-0.10))
        self.assertLess(res["ci"][0], -0.10)
        self.assertGreater(res["ci"][1], -0.10)
        self.assertLess(res["ci"][1], 0)

    def test_lpm_null_covers_zero(self):
        res = E.lpm(synthetic(0.0, seed=3))
        self.assertLess(res["ci"][0], 0)
        self.assertGreater(res["ci"][1], 0)

    def test_clogit_runs_and_sign(self):
        res = E.clogit(synthetic(-0.15, seed=5), boot=30)
        self.assertLess(res["log_or"], 0)
        self.assertGreater(res["successful_bootstraps"], 20)


class DictionaryTests(unittest.TestCase):
    def test_theory_regex(self):
        self.assertTrue(E.THEORY_RE.search("theorem 3 gives an o(1/n) bound"))
        self.assertTrue(E.THEORY_RE.search("the estimator is consistent"))
        self.assertFalse(E.THEORY_RE.search("the method reaches 91% accuracy on imagenet"))

    def test_reason_precedence(self):
        self.assertEqual(E.reason("Weights were not released, and only a toy run exists"), "artifact_absent")
        self.assertEqual(E.reason("The experiment was not run."), "not_attempted")
        self.assertEqual(E.reason("Only a toy synthetic check."), "reduced_scale")
        self.assertEqual(E.reason("Evidence does not match the claim."), "other")


if __name__ == "__main__":
    unittest.main()
