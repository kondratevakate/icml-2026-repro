"""Unit tests for verdict_audit/analyze.py on synthetic data."""
import unittest

import numpy as np

import analyze as A


class AgreementTests(unittest.TestCase):
    def test_perfect_agreement_gives_kappa_one(self):
        groups = {"p1": [["verified", "verified"], ["falsified", "falsified"]],
                  "p2": [["toy", "toy", "toy"]]}
        k = A.kappa_from_table(A.paper_table(groups, A.CATS))
        self.assertAlmostEqual(k["po"], 1.0)
        self.assertAlmostEqual(k["kappa"], 1.0)

    def test_hand_computed_example(self):
        # Claim 1: V,V,F -> P=2/6; claim 2: V,F -> P=0. Po=1/6.
        # Pooled: V=3, F=2 of 5 -> Pe=(9+4)/25.
        groups = {"p": [["verified", "verified", "falsified"], ["verified", "falsified"]]}
        k = A.kappa_from_table(A.paper_table(groups, A.CATS))
        pe = 13 / 25
        self.assertAlmostEqual(k["po"], 1 / 6)
        self.assertAlmostEqual(k["pe"], pe)
        self.assertAlmostEqual(k["kappa"], (1 / 6 - pe) / (1 - pe))

    def test_permutation_null_centres_on_chance(self):
        rng = np.random.default_rng(0)
        cats = ["verified", "falsified", "toy", "inconclusive"]
        claims = [[cats[i % 4]] * 3 for i in range(400)]  # perfect agreement
        null = A.permutation_po(claims, rng)
        pe = A.kappa_from_table(A.paper_table({"x": claims}, A.CATS))["pe"]
        self.assertLess(abs(null.mean() - pe), 0.02)
        self.assertLess(null.max(), 0.5)

    def test_shuffling_destroys_agreement(self):
        # Mutation test: identical labels within claims give kappa near 1;
        # labels shuffled across claims give kappa near 0.
        rng = np.random.default_rng(1)
        cats = ["verified", "falsified", "toy", "inconclusive"]
        claims = [[cats[rng.integers(4)]] * 2 for _ in range(2000)]
        flat = [r for c in claims for r in c]
        rng.shuffle(flat)
        shuffled = [flat[i:i + 2] for i in range(0, len(flat), 2)]
        k_true = A.kappa_from_table(A.paper_table({"x": claims}, A.CATS))["kappa"]
        k_mut = A.kappa_from_table(A.paper_table({"x": shuffled}, A.CATS))["kappa"]
        self.assertGreater(k_true, 0.99)
        self.assertLess(abs(k_mut), 0.05)


class DataHandlingTests(unittest.TestCase):
    def test_norm_and_category(self):
        self.assertEqual(A.norm("  Claim A \n holds "), "claim a holds")
        self.assertEqual(A.category("Verified"), "verified")
        self.assertEqual(A.category("unverified"), "other")
        self.assertEqual(A.category(None), "other")

    def test_dedupe_keeps_latest_per_reproducer_and_paper(self):
        recs = [dict(space_id="u/a", orid="P", judged_at="2026-07-10"),
                dict(space_id="u/b", orid="P", judged_at="2026-07-20"),
                dict(space_id="v/a", orid="P", judged_at="2026-07-01")]
        kept, dropped = A.dedupe(recs)
        self.assertEqual(dropped, 1)
        self.assertEqual({r["space_id"] for r in kept}, {"u/b", "v/a"})

    def test_multi_rated_requires_two_reproducers_and_anchor(self):
        anchored = {"P": {A.norm("Claim one")}}
        books = [dict(space_id="u/x", orid="P", claims=[dict(claim="claim ONE", verdict="verified"),
                                                         dict(claim="other", verdict="toy")]),
                 dict(space_id="v/x", orid="P", claims=[dict(claim="Claim one", verdict="falsified")]),
                 dict(space_id="w/x", orid="Q", claims=[dict(claim="Claim one", verdict="verified")])]
        groups = A.multi_rated(books, anchored, A.norm)
        self.assertEqual(groups, {"P": [["verified", "falsified"]]})

    def test_medical_gap_label_permutation_null(self):
        # Two groups with identical verdict mix: difference CI covers zero.
        books = []
        for i in range(60):
            verdicts = ["verified", "toy", "inconclusive", "verified"]
            books.append(dict(space_id=f"u/{i}", orid=f"P{i}",
                              claims=[dict(claim="c", verdict=v) for v in verdicts]))
        medical = {f"P{i}" for i in range(30)}
        out = A.medical_gap(books, medical, np.random.default_rng(2))
        self.assertAlmostEqual(out["diff"]["verified"], 0.0)
        self.assertLessEqual(out["diff_ci"]["verified"][0], 0.0)
        self.assertGreaterEqual(out["diff_ci"]["verified"][1], 0.0)


if __name__ == "__main__":
    unittest.main()
