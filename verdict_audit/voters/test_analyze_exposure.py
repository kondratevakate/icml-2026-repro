import numpy as np

import analyze_exposure as X


def test_mh_rd_single_stratum():
    assert abs(X.mh_rd(np.array([[10, 8, 10, 5]], float)) - 0.3) < 1e-12


def test_mh_rd_removes_confounding_by_stratum():
    # Within each stratum the exposed-minus-unexposed difference is 0.1, but exposure is
    # concentrated in the high-agreement stratum, so the crude difference is larger.
    t = np.array([[90, 81, 10, 8], [10, 3, 90, 18]], float)
    crude = 84 / 100 - 26 / 100
    assert crude > 0.5
    assert abs(X.mh_rd(t) - 0.1) < 1e-12
    assert X.mh_rd(np.array([[5, 3, 0, 0]], float)) is None


def test_pair_table_counts():
    t = X.pair_table(np.array([0, 0, 1]), np.array([0, 1, 1]), np.array([1, 0, 1]), np.array([1, 0, 1]), 2, 2)
    assert t[0, 0].tolist() == [1, 1, 0, 0]
    assert t[0, 1].tolist() == [0, 0, 1, 0]
    assert t[1, 1].tolist() == [1, 1, 0, 0]


def test_unique_plurality_and_bins():
    assert X.unique_plurality(["verified", "verified", "toy"]) == "verified"
    assert X.unique_plurality(["verified", "toy"]) is None
    assert X.d_bin(0.2) == 0 and X.d_bin(3) == 3 and X.d_bin(30) == 5
    assert X.week_index("2026-07-21T23:00:00+00:00") == 0
    assert X.week_index("2026-07-22T00:00:00+00:00") == 1
    assert X.week_index("2026-08-01T00:00:00+00:00") == 2


def test_ac1_and_specific_agreement():
    out = X.ac1_and_specific([["verified", "verified"], ["verified", "falsified"]])
    assert abs(out["specific_agreement"]["verified"] - 2 / 3) < 1e-12
    assert out["specific_agreement"]["falsified"] == 0.0
    perfect = X.ac1_and_specific([["verified", "verified"], ["falsified", "falsified"]])
    assert abs(perfect["ac1"] - 1.0) < 1e-12
