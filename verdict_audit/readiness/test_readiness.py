import numpy as np

import analyze_readiness as R


def unit(stratum="iclr-2025", url="None", media=(), title="t", abstract="a"):
    return dict(stratum=stratum, url=url, media_urls=list(media), title=title, abstract=abstract)


def test_code_link_rules():
    assert R.outcomes(unit(url="https://github.com/a/b"))["o1"] == 1
    assert R.outcomes(unit(media=["https://openreview.net/forum?id=x"]))["o1"] == 0
    assert R.outcomes(unit(media=["https://openreview.net/forum?id=x"]))["o2"] == 0
    assert R.outcomes(unit(url="https://myproject.github.io"))["o1"] == 0
    assert R.outcomes(unit(url="https://myproject.github.io"))["o2"] == 1
    assert R.outcomes(unit(stratum="cvpr-2025", url="https://cvpr.thecvf.com/x"))["o2"] == 0
    o = R.outcomes(unit(url="https://anonymous.4open.science/r/x"))
    assert o["o1"] == 1 and o["o1_anonymous"] == 1
    assert R.outcomes(unit(abstract="we use MIMIC-IV data"))["o3"] == 1
    assert R.outcomes(unit(url="/virtual/2025/poster/1"))["o2"] == 0


def test_mh_rd_and_stratified():
    assert abs(R.mh_rd(np.array([[10, 2, 10, 5]], float)) - (-0.3)) < 1e-12
    units = [dict(stratum="s1", g=1, y=0)] * 8 + [dict(stratum="s1", g=1, y=1)] * 2 + \
            [dict(stratum="s1", g=0, y=1)] * 5 + [dict(stratum="s1", g=0, y=0)] * 5
    out = R.stratified(units, lambda u: u["g"], "y", ["s1"], np.random.default_rng(0))
    assert abs(out["rd"] - (-0.3)) < 1e-12 and out["negative_strata"] == 1


def test_posthoc_mh_matches_risk_difference():
    import posthoc_p8 as PH
    stratum = lambda g: [dict(g=g, open=1, decided=int(i < 2)) for i in range(10)] + \
                        [dict(g=g, open=0, decided=int(i < 5)) for i in range(10)]
    assert abs(PH.mh(stratum("medical"), "open") - (-0.3)) < 1e-12
    assert abs(PH.mh(stratum("medical") + stratum("other"), "open") - (-0.3)) < 1e-12
    assert PH.mh([dict(g="medical", open=1, decided=1)], "open") is None


def test_consensus_order_does_not_depend_on_hash_seed():
    """Note 8-4: bootstrap input order must not follow set iteration order."""
    import pytest
    import p8_common as P
    if not (P.P8 / "keys.json").exists():
        pytest.skip("protocol 8 keys not present locally")
    import analyze_p8 as A8
    final, _, _ = A8.consensus("h85", tuple(A8.H85), A8.H85, False)
    assert list(final) == sorted(final)
