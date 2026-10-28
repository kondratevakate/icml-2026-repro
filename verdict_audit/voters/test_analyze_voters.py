import numpy as np

import analyze_voters as V
import common as C


def test_jaccard_and_containment():
    a = V.shingles("one two three four five six seven eight")
    b = V.shingles("one two three four five six seven nine")
    assert len(a) == 4 and len(b) == 4
    assert V.jaccard(a, a) == 1.0
    assert abs(V.jaccard(a, b) - 3 / 5) < 1e-12
    assert V.containment(a, b) == 0.75
    assert np.array_equal(V.shingles("A  b C d E"), V.shingles("a b c d e"))  # normalized


def test_choose_tau_ladder():
    assert V.choose_tau(np.zeros(1000)) == 0.5
    assert V.choose_tau(np.full(1000, 0.65)) == 0.7
    assert V.choose_tau(np.full(1000, 0.95)) is None


def test_collapse_keeps_earliest_verdict():
    raters = {"x/a": "verified", "y/b": "verified", "z/c": "falsified"}
    created = {"x/a": "2026-07-20T00:00:00Z", "y/b": "2026-07-15T00:00:00Z", "z/c": None}
    edges = {frozenset(("x/a", "z/c"))}
    out = sorted(V.collapse(raters, edges, created))
    assert out == ["verified", "verified"]  # z/c merged into x/a, x/a is earlier (None sorts last)
    assert len(V.components(sorted(raters), edges)) == 2


def test_spearman_brown_matches_formula():
    assert abs(V.spearman_brown(0.23, 0.7) - 0.7 * 0.77 / (0.23 * 0.3)) < 1e-12
    assert V.spearman_brown(0.0, 0.7) is None


def test_majority():
    assert V.majority(["verified", "falsified", "verified", "toy"]) == "verified"
    assert V.majority(["verified", "falsified"]) == "tie"
    assert V.majority(["toy"]) == "no_decisive"


def test_dawid_skene_recovers_truth_and_downweights_bad_annotator():
    rng = np.random.default_rng(0)
    truth = rng.integers(0, 2, 300)
    items = []
    for t in truth:
        votes = [(f"good{j}", int(t) if rng.random() < 0.9 else 1 - int(t)) for j in range(3)]
        votes.append(("bad", 1 - int(t)))  # always wrong
        items.append(votes)
    out = V.dawid_skene(items)
    assert out["converged"]
    assert np.mean((out["posterior"] > 0.5) == truth) > 0.95


def test_claim_page_rule():
    assert V.is_claim_page("pages/claim-1-foo/page.md")
    assert V.is_claim_page("pages/Claim_2.md")
    assert not V.is_claim_page("pages/executive-summary/page.md")
    text, had = V.logbook_text({"logbook.json": "{}", "pages/index.md": "x", "pages/claim-1/page.md": "y"})
    assert had and text == "y"


def test_claim_map_matches_frozen_multi_rated():
    logbooks, _, _, anchored, _ = C.load()
    mine = C.multi_rated_claims(logbooks, anchored)
    frozen = C.A.multi_rated(logbooks, anchored, C.A.norm)
    assert len(mine) == sum(len(v) for v in frozen.values())
    assert sorted(sorted(v.values()) for v in mine.values()) == sorted(sorted(c) for v in frozen.values() for c in v)
