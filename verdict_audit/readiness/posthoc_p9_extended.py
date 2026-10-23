"""Protocol 9, Note 9-13 (post hoc): team strength and experience, clinical versus life science, outcome definitions,
venue families.

Usage: python posthoc_p9_extended.py   -> results/p9_extended.json
"""
from __future__ import annotations

import csv
import json
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import p8_common as P8  # noqa: E402
from analyze_p9 import B, SEED_BOOT, boot_ci, final_codes, mh_rd  # noqa: E402
from build_p9_sheets import KEYS  # noqa: E402
from posthoc_p9_adversarial import POS, icml_units  # noqa: E402
from posthoc_p9_areas import LIFE, boot_papers  # noqa: E402

VISION = ("cvpr", "iccv", "eccv")


def load_papers(lab: dict) -> list[dict]:
    cv = P8.claim_verdicts()
    first = {}
    for r in P8.A.load_json(P8.VERDICTS).values():
        ns = P8.A.namespace(r["space_id"])
        if ns == P8.AUTHOR or r["orid"] not in lab:
            continue
        t = datetime.fromisoformat(r["judged_at"])
        k = (r["orid"], ns)
        if k not in first or (t, r["space_id"]) < first[k]:
            first[k] = (t, r["space_id"])
    teams = defaultdict(list)
    for (o, ns), (t, sid) in first.items():
        teams[o].append((t, sid, ns))
    out = []
    for o in sorted(teams):
        by_team = defaultdict(list)
        for c in cv.get(o, []):
            by_team[P8.A.namespace(c["space"])].append(c["verdict"])
        order = [ns for _, _, ns in sorted(teams[o])]
        team_dec = {ns: int(any(v in P8.DECISIVE for v in by_team.get(ns, []))) for ns in order}
        out.append(dict(orid=o, med=lab[o] in LIFE, label=lab[o] or "other", first_ns=order[0], team_dec=team_dec,
                        first_decided=team_dec[order[0]], decided=int(any(team_dec.values())),
                        fully_verified=int(any(v and all(x == "verified" for x in v) for v in by_team.values())),
                        any_falsified=int(any("falsified" in v for v in by_team.values())),
                        no_decisive=int(not any(team_dec.values()))))
    return out


def team_features(papers):
    hist = defaultdict(dict)
    for p in papers:
        for ns, d in p["team_dec"].items():
            hist[ns][p["orid"]] = d
    for p in papers:
        h = {o: d for o, d in hist[p["first_ns"]].items() if o != p["orid"]}
        p["strength"] = float(np.mean(list(h.values()))) if len(h) >= 3 else None
        p["experience"] = len(hist[p["first_ns"]])
    vals = sorted(p["strength"] for p in papers if p["strength"] is not None)
    lo, hi = np.percentile(vals, [100 / 3, 200 / 3])
    ex = sorted(p["experience"] for p in papers)
    elo, ehi = np.percentile(ex, [100 / 3, 200 / 3])
    for p in papers:
        s = p["strength"]
        p["s_bin"] = "few" if s is None else 0 if s <= lo else 1 if s <= hi else 2
        p["e_bin"] = 0 if p["experience"] <= elo else 1 if p["experience"] <= ehi else 2


def gap(papers, y, strat=lambda p: 0) -> dict:
    stat = lambda s: mh_rd([(strat(p), p["med"], p[y]) for p in s])
    return dict(n=len(papers), rd=stat(papers), ci=boot_papers(papers, stat))


def explains(adj, unadj) -> bool:
    return bool(abs(adj["rd"]) < abs(unadj["rd"]) / 2 and adj["ci"][0] <= 0 <= adj["ci"][1])


def mean_diff(a, b) -> dict:
    rng = np.random.default_rng(SEED_BOOT)
    a, b = np.array(sorted(a)), np.array(sorted(b))
    v = [rng.choice(a, len(a)).mean() - rng.choice(b, len(b)).mean() for _ in range(B)]
    return dict(n=[len(a), len(b)], mean=[float(a.mean()), float(b.mean())], diff=float(a.mean() - b.mean()),
                ci=[float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))])


def adjust_block(papers, binname) -> dict:
    out = {}
    for y in ("first_decided", "decided"):
        u, a = gap(papers, y), gap(papers, y, strat=lambda p: p[binname])
        out[y] = dict(unadjusted=u, adjusted=a, explains=explains(a, u))
    out["supported"] = out["first_decided"]["explains"] and out["decided"]["explains"]
    return out


def doc_rows(keys, final, f, keep):
    val = POS[f]
    return [(keys[i]["stratum"], keys[i]["group"] == "life_science", int(final[i][f] == val))
            for i in sorted(final) if keep(i) and final[i][f] != "unclear"]


def main() -> dict:
    lab = icml_units()
    papers = load_papers(lab)
    team_features(papers)
    res = dict(units=dict(medical=sum(p["med"] for p in papers), other=sum(not p["med"] for p in papers)))
    sm = [p["strength"] for p in papers if p["med"] and p["strength"] is not None]
    so = [p["strength"] for p in papers if not p["med"] and p["strength"] is not None]
    res["E1"] = dict(first_team_strength=mean_diff(sm, so),
                     few_share={g: float(np.mean([p["strength"] is None for p in papers if p["med"] == m]))
                                for g, m in (("medical", True), ("other", False))},
                     **adjust_block(papers, "s_bin"))
    res["E2"] = dict(first_team_experience=mean_diff([p["experience"] for p in papers if p["med"]],
                                                     [p["experience"] for p in papers if not p["med"]]),
                     **adjust_block(papers, "e_bin"))

    keys = json.loads(KEYS.read_text(encoding="utf-8"))
    final = final_codes()
    labels = {r["unit"]: r["label"] for r in csv.DictReader((HERE / "codes" / "labels.csv").open(encoding="utf-8"))
              if r["kind"] == "cand"}
    sub = lambda i: labels.get(f"{keys[i]['stratum']}:{keys[i]['forum_id']}")
    e3 = {}
    for kind in LIFE:
        keep = lambda i, kind=kind: keys[i]["group"] == "other" or sub(i) == kind
        doc = {}
        for f in POS:
            rows = doc_rows(keys, final, f, keep)
            doc[f] = dict(n=len(rows), rd=mh_rd(rows), ci=boot_ci(rows, mh_rd))
        pp = [p for p in papers if not p["med"] or p["label"] == kind]
        e3[kind] = dict(documentation=doc, decided=gap(pp, "decided"),
                        decided_adj_strength=gap(pp, "decided", strat=lambda p: p["s_bin"]))
    res["E3"] = e3
    res["E4"] = {y: dict(unadjusted=gap(papers, y), adj_strength=gap(papers, y, strat=lambda p: p["s_bin"]))
                 for y in ("fully_verified", "any_falsified", "no_decisive")}
    e5 = {}
    for fam, test in (("vision", lambda s: s.startswith(VISION)), ("learning", lambda s: not s.startswith(VISION))):
        e5[fam] = {}
        for f in POS:
            rows = doc_rows(keys, final, f, lambda i: test(keys[i]["stratum"]))
            e5[fam][f] = dict(n=len(rows), rd=mh_rd(rows), ci=boot_ci(rows, mh_rd))
    res["E5"] = e5
    (HERE / "results" / "p9_extended.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    return res


if __name__ == "__main__":
    print(json.dumps(main(), indent=1)[:2000])
