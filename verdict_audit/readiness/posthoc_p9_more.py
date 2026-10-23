"""Protocol 9, Note 9-14 (post hoc): G1 full verification, G2 within-paper team strength, G3 herding robustness,
G4 within-team claim level by subgroup.

Usage: python posthoc_p9_more.py   -> results/p9_more.json
"""
from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import p8_common as P8  # noqa: E402
from analyze_p9 import B, SEED_BOOT, mh_rd  # noqa: E402
from posthoc_p9_adversarial import icml_units  # noqa: E402
from posthoc_p9_areas import LIFE, boot_papers  # noqa: E402


def load(lab, drop_ns=None):
    cv = P8.claim_verdicts()
    first = {}
    for r in P8.A.load_json(P8.VERDICTS).values():
        ns = P8.A.namespace(r["space_id"])
        if ns in (P8.AUTHOR, drop_ns) or r["orid"] not in lab:
            continue
        t = datetime.fromisoformat(r["judged_at"])
        k = (r["orid"], ns)
        if k not in first or (t, r["space_id"]) < first[k]:
            first[k] = (t, r["space_id"])
    teams = defaultdict(list)
    for (o, ns), (t, sid) in first.items():
        teams[o].append((t, sid, ns))
    t0 = min(t for v in teams.values() for t, _, _ in v)
    out = []
    for o in sorted(teams):
        by_team = defaultdict(list)
        for c in cv.get(o, []):
            ns = P8.A.namespace(c["space"])
            if ns != drop_ns:
                by_team[ns].append(c["verdict"])
        order = sorted(teams[o])
        dec = {ns: int(any(v in P8.DECISIVE for v in by_team.get(ns, []))) for _, _, ns in order}
        out.append(dict(orid=o, med=lab[o] in LIFE, label=lab[o] or "other", first_ns=order[0][2], team_dec=dec,
                        by_team=dict(by_team), first_decided=dec[order[0][2]], second=int(len(order) >= 2),
                        week=min((order[0][0] - t0).days // 7, 3),
                        fully_verified=int(any(v and all(x == "verified" for x in v) for v in by_team.values()))))
    hist = defaultdict(dict)
    for p in out:
        for ns, d in p["team_dec"].items():
            hist[ns][p["orid"]] = d
    for p in out:
        p["strength_of"] = {}
        for ns in p["team_dec"]:
            h = [d for o, d in hist[ns].items() if o != p["orid"]]
            p["strength_of"][ns] = float(np.mean(h)) if len(h) >= 3 else None
        p["strength"] = p["strength_of"][p["first_ns"]]
    vals = sorted(p["strength"] for p in out if p["strength"] is not None)
    lo, hi = np.percentile(vals, [100 / 3, 200 / 3])
    for p in out:
        s = p["strength"]
        p["s_bin"] = "few" if s is None else 0 if s <= lo else 1 if s <= hi else 2
    return out


def gap(papers, y, strat=lambda p: 0):
    stat = lambda s: mh_rd([(strat(p), p["med"], p[y]) for p in s])
    return dict(n=len(papers), rd=stat(papers), ci=boot_papers(papers, stat))


def boot_any(items, stat):
    rng = np.random.default_rng(SEED_BOOT)
    items = sorted(items, key=lambda p: p["orid"])
    v = [stat([items[i] for i in rng.integers(0, len(items), len(items))]) for _ in range(B)]
    v = [x for x in v if x is not None]
    return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]


def main() -> dict:
    lab = icml_units()
    papers = load(lab)
    res = {}

    # G1
    u = gap(papers, "fully_verified")
    rng = np.random.default_rng(SEED_BOOT)
    y, m = np.array([p["fully_verified"] for p in papers]), np.array([p["med"] for p in papers])
    obs = y[m].mean() - y[~m].mean()
    perm = [(lambda s: y[s].mean() - y[~s].mean())(rng.permutation(m)) for _ in range(B)]
    top = Counter(ns for p in papers if p["med"] for ns in p["team_dec"]).most_common(1)[0][0]
    no_top = load(lab, drop_ns=top)
    sub = {k: gap([p for p in papers if not p["med"] or p["label"] == k], "fully_verified", lambda p: p["s_bin"])
           for k in LIFE}
    g1 = dict(unadjusted=u, adj_strength=gap(papers, "fully_verified", lambda p: p["s_bin"]),
              permutation_p=float(np.mean(np.abs(perm) >= abs(obs))),
              drop_top_adj_strength=gap(no_top, "fully_verified", lambda p: p["s_bin"]), subgroups_adj_strength=sub)
    g1["survives"] = bool(g1["permutation_p"] < 0.05 and g1["drop_top_adj_strength"]["ci"][1] < 0
                          and sub["clinical"]["ci"][1] < 0)
    res["G1"] = g1

    # G2
    allv = sorted(s for p in papers for s in p["strength_of"].values() if s is not None)
    med = float(np.median(allv))

    def within(s):
        rows = [(p["orid"], st > med, p["team_dec"][ns]) for p in s for ns, st in p["strength_of"].items()
                if st is not None]
        return mh_rd(rows)
    multi = [p for p in papers if len(p["team_dec"]) >= 2]
    g2 = dict(median_strength=med, papers=len(multi), rd=within(multi), ci=boot_any(multi, within))
    mm = [p for p in multi if p["med"]]
    g2["medical_only"] = dict(papers=len(mm), rd=within(mm), ci=boot_any(mm, within) if len(mm) > 5 else None)
    g2["supported"] = bool(g2["ci"][0] > 0)
    res["G2"] = g2

    # G3
    def herd(s, strat=lambda p: 0):
        eff = []
        for flag in (True, False):
            g = [p for p in s if p["med"] == flag]
            rows = [(strat(p), bool(p["first_decided"]), p["second"]) for p in g]
            e = mh_rd(rows)
            if e is None:
                return None
            eff.append(e)
        return eff[0] - eff[1]
    g3 = dict(drop_top=dict(diff=herd(no_top), ci=boot_papers(no_top, herd)),
              within_week=dict(diff=herd(papers, lambda p: p["week"]),
                               ci=boot_papers(papers, lambda s: herd(s, lambda p: p["week"]))))
    g3["supported"] = bool(g3["drop_top"]["ci"][1] < 0 and g3["within_week"]["ci"][1] < 0)
    res["G3"] = g3

    # G4
    g4 = {}
    for k in LIFE:
        s = [p for p in papers if not p["med"] or p["label"] == k]
        st = lambda q: mh_rd([(ns, p["med"], int(v in P8.DECISIVE)) for p in q for ns, vs in p["by_team"].items()
                              for v in vs])
        d = dict(rd=st(s), ci=boot_papers(s, st))
        d["refuted"] = bool(d["ci"][1] < 0 and d["rd"] < -0.05)
        g4[k] = d
    res["G4"] = g4
    (HERE / "results" / "p9_more.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    return res


if __name__ == "__main__":
    print(json.dumps(main(), indent=1))
