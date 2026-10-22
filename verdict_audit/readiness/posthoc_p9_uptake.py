"""Protocol 9, Note 9-11 (post hoc, descriptive): why do fewer teams take up medical ICML 2026 papers?

Outcome: a second team on an attempted paper. T1 herding (first logbook decided), T2 expected points (claim count,
compute tier), T3 who takes medical papers, T4 entry threshold (data scale) and documentation.

Usage: python posthoc_p9_uptake.py   -> results/p9_uptake.json
"""
from __future__ import annotations

import csv
import json
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import p8_common as P8  # noqa: E402
from analyze_p9 import B, SEED_BOOT, final_codes, mh_rd  # noqa: E402
from build_p9_sheets import KEYS  # noqa: E402
from posthoc_p9_areas import LIFE, boot_papers  # noqa: E402


def tertile(values):
    lo, hi = np.percentile(values, [100 / 3, 200 / 3])
    return lambda v: 0 if v <= lo else 1 if v <= hi else 2


def gap(papers, strat, keep=lambda p: True) -> dict:
    sel = [p for p in papers if keep(p)]
    stat = lambda s: mh_rd([(strat(p), p["med"], p["second"]) for p in s])
    return dict(n=len(sel), rd=stat(sel), ci=boot_papers(sel, stat))


def accounts(adj, unadj) -> bool:
    return bool(adj["ci"][0] <= 0 <= adj["ci"][1] and abs(adj["rd"]) < abs(unadj["rd"]) / 2)


def mean_diff(a, b) -> dict:
    """Mean of a minus mean of b, bootstrap over papers within each side."""
    rng = np.random.default_rng(SEED_BOOT)
    a, b = np.array(sorted(a)), np.array(sorted(b))
    if not len(a) or not len(b):
        return dict(n=[len(a), len(b)], diff=None, ci=None)
    v = [rng.choice(a, len(a)).mean() - rng.choice(b, len(b)).mean() for _ in range(B)]
    return dict(n=[len(a), len(b)], mean=[float(a.mean()), float(b.mean())], diff=float(a.mean() - b.mean()),
                ci=[float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))])


def main() -> dict:
    units = [u for u in json.loads((P8.DATA / "papers.json").read_text(encoding="utf-8")) if u["stratum"] == "icml-2026"]
    labels = {}
    with (HERE / "codes" / "labels.csv").open(newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            if r["kind"] == "cand" and r["unit"].startswith("icml-2026:"):
                labels[r["unit"].split(":", 1)[1]] = r["label"]
    keep = {u["forum_id"]: labels.get(u["forum_id"]) in LIFE for u in units
            if not (u["candidate"] and labels.get(u["forum_id"]) not in LIFE)}

    # earliest judged logbook per (team, paper), author's namespace excluded
    first = {}
    for r in P8.A.load_json(P8.VERDICTS).values():
        ns = P8.A.namespace(r["space_id"])
        if ns == P8.AUTHOR or r["orid"] not in keep:
            continue
        t = datetime.fromisoformat(r["judged_at"])
        dec = any(P8.A.category(c.get("verdict")) in P8.DECISIVE for c in r.get("claims") or [])
        k = (r["orid"], ns)
        if k not in first or (t, r["space_id"]) < first[k][:2]:
            first[k] = (t, r["space_id"], dec)
    entries = defaultdict(list)
    for (o, ns), (t, sid, dec) in first.items():
        entries[o].append((t, sid, ns, dec))
    anchored = json.loads((P8.ROOT / "claims_anchored.json").read_text(encoding="utf-8-sig"))
    t0 = min(e[0] for v in entries.values() for e in v)
    papers = []
    for o in sorted(entries):
        e = sorted(entries[o])
        papers.append(dict(orid=o, med=keep[o], teams=len(e), second=int(len(e) >= 2), first_decided=int(e[0][3]),
                           week=min((e[0][0] - t0).days // 7, 3), n_anchored=len(anchored.get(o, [])),
                           team_list=[x[2] for x in e]))
    res = dict(units=dict(Counter("medical" if p["med"] else "other" for p in papers)))
    u = gap(papers, lambda p: 0)

    # T1
    t1 = dict(unadjusted=u)
    t1["rates"] = {g: {s: dict(n=len(sel), second=float(np.mean([p["second"] for p in sel])) if sel else None)
                       for s, sel in (("first_decided", [p for p in papers if p["med"] == m and p["first_decided"]]),
                                      ("first_not_decided", [p for p in papers if p["med"] == m and not p["first_decided"]]))}
                   for g, m in (("medical", True), ("other", False))}
    t1["first_decided_share"] = {g: float(np.mean([p["first_decided"] for p in papers if p["med"] == m]))
                                 for g, m in (("medical", True), ("other", False))}
    t1["adj_first_status"] = gap(papers, lambda p: p["first_decided"])
    t1["adj_first_status_and_week"] = gap(papers, lambda p: (p["first_decided"], p["week"]))
    for k in ("adj_first_status", "adj_first_status_and_week"):
        t1[k]["accounts_for_gap"] = accounts(t1[k], u)
    res["T1_herding"] = t1

    # T2
    ct = tertile([p["n_anchored"] for p in papers])
    t2 = dict(mean_anchored={g: float(np.mean([p["n_anchored"] for p in papers if p["med"] == m]))
                             for g, m in (("medical", True), ("other", False))},
              adj_claim_count=gap(papers, lambda p: ct(p["n_anchored"])))
    t2["adj_claim_count"]["accounts_for_gap"] = accounts(t2["adj_claim_count"], u)
    keys = json.loads(KEYS.read_text(encoding="utf-8"))
    final = final_codes()
    code = {keys[i]["forum_id"]: final[i] for i in final if keys[i]["stratum"] == "icml-2026"}
    frame = [dict(p, **code[p["orid"]]) for p in papers if p["orid"] in code]
    fu = gap(frame, lambda p: 0)
    t2["frame_unadjusted"] = fu
    t2["frame_adj_compute_tier"] = gap(frame, lambda p: p["compute_tier"])
    t2["frame_adj_compute_tier"]["accounts_for_gap"] = accounts(t2["frame_adj_compute_tier"], fu)
    res["T2_points"] = t2

    # T3
    t3 = {}
    for g, m in (("medical", True), ("other", False)):
        att = Counter(ns for p in papers if p["med"] == m for ns in p["team_list"])
        tot = sum(att.values())
        top = [c for _, c in att.most_common(5)]
        t3[g] = dict(attempts=tot, distinct_teams=len(att), top1_share=top[0] / tot, top5_share=sum(top) / tot)
    med_teams = {ns for p in papers if p["med"] for ns in p["team_list"]}
    shares = []
    for ns in sorted(med_teams):
        mine = [p["med"] for p in papers if ns in p["team_list"]]
        shares.append(sum(mine) / len(mine))
    t3["teams_with_medical_attempt"] = dict(n=len(shares), median_medical_share=float(np.median(shares)),
                                            share_of_teams_mostly_medical=float(np.mean([s >= 0.5 for s in shares])))
    res["T3_who"] = t3

    # T4
    t4 = dict(frame_unadjusted=fu, frame_adj_data_scale=gap(frame, lambda p: p["data_scale"]))
    t4["frame_adj_data_scale"]["accounts_for_gap"] = accounts(t4["frame_adj_data_scale"], fu)
    t4["documentation_vs_teams"] = {}
    for f, val in (("code", "released"), ("main_data_access", "open"), ("synthetic_or_sample_release", "yes")):
        t4["documentation_vs_teams"][f] = {
            g: mean_diff([p["teams"] for p in frame if p["med"] == m and p[f] == val],
                         [p["teams"] for p in frame if p["med"] == m and p[f] not in (val, "unclear")])
            for g, m in (("medical", True), ("other", False))}
    res["T4_threshold"] = t4
    (HERE / "results" / "p9_uptake.json").write_text(json.dumps(res, indent=1, default=str), encoding="utf-8")
    return res


if __name__ == "__main__":
    print(json.dumps(main(), indent=1, default=str))
