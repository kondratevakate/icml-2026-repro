"""Protocol 9, Note 9-10 (post hoc, descriptive): medical papers versus other papers split by ICML area.

PH9-c challenge outcomes per row (all ICML 2026 papers); PH9-d medical versus all other, adjusted for teams, and the
claim-level difference within team; PH9-e documentation per row (P9 codes, ICML 2025 and 2026 frame papers).

Usage: python posthoc_p9_areas.py   -> results/p9_areas.json
"""
from __future__ import annotations

import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import p8_common as P8  # noqa: E402
from analyze_p8 import wilson  # noqa: E402
from analyze_p9 import B, SEED_BOOT, final_codes, mh_rd  # noqa: E402
from build_p9_sheets import KEYS  # noqa: E402

LIFE = ("clinical", "life_science")
MED = "medical"


def area(topic: str) -> str:
    return (topic or "").split("->")[0].strip() or "unknown"


def boot_papers(papers, stat) -> list[float]:
    """Bootstrap over papers within group (medical / other), sorted input, fixed seed."""
    rng = np.random.default_rng(SEED_BOOT)
    groups = [sorted((p for p in papers if p["med"] == m), key=lambda p: p["orid"]) for m in (True, False)]
    vals = []
    for _ in range(B):
        s = [g[i] for g in groups for i in rng.integers(0, len(g), len(g))]
        v = stat(s)
        if v is not None:
            vals.append(v)
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]


def main() -> dict:
    units = [u for u in json.loads((P8.DATA / "papers.json").read_text(encoding="utf-8")) if u["stratum"] == "icml-2026"]
    labels = {}
    with (HERE / "codes" / "labels.csv").open(newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            if r["kind"] == "cand" and r["unit"].startswith("icml-2026:"):
                labels[r["unit"].split(":", 1)[1]] = r["label"]
    excluded = [u for u in units if u["candidate"] and labels.get(u["forum_id"]) not in LIFE]
    cv = P8.claim_verdicts()
    papers = []
    for u in units:
        o = u["forum_id"]
        if u in excluded:
            continue
        med = labels.get(o) in LIFE
        cl = cv.get(o)
        papers.append(dict(orid=o, med=med, row=MED if med else area(u["topic"]), attempted=cl is not None,
                           teams=len({P8.A.namespace(c["space"]) for c in cl}) if cl else 0,
                           rows=[(P8.A.namespace(c["space"]), c["verdict"] in P8.DECISIVE) for c in cl or []],
                           decided=int(any(c["verdict"] in P8.DECISIVE for c in cl or []))))
    res = dict(papers=len(papers), excluded_candidates_not_life_science=len(excluded))

    # PH9-c
    rows = {}
    for name in sorted({p["row"] for p in papers}, key=lambda r: (r != MED, r)):
        sel = [p for p in papers if p["row"] == name]
        att = [p for p in sel if p["attempted"]]
        cr = [d for p in att for _, d in p["rows"]]
        rows[name] = dict(papers=len(sel), attempted=len(att), attempted_share=len(att) / len(sel),
                          attempted_wilson=wilson(len(att), len(sel)),
                          mean_teams=float(np.mean([p["teams"] for p in att])) if att else None,
                          decided=sum(p["decided"] for p in att) / len(att) if att else None,
                          decided_wilson=wilson(sum(p["decided"] for p in att), len(att)),
                          claim_rows=len(cr), claim_decisive=sum(cr) / len(cr) if cr else None,
                          claim_decisive_wilson=wilson(sum(cr), len(cr)))
    res["PH9_c"] = rows

    # PH9-d
    att = [p for p in papers if p["attempted"]]
    team = lambda p: min(p["teams"], 3)
    d = {}
    for name, strat in (("unadjusted", lambda p: 0), ("teams", team)):
        stat = lambda s, strat=strat: mh_rd([(strat(p), p["med"], p["decided"]) for p in s])
        d[name] = dict(n=len(att), rd=stat(att), ci=boot_papers(att, stat))
    u, t = d["unadjusted"], d["teams"]
    t["accounts_for_gap"] = bool(t["ci"][0] <= 0 <= t["ci"][1] and abs(t["rd"]) < abs(u["rd"]) / 2)
    within = lambda s: mh_rd([(ns, p["med"], int(dec)) for p in s for ns, dec in p["rows"]])
    d["claim_level_within_team"] = dict(rd=within(att), ci=boot_papers(att, within))
    raw = lambda s: mh_rd([(0, p["med"], int(dec)) for p in s for _, dec in p["rows"]])
    d["claim_level_unadjusted"] = dict(rd=raw(att), ci=boot_papers(att, raw))
    res["PH9_d"] = d

    # PH9-e
    keys = json.loads(KEYS.read_text(encoding="utf-8"))
    topic = {(u["stratum"], u["forum_id"]): u["topic"] for u in json.loads((P8.DATA / "papers.json").read_text(encoding="utf-8"))
             if u["stratum"] in ("icml-2025", "icml-2026")}
    final = final_codes()
    pos = dict(code="released", main_data_access="open", weights_release="yes", variance="yes")
    by = defaultdict(list)
    for i, v in final.items():
        k = keys[i]
        if k["stratum"] in ("icml-2025", "icml-2026"):
            by[MED if k["group"] == "life_science" else area(topic.get((k["stratum"], k["forum_id"]), ""))].append(v)
    doc = {}
    for name in sorted(by, key=lambda r: (r != MED, r)):
        doc[name] = dict(n=len(by[name]))
        for f, val in pos.items():
            known = [v for v in by[name] if v[f] != "unclear"]
            k = sum(v[f] == val for v in known)
            doc[name][f] = dict(n_known=len(known), share=k / len(known) if known else None, wilson=wilson(k, len(known)))
    res["PH9_e"] = doc
    (HERE / "results" / "p9_areas.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    return res


if __name__ == "__main__":
    print(json.dumps(main(), indent=1)[:6000])
