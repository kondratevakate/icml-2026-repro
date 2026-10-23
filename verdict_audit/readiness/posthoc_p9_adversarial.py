"""Protocol 9, Note 9-12: adversarial checks of the post hoc claims (F1-F5, refutation criteria fixed in the note).

Usage: python posthoc_p9_adversarial.py   -> results/p9_adversarial.json
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
from analyze_p9 import (B, REF_FIELDS, SEED_BOOT, binary_agreement, boot_ci, coder2_all, final_codes, mh_rd,  # noqa: E402
                        read, reference_codes)
from build_p9_adjudication import ref_texts  # noqa: E402
from build_p9_sheets import KEYS, excerpts  # noqa: E402
from posthoc_p9_areas import LIFE, boot_papers  # noqa: E402

POS = dict(code="released", main_data_access="open", weights_release="yes", variance="yes")


def ci_lo_pos(d):
    return bool(d["ci"] and d["ci"][0] > 0)


def est(rows):
    return dict(n=len(rows), rd=mh_rd(rows), ci=boot_ci(rows, mh_rd))


def f1(keys, final) -> dict:
    ex = excerpts()
    c1, c2 = read("coder1"), coder2_all()
    length = {i: len(ex[keys[i]["key"]]["excerpt"]) for i in final}
    trunc = {i: ex[keys[i]["key"]]["truncated"] for i in final}
    lo, hi = np.percentile(sorted(length.values()), [100 / 3, 200 / 3])
    lt = lambda i: 0 if length[i] <= lo else 1 if length[i] <= hi else 2
    out = {}
    for f, val in POS.items():
        def rows(codes, keep=lambda i: True, strat=lambda i: keys[i]["stratum"]):
            return [(strat(i), keys[i]["group"] == "life_science", int(codes[i][f] == val))
                    for i in sorted(final) if keep(i) and codes[i][f] != "unclear"]
        strata = sorted({keys[i]["stratum"] for i in final})
        loo = {s: mh_rd([r for r in rows(final) if r[0] != s]) for s in strata}
        r = dict(leave_one_out=loo, loo_all_positive=all(v > 0 for v in loo.values()))
        if f in ("code", "main_data_access", "weights_release"):  # above the gate: final = coder 1
            r["coder1_only"] = est(rows(c1))
            c2_shared = {i: c2[i] for i in c2}
            r["coder2_only"] = est(rows(c2_shared, keep=lambda i: i in c2_shared))
        else:
            r["coder1_only"] = est(rows(c1))
            r["coder2_only"] = est(rows(c2))
        r["not_truncated"] = est(rows(final, keep=lambda i: not trunc[i]))
        r["by_venue_and_length"] = est(rows(final, strat=lambda i: (keys[i]["stratum"], lt(i))))
        r["survives"] = bool(r["loo_all_positive"] and ci_lo_pos(r["coder1_only"]) and ci_lo_pos(r["coder2_only"])
                             and ci_lo_pos(r["not_truncated"]) and ci_lo_pos(r["by_venue_and_length"]))
        out[f] = r
    return out


def icml_units():
    units = [u for u in json.loads((P8.DATA / "papers.json").read_text(encoding="utf-8")) if u["stratum"] == "icml-2026"]
    labels = {}
    with (HERE / "codes" / "labels.csv").open(newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            if r["kind"] == "cand" and r["unit"].startswith("icml-2026:"):
                labels[r["unit"].split(":", 1)[1]] = r["label"]
    return {u["forum_id"]: labels.get(u["forum_id"]) for u in units
            if not (u["candidate"] and labels.get(u["forum_id"]) not in LIFE)}


def papers_from(lab, drop_ns=None):
    """Attempted papers with decided (latest logbook per team) and the first team's first logbook."""
    cv = P8.claim_verdicts()
    first = {}
    for r in P8.A.load_json(P8.VERDICTS).values():
        ns = P8.A.namespace(r["space_id"])
        if ns in (P8.AUTHOR, drop_ns) or r["orid"] not in lab:
            continue
        t = datetime.fromisoformat(r["judged_at"])
        dec = any(P8.A.category(c.get("verdict")) in P8.DECISIVE for c in r.get("claims") or [])
        k = (r["orid"], ns)
        if k not in first or (t, r["space_id"]) < first[k][:2]:
            first[k] = (t, r["space_id"], dec)
    per = defaultdict(list)
    for (o, ns), (t, sid, dec) in first.items():
        per[o].append((t, sid, ns, dec))
    out = []
    for o in sorted(per):
        e = sorted(per[o])
        rows = [(P8.A.namespace(c["space"]), c["verdict"] in P8.DECISIVE) for c in cv.get(o, [])
                if P8.A.namespace(c["space"]) != drop_ns]
        out.append(dict(orid=o, med=lab[o] in LIFE, label=lab[o], rows=rows, first_ns=e[0][2],
                        first_decided=int(e[0][3]), second=int(len(e) >= 2), decided=int(any(d for _, d in rows))))
    return out


def gap(papers, y, strat=lambda p: 0):
    stat = lambda s: mh_rd([(strat(p), p["med"], p[y]) for p in s])
    return dict(n=len(papers), rd=stat(papers), ci=boot_papers(papers, stat))


def f2_f4(lab) -> dict:
    papers = papers_from(lab)
    res = {}
    u = gap(papers, "decided")
    rng = np.random.default_rng(SEED_BOOT)
    y, m = np.array([p["decided"] for p in papers]), np.array([p["med"] for p in papers])
    obs = y[m].mean() - y[~m].mean()
    perm = []
    for _ in range(B):
        s = rng.permutation(m)
        perm.append(y[s].mean() - y[~s].mean())
    p_perm = float(np.mean(np.abs(perm) >= abs(obs)))
    sub = {k: gap([p for p in papers if not p["med"] or p["label"] == k], "decided") for k in LIFE}
    top = Counter(ns for p in papers if p["med"] for ns, _ in set(p["rows"])).most_common(1)[0][0]
    no_top = [p for p in papers_from(lab, drop_ns=top) if p["rows"]]
    nt = gap(no_top, "decided")
    res["F2"] = dict(unadjusted=u, permutation=dict(diff=float(obs), p=p_perm), label_split=sub,
                     top_medical_team_dropped=nt,
                     survives=bool(p_perm < 0.05 and all(v["ci"][1] < 0 for v in sub.values()) and nt["ci"][1] < 0))

    first = gap(papers, "first_decided")
    shared = {p["first_ns"] for p in papers if p["med"]} & {p["first_ns"] for p in papers if not p["med"]}
    fp = [p for p in papers if p["first_ns"] in shared]
    adj = gap(fp, "first_decided", strat=lambda p: p["first_ns"])
    adj_unadj = gap(fp, "first_decided")
    within = lambda s: mh_rd([(ns, p["med"], int(d)) for p in s for ns, d in p["rows"]])
    w = dict(rd=within(no_top), ci=boot_papers(no_top, within))
    res["F3"] = dict(all_team_gap=u, first_logbook_gap=first,
                     depth_alone_explains=bool(abs(first["rd"]) < abs(u["rd"]) / 2),
                     first_logbook_shared_teams=dict(unadjusted=adj_unadj, adjusted_for_first_team=adj, teams=len(shared)),
                     composition_explains=bool(abs(adj["rd"]) < abs(adj_unadj["rd"]) / 2 and adj["ci"][0] <= 0 <= adj["ci"][1]),
                     within_team_claim_level_without_top=w,
                     within_team_refuted=bool(w["ci"][1] < 0 and w["rd"] < -0.05))

    def herd(s):
        eff = []
        for m_ in (True, False):
            g = [p for p in s if p["med"] == m_]
            a = [p["second"] for p in g if p["first_decided"]]
            b = [p["second"] for p in g if not p["first_decided"]]
            if not a or not b:
                return None
            eff.append(np.mean(a) - np.mean(b))
        return float(eff[0] - eff[1])
    h = dict(diff_in_herding=herd(papers), ci=boot_papers(papers, herd))
    h["supported"] = bool(h["ci"][1] < 0)
    res["F4"] = h
    return res


def f5(keys, final) -> dict:
    by = {i: final[i] for i in final}
    trunc = {i: t == "yes" for i, (t, _) in ref_texts().items()}
    variants = dict(adjudicated=reference_codes(), ref1=read("ref1", REF_FIELDS), ref2=read("ref2", REF_FIELDS))
    variants["not_truncated"] = {i: v for i, v in variants["adjudicated"].items() if not trunc.get(i, True)}
    out = {}
    for name, ref in variants.items():
        out[name] = {}
        for ft, pos, rf in (("code", "released", "code_used"), ("main_data_access", "open", "data_obtained"),
                            ("weights_release", "yes", "weights_used")):
            pairs = [(by[i][ft] == pos, r[rf] == "yes") for i, r in sorted(ref.items())
                     if by[i][ft] != "unclear" and r[rf] != "unclear"]
            out[name][rf] = binary_agreement(pairs)
    out["refuted"] = any((v[rf]["kappa"] or 0) >= 0.40 for k, v in out.items() if k != "refuted"
                         for rf in ("code_used", "data_obtained"))
    return out


def main() -> dict:
    keys = json.loads(KEYS.read_text(encoding="utf-8"))
    final = final_codes()
    res = dict(F1=f1(keys, final))
    res.update(f2_f4(icml_units()))
    res["F5"] = f5(keys, final)
    (HERE / "results" / "p9_adversarial.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    return res


if __name__ == "__main__":
    print(json.dumps(main(), indent=1)[:3000])
