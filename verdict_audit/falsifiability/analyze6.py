"""Protocol 6: agreement, groups, and descriptive quantities D1-D5 with bootstrap CIs.

Usage: python analyze6.py agreement   # kappa and disagreement list
       python analyze6.py results     # needs adjudication.csv for all disagreements
"""
import csv
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE.parent))
import analyze as A  # noqa: E402

SNAP = ROOT / "data/verdicts_confirmation_2026-10-17/verdicts.json"
SNAP_SHA = "652f1c162462d7ad142ff32e2bd13d5853e21c639681ef416971776fdd33c526"
SEED, B = 20261017, 2000
KATE = "kondratevakate"


def read(p):
    with Path(p).open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def kappa(a, b):
    n = len(a)
    po = sum(x == y for x, y in zip(a, b)) / n
    ca, cb = Counter(a), Counter(b)
    pe = sum(ca[k] * cb[k] for k in set(ca) | set(cb)) / n ** 2
    return dict(n=n, po=round(po, 4), kappa=round((po - pe) / (1 - pe), 4))


def consensus():
    a = {r["item"]: r["label"] for r in read(HERE / "codes_A.csv")}
    b = {r["item"]: r["label"] for r in read(HERE / "codes_B.csv")}
    assert set(a) == set(b) and len(a) == 255
    adj_p = HERE / "adjudication.csv"
    adj = {r["item"]: r["label"] for r in read(adj_p)} if adj_p.exists() else {}
    items = sorted(a, key=int)
    dis = [i for i in items if a[i] != b[i]]
    cons = {i: a[i] if a[i] == b[i] else adj.get(i) for i in items}
    return a, b, cons, dis, [i for i in dis if i not in adj]


def agreement():
    a, b, _, dis, _ = consensus()
    items = sorted(a, key=int)
    sheet = {r["item"]: r for r in read(HERE / "sheets/candidates.csv")}
    print(json.dumps(kappa([a[i] for i in items], [b[i] for i in items])), "disagreements", len(dis))
    for i in dis:
        print(f"[{i}] A={a[i]} B={b[i]} :: {sheet[i]['title'][:110]} | {sheet[i]['abstract'][-260:]}")


def boot_ci(values_by_paper, stat, rng):
    n = len(values_by_paper)
    draws = [stat([values_by_paper[k] for k in rng.integers(0, n, n)]) for _ in range(B)]
    return [round(float(x), 4) for x in np.percentile(draws, [2.5, 97.5])]


def quantities(papers, logs_by_paper, rng):
    """papers: list of orids in the group; logs_by_paper: orid -> list of logbooks."""
    att = [o for o in papers if logs_by_paper.get(o)]
    rows = []
    for o in att:
        L = logs_by_paper[o]
        teams = len({A.namespace(r["space_id"]) for r in L})
        verdicts = [A.category(c.get("verdict")) for r in L for c in r.get("claims") or []]
        rows.append(dict(
            teams=teams,
            full=any(r.get("claims") and all(A.category(c.get("verdict")) == "verified" for c in r["claims"]) for r in L),
            any_f="falsified" in verdicts,
            undecided=not ({"verified", "falsified"} & set(verdicts)),
            counts=Counter(verdicts)))
    out = dict(papers=len(papers), attempted=len(att), D1_attempt_rate=round(len(att) / len(papers), 4))
    out["D1_ci"] = boot_ci([int(bool(logs_by_paper.get(o))) for o in papers], np.mean, rng)
    for key, f in [("D2_mean_teams", lambda r: r["teams"]), ("D2_share_ge2", lambda r: r["teams"] >= 2),
                   ("D2_share_ge3", lambda r: r["teams"] >= 3), ("D3_fully_verified", lambda r: r["full"]),
                   ("D3_any_falsified", lambda r: r["any_f"]), ("D3_undecided", lambda r: r["undecided"])]:
        vals = [float(f(r)) for r in rows]
        out[key] = round(float(np.mean(vals)), 4)
        out[key + "_ci"] = boot_ci(vals, np.mean, rng)
    tot = sum((r["counts"] for r in rows), Counter())
    n = sum(tot.values())
    out["D4_verdict_mix"] = {k: round(tot[k] / n, 4) for k in ("verified", "falsified", "toy", "inconclusive")}
    out["D4_claims"] = n
    for rho in (0.23, 0.15):
        sb = [r["teams"] * rho / (1 + (r["teams"] - 1) * rho) for r in rows]
        out[f"D5_mean_reliability_rho{rho}"] = round(float(np.mean(sb)), 4)
    out["_rows"] = rows
    return out


def diff(m, o, key, rng):
    a = [float(r[key]) if not isinstance(r[key], bool) else float(r[key]) for r in m["_rows"]]
    b = [float(r[key]) for r in o["_rows"]]
    d = [np.mean([a[k] for k in rng.integers(0, len(a), len(a))]) - np.mean([b[k] for k in rng.integers(0, len(b), len(b))])
         for _ in range(B)]
    return dict(diff=round(float(np.mean(a) - np.mean(b)), 4), ci=[round(float(x), 4) for x in np.percentile(d, [2.5, 97.5])])


def results():
    assert hashlib.sha256(SNAP.read_bytes()).hexdigest() == SNAP_SHA
    a, b, cons, dis, missing = consensus()
    if missing:
        sys.exit(f"unadjudicated items: {missing}")
    sheet = {r["item"]: r["orid"] for r in read(HERE / "sheets/candidates.csv")}
    items = sorted(a, key=int)
    med = {sheet[i] for i in items if cons[i] in ("clinical", "life_science")}
    clin = {sheet[i] for i in items if cons[i] == "clinical"}
    cand = set(sheet.values())
    broad = A.load_orids(ROOT / "audit/public_repro_audit.csv")
    allp = {p["orid"] for p in json.loads((ROOT / "papers.json").read_text(encoding="utf-8-sig"))["papers"]}
    other = allp - cand - broad
    logs, _ = A.dedupe(list(A.load_json(SNAP).values()))
    res = dict(agreement=kappa([a[i] for i in items], [b[i] for i in items]), disagreements=len(dis),
               consensus_counts=dict(Counter(cons.values())), medical_n=len(med), clinical_n=len(clin),
               other_n=len(other), excluded_ambiguous=len(allp) - len(med) - len(other))
    for tag, keep in [("without_kate", lambda r: A.namespace(r["space_id"]).lower() != KATE), ("with_kate", lambda r: True)]:
        by = defaultdict(list)
        for r in logs:
            if keep(r):
                by[r["orid"]].append(r)
        rng = np.random.default_rng(SEED)
        qm, qc, qo = quantities(sorted(med), by, rng), quantities(sorted(clin), by, rng), quantities(sorted(other), by, rng)
        diffs = {k: diff(qm, qo, f, rng) for k, f in [("D2_share_ge3", "teams"), ("D3_fully_verified", "full"),
                                                       ("D3_undecided", "undecided")]}
        diffs["D2_share_ge3"] = None  # teams is a count; the share difference is computed below
        ge3 = lambda q: [float(r["teams"] >= 3) for r in q["_rows"]]
        am, ao = ge3(qm), ge3(qo)
        d = [np.mean([am[k] for k in rng.integers(0, len(am), len(am))]) - np.mean([ao[k] for k in rng.integers(0, len(ao), len(ao))]) for _ in range(B)]
        diffs["D2_share_ge3"] = dict(diff=round(float(np.mean(am) - np.mean(ao)), 4), ci=[round(float(x), 4) for x in np.percentile(d, [2.5, 97.5])])
        for q in (qm, qc, qo):
            q.pop("_rows")
        res[tag] = dict(medical=qm, clinical_only=qc, other=qo, medical_minus_other=diffs)
    rho = 0.23
    res["teams_needed"] = {str(t): round(t * (1 - rho) / (rho * (1 - t)), 1) for t in (0.7, 0.8)}
    (HERE / "results6.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    {"agreement": agreement, "results": results}[sys.argv[1]]()
