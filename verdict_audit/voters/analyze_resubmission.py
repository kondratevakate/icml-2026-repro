"""Protocol 3 of verdict_audit/voters/PROTOCOL.md: R1, R2, R4 and the exploratory analysis.

Writes aggregates only to results/resubmission.json (+ resubmission_manifest.json).
"""
from __future__ import annotations

import csv
import itertools
import json
import re
import subprocess
import sys
import time
from collections import Counter, defaultdict

import numpy as np

import analyze_voters as V
import common as C
import resubmission_events as R
from analyze_exposure import ts

A = C.A
OUT = C.HERE / "results"
TEXT_PATHS = ("logbook.json", "README.md")
FIXABLE = re.compile(r"\b(toy|small[- ]scale|subset|not run|missing|incomplete)\b", re.IGNORECASE)
DECISIVE = ("verified", "falsified")


def is_text_path(path: str) -> bool:
    return path in TEXT_PATHS or path.startswith("pages/")


def change_type(t1: dict | None, t2: dict | None) -> tuple[str, int]:
    """('text_only' | 'artifact' | 'missing_tree', number of changed paths)."""
    if t1 is None or t2 is None:
        return "missing_tree", 0
    changed = [p for p in set(t1) | set(t2) if t1.get(p) != t2.get(p)]
    return ("text_only" if all(is_text_path(p) for p in changed) else "artifact"), len(changed)


def shifts(v1: str, v2: str) -> dict:
    return dict(up=int(v1 != "verified" and v2 == "verified"), down=int(v1 == "verified" and v2 != "verified"),
                up_d=int(v1 not in DECISIVE and v2 in DECISIVE), down_d=int(v1 in DECISIVE and v2 not in DECISIVE),
                up_f=int(v1 != "falsified" and v2 == "falsified"), down_f=int(v1 == "falsified" and v2 != "falsified"))


def team_sums(rows: list[dict], up: str, down: str) -> tuple[list[str], np.ndarray]:
    agg = defaultdict(lambda: np.zeros(2))
    for r in rows:
        agg[r["team"]] += [r[up] - r[down], 1]
    teams = sorted(agg)
    return teams, np.array([agg[t] for t in teams]).reshape(-1, 2)


def boot_draws(n: int, rng: np.random.Generator) -> np.ndarray:
    return rng.integers(0, n, (A.B, n)) if n else np.zeros((A.B, 0), int)


def net_shift(rows: list[dict], up="up", down="down", seed=A.SEED) -> dict:
    teams, arr = team_sums(rows, up, down)
    if not len(arr) or arr[:, 1].sum() == 0:
        return dict(n_claims=0, n_teams=0, net_shift=None, ci=None, draws=None)
    draws = boot_draws(len(arr), np.random.default_rng(seed))
    sums = arr[draws].sum(1)
    boot = sums[:, 0] / sums[:, 1]
    return dict(n_claims=int(arr[:, 1].sum()), n_teams=len(teams),
                up_rate=float(np.mean([r[up] for r in rows])), down_rate=float(np.mean([r[down] for r in rows])),
                net_shift=float(arr[:, 0].sum() / arr[:, 1].sum()),
                ci=[float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))], draws=boot)


def difference(a: dict, b: dict) -> dict:
    if a["draws"] is None or b["draws"] is None:
        return dict(diff=None, ci=None)
    d = a["draws"] - b["draws"]
    return dict(diff=a["net_shift"] - b["net_shift"], ci=[float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))])


def public(x: dict) -> dict:
    return {k: v for k, v in x.items() if k != "draws"}


def build_rows(recs, anchored, ctime, trees) -> tuple[list[dict], Counter]:
    rows, counts = [], Counter()
    for ev in R.events(recs):
        counts[f"events_{ev['kind']}"] += 1
        sha1, sha2 = ev["r1"]["sha"], ev["r2"]["sha"]
        if ev["kind"] == "new_sha":
            c2 = ctime.get(ev["space"], {}).get(sha2)
            eligible = c2 is not None and c2 > ts(ev["r1"]["judged_at"])
            t1 = trees.get(f"{ev['space']}@{sha1}", {})
            t2 = trees.get(f"{ev['space']}@{sha2}", {})
            ctype, n_changed = change_type(t1.get("files") if t1.get("status") == "ok" else None,
                                           t2.get("files") if t2.get("status") == "ok" else None)
            counts[f"new_sha_{'eligible' if eligible else 'not_eligible'}"] += 1
            counts[f"new_sha_{ctype}"] += 1
            counts["new_sha_empty_diff"] += ctype == "text_only" and n_changed == 0
        else:
            eligible, ctype = True, "same_sha"
        claims = R.matched_claims(ev, anchored)
        counts[f"events_{ev['kind']}_with_match"] += bool(claims)
        for t, v1, v2, e1, e2 in claims:
            rows.append(dict(team=ev["team"], space=ev["space"], kind=ev["kind"], eligible=eligible, ctype=ctype,
                             claim=t, v1=v1, v2=v2, e1=e1, e2=e2, orid=ev["r2"]["orid"],
                             title=ev["r2"].get("paper_title", ""), fixable=bool(FIXABLE.search(e1 or "")),
                             **shifts(v1, v2)))
    return rows, counts


def first_published(recs, anchored, logbooks) -> dict[tuple[str, str], str]:
    by_sid = {r["space_id"]: r for r in logbooks}
    best: dict[tuple[str, str], tuple[str, str]] = {}
    for sid, rs in recs.items():
        if sid not in by_sid:
            continue
        for (j, _), rec in rs.items():
            if rec.get("orid") != by_sid[sid]["orid"]:
                continue
            for c in rec.get("claims") or []:
                t = A.norm(c.get("claim", ""))
                if t in anchored.get(rec["orid"], set()) and ((sid, t) not in best or j < best[(sid, t)][0]):
                    best[(sid, t)] = (j, A.category(c.get("verdict")))
    return {k: v for k, (_, v) in best.items()}


def r4(recs, logbooks, raw_anch, anchored, created, pages_manifest) -> dict:
    first = first_published(recs, anchored, logbooks)
    final = C.claim_raters(logbooks, anchored)
    swapped = {k: {s: first.get((s, k[1]), v) for s, v in raters.items()} for k, raters in final.items()}
    n_changed = sum(first.get((s, k[1]), v) != v for k, raters in final.items() for s, v in raters.items())
    multi_keys = [k for k, r in final.items() if len(r) >= 2]

    def kappa(src):
        groups = V.groups_by_paper({k: list(src[k].values()) for k in multi_keys})
        res = A.agreement(groups, np.random.default_rng(A.SEED), permute=False)
        return dict(kappa=res["kappa"], ci=res["kappa_ci"], n_claims=res["n_claims"])

    cand = {frozenset(p) for k in multi_keys for p in itertools.combinations(sorted(final[k]), 2)}
    feats = {}
    for s in {x for p in cand for x in p}:
        e = pages_manifest.get(s, {})
        if e.get("status") == "ok":
            f = V.shingles(V.logbook_text(C.read_pages(s, e))[0])
            if len(f) >= V.MIN_SHINGLES:
                feats[s] = f
    edges = {p for p in cand if all(x in feats for x in p) and V.jaccard(*(feats[x] for x in p)) >= V.TAU}

    def aggregate(src):
        keys, items, collapsed = [], [], {}
        for k, raters in src.items():
            comps = V.components(sorted(raters), edges) if len(raters) >= 2 else [[s] for s in raters]
            reps = [min(c, key=lambda s: V.created_key(created, s)) for c in comps]
            collapsed[k] = [raters[s] for s in reps]
            votes = [(A.namespace(s), 1 if raters[s] == "verified" else 0) for s in reps if raters[s] in DECISIVE]
            if votes:
                keys.append(k)
                items.append(votes)
        ds = V.dawid_skene(items)
        post = dict(zip(keys, ds["posterior"].tolist()))
        with (C.HERE / "reference_falsifications.csv").open(newline="", encoding="utf-8") as fh:
            ref = list(csv.DictReader(fh))
        per = {}
        for row in ref:
            k = (row["orid"], A.norm(raw_anch[row["orid"]][int(row["mapped_claim_index"])]["text"]))
            per[row["short_name"]] = dict(votes=dict(Counter(collapsed.get(k, []))),
                                          majority=V.majority(collapsed.get(k, [])),
                                          ds_posterior_verified=post.get(k) if ds["converged"] else None)
        return dict(ds_converged=ds["converged"], ds_iterations=ds["iterations"], prior_verified=ds["prior_verified"],
                    reference=per,
                    majority_agrees=sum(v["majority"] == "falsified" for v in per.values()),
                    ds_agrees=sum(v["ds_posterior_verified"] is not None and v["ds_posterior_verified"] < 0.5
                                  for v in per.values()))

    return dict(ratings_changed_by_first_published=n_changed,
                ratings_total=sum(len(r) for r in final.values()),
                kappa_final=kappa(final), kappa_first_published=kappa(swapped),
                q4_final=aggregate(final), q4_first_published=aggregate(swapped), copy_edges=len(edges))


def main() -> None:
    started = time.monotonic()
    out_file = OUT / "resubmission.json"
    if out_file.exists():
        sys.exit("results/resubmission.json exists; refusing to overwrite")
    hashes = C.check_inputs()
    recs = R.load_records()
    ctime = R.commit_times()
    trees = json.loads((C.DATA / "trees_manifest.json").read_text(encoding="utf-8"))
    pages_manifest = json.loads((C.DATA / "pages_manifest.json").read_text(encoding="utf-8"))
    logbooks, _, raw_anch, anchored, created = C.load()

    rows, counts = build_rows(recs, anchored, ctime, trees)
    new_e = [r for r in rows if r["kind"] == "new_sha" and r["eligible"]]
    same = [r for r in rows if r["kind"] == "same_sha"]

    # R1
    n_new, n_same = net_shift(new_e), net_shift(same)
    d = difference(n_new, n_same)
    r1 = dict(new_revision=public(n_new), same_revision=public(n_same), difference=d,
              label_toward_verified=bool(n_new["ci"] and n_new["ci"][0] > 0 and d["ci"] and d["ci"][0] > 0))
    for name, up, down in (("toward_decisive", "up_d", "down_d"), ("toward_falsified", "up_f", "down_f")):
        a, b = net_shift(new_e, up, down), net_shift(same, up, down)
        r1[name] = dict(new_revision=public(a), same_revision=public(b), difference=difference(a, b))
    event_counts = Counter(e["team"] for e in R.events(recs) if e["kind"] == "new_sha")
    top5 = {t for t, _ in event_counts.most_common(5)}
    a, b = net_shift([r for r in new_e if r["team"] not in top5]), net_shift([r for r in same if r["team"] not in top5])
    loo = [net_shift([r for r in new_e if r["team"] != t])["net_shift"] for t in sorted({r["team"] for r in new_e})]
    all_new = net_shift([r for r in rows if r["kind"] == "new_sha"])
    r1["sensitivity"] = dict(
        exclude_top5_teams=dict(new_revision=public(a), same_revision=public(b), difference=difference(a, b),
                                top5_share_of_new_events=sum(event_counts[t] for t in top5) / sum(event_counts.values())),
        leave_one_team_out=dict(min=float(np.nanmin([x for x in loo if x is not None])),
                                max=float(np.nanmax([x for x in loo if x is not None])), n_teams=len(loo)),
        non_eligible_added=dict(new_revision=public(all_new), difference=difference(all_new, n_same)))

    # R2
    txt = net_shift([r for r in new_e if r["ctype"] == "text_only"])
    art = net_shift([r for r in new_e if r["ctype"] == "artifact"])
    r2 = dict(text_only=public(txt), artifact=public(art), difference_text_minus_artifact=difference(txt, art),
              label_upgrades_without_new_artifacts=bool(txt["ci"] and txt["ci"][0] > 0),
              missing_tree_claims=sum(r["ctype"] == "missing_tree" for r in new_e),
              caveat="text-only changes can carry results produced outside the repository")

    # Exploratory
    base = [r for r in new_e if r["v1"] != "verified"]
    fx = net_shift([r for r in base if r["fixable"]], "up", "down")
    nf = net_shift([r for r in base if not r["fixable"]], "up", "down")
    explo = dict(note="up rate among claims not verified at the earlier judgement",
                 fixable_reason=dict(n=fx["n_claims"], up_rate=fx.get("up_rate")),
                 no_fixable_reason=dict(n=nf["n_claims"], up_rate=nf.get("up_rate")))

    results = dict(counts=dict(counts), matched_claims=dict(Counter((r["kind"], r["eligible"], r["ctype"]) for r in rows)
                                                         .items()).__repr__(),
                   R1=r1, R2=r2, R4=r4(recs, logbooks, raw_anch, anchored, created, pages_manifest), exploratory=explo)
    out_file.write_text(json.dumps(results, indent=2, default=float), encoding="utf-8")
    manifest = dict(protocol_sha256=A.sha256(C.HERE / "PROTOCOL.md"),
                    code_sha256={f: A.sha256(C.HERE / f) for f in
                                 ("analyze_resubmission.py", "resubmission_events.py", "analyze_voters.py", "common.py")},
                    trees_manifest_sha256=A.sha256(C.DATA / "trees_manifest.json"),
                    commits_manifest_sha256=A.sha256(C.DATA / "commits_manifest.json"),
                    git_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=C.ROOT, text=True).strip(),
                    inputs=hashes, seed=A.SEED, resamples=A.B, numpy=np.__version__, python=sys.version.split()[0],
                    seconds=round(time.monotonic() - started, 1), results_sha256=A.sha256(out_file))
    (OUT / "resubmission_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(dict(r1=r1["new_revision"]["net_shift"], d=r1["difference"]["diff"], label=r1["label_toward_verified"])),
          f"{manifest['seconds']}s")


if __name__ == "__main__":
    main()
