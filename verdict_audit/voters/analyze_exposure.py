"""Protocol 2 of verdict_audit/voters/PROTOCOL.md: exposure to published verdicts and first logbooks.

Writes aggregates only to results/exposure.json (+ exposure_manifest.json).
"""
from __future__ import annotations

import csv
import itertools
import json
import subprocess
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime

import numpy as np

import analyze_voters as V
import common as C

A = C.A
OUT = C.HERE / "results"
B_RES = A.B
SNAPSHOTS = {
    "verdicts_live.json": "76eeb935cc0972ec5b0ef9464a43d0920e00af1328db5460a71b61971351d544",
    "verdicts_latest_tmp.json": "b4bae4eb8a333dc98c3edd6865452ff96009b5b1a6dd1685d96a61bdaf31c239",
    "data/verdicts_confirmation_2026-10-17/verdicts.json": "652f1c162462d7ad142ff32e2bd13d5853e21c639681ef416971776fdd33c526",
}
D_BINS = [0, 0.5, 1, 2, 4, 7, float("inf")]
WEEK_CUTS = ["2026-07-22", "2026-07-29"]  # B judged <= 07-21 | 07-22..07-28 | >= 07-29
LATE_FROM = "2026-07-22"
QUALITY = {"low": 0, "medium": 1, "high": 2}
DAY = 86400.0


def ts(text: str) -> float:
    return datetime.fromisoformat(text.replace("Z", "+00:00")).timestamp()


def week_index(judged_at: str) -> int:
    return sum(judged_at[:10] >= cut for cut in WEEK_CUTS)


def d_bin(days: float) -> int:
    return next(i for i in range(len(D_BINS) - 1) if D_BINS[i] <= days < D_BINS[i + 1])


def unique_plurality(votes: list[str]):
    if not votes:
        return None
    counts = Counter(votes).most_common()
    if len(counts) > 1 and counts[0][1] == counts[1][1]:
        return None
    return counts[0][0]


# ---------- Mantel-Haenszel risk difference ----------

def mh_rd(table: np.ndarray) -> float | None:
    """table[s] = [n_exposed, agree_exposed, n_unexposed, agree_unexposed] per stratum."""
    ne, ae, nu, au = (table[:, i] for i in range(4))
    tot = ne + nu
    ok = (ne > 0) & (nu > 0)
    if not ok.any():
        return None
    num = ((ae * nu - au * ne) / np.where(ok, tot, 1))[ok].sum()
    den = ((ne * nu) / np.where(ok, tot, 1))[ok].sum()
    return float(num / den)


def pair_table(paper, stratum, exposed, agree, n_papers, n_strata) -> np.ndarray:
    """counts[paper, stratum, 4] for pairs with a defined exposure (exposed in {0, 1})."""
    t = np.zeros((n_papers, n_strata, 4))
    e = exposed == 1
    u = exposed == 0
    np.add.at(t, (paper[e], stratum[e], 0), 1)
    np.add.at(t, (paper[e], stratum[e], 1), agree[e])
    np.add.at(t, (paper[u], stratum[u], 2), 1)
    np.add.at(t, (paper[u], stratum[u], 3), agree[u])
    return t


def rd_with_ci(counts: np.ndarray, rng: np.random.Generator, draws=None) -> dict:
    est = mh_rd(counts.sum(0))
    n = len(counts)
    if draws is None:
        draws = [rng.integers(0, n, n) for _ in range(B_RES)]
    boot = np.array([v for v in (mh_rd(counts[d].sum(0)) for d in draws) if v is not None])
    tot = counts.sum((0, 1))
    return dict(rd=est, ci=[float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))] if len(boot) else None,
                n_exposed=int(tot[0]), n_unexposed=int(tot[2]),
                agree_exposed=float(tot[1] / tot[0]) if tot[0] else None,
                agree_unexposed=float(tot[3] / tot[2]) if tot[2] else None)


# ---------- agreement measures (exploratory) ----------

def ac1_and_specific(claims: list[list[str]]) -> dict:
    cats = sorted({r for c in claims for r in c})
    counts = np.array([[c.count(k) for k in cats] for c in claims], float)
    n = counts.sum(1)
    pa = float(np.mean((counts * (counts - 1)).sum(1) / (n * (n - 1))))
    pi = (counts / n[:, None]).mean(0)
    pe = float((pi * (1 - pi)).sum() / (len(cats) - 1))
    specific = {k: float((counts[:, j] * (counts[:, j] - 1)).sum() / (counts[:, j] * (n - 1)).sum())
                for j, k in enumerate(cats) if counts[:, j].sum()}
    return dict(categories=cats, pa=pa, pe_ac1=pe, ac1=(pa - pe) / (1 - pe), specific_agreement=specific)


# ---------- main ----------

def main() -> None:
    started = time.monotonic()
    out_file = OUT / "exposure.json"
    if out_file.exists():
        sys.exit("results/exposure.json exists; refusing to overwrite")
    hashes = C.check_inputs()
    for name, expected in SNAPSHOTS.items():
        got = A.sha256(C.ROOT / name)
        if got != expected:
            sys.exit(f"Hash mismatch for {name}: {got}")
        hashes[name] = got
    commits_raw = json.loads((C.DATA / "commits_manifest.json").read_text(encoding="utf-8"))
    pages_manifest = json.loads((C.DATA / "pages_manifest.json").read_text(encoding="utf-8"))
    logbooks, _, raw_anch, anchored, _ = C.load()
    by_sid = {r["space_id"]: r for r in logbooks}
    multi = C.multi_rated_claims(logbooks, anchored)

    # commit times
    ctime = {s: {cid: ts(t) for cid, t in e["commits"]} for s, e in commits_raw.items() if e.get("status") == "ok"}
    start = {s: min(v.values()) for s, v in ctime.items() if v}
    c_final = {s: ctime[s].get(by_sid[s]["sha"]) for s in by_sid if s in ctime}

    # first-published verdict per (space, claim): earliest record containing the anchored text
    first_pub: dict[tuple[str, str], tuple[float, str, str]] = {}
    for name in SNAPSHOTS:
        for sid, rec in A.load_json(C.ROOT / name).items():
            if sid not in by_sid or rec.get("orid") != by_sid[sid]["orid"]:
                continue
            j = ts(rec["judged_at"])
            for c in rec.get("claims") or []:
                t = A.norm(c.get("claim", ""))
                key = (sid, t)
                if t in anchored.get(rec["orid"], set()) and (key not in first_pub or j < first_pub[key][0]):
                    first_pub[key] = (j, A.category(c.get("verdict")), rec["sha"])

    # copy edges from Protocol 1 (same functions, same threshold)
    cand = {frozenset(p) for r in multi.values() for p in itertools.combinations(sorted(r), 2)}
    feats = {}
    for s in {x for p in cand for x in p}:
        e = pages_manifest.get(s, {})
        if e.get("status") == "ok":
            f = V.shingles(V.logbook_text(C.read_pages(s, e))[0])
            if len(f) >= V.MIN_SHINGLES:
                feats[s] = f
    edges = {p for p in cand if all(x in feats for x in p) and V.jaccard(*(feats[x] for x in p)) >= V.TAU}

    # build ordered pairs
    papers = sorted({k[0] for k in multi})
    pidx = {o: i for i, o in enumerate(papers)}
    n_strata = (len(D_BINS) - 1) * (len(WEEK_CUTS) + 1)
    excl = Counter()
    rows = []
    recs = {}  # A first-published record id -> (r_A, latency)
    for (orid, t), raters in multi.items():
        for a, b in itertools.permutations(sorted(raters), 2):
            if a not in start or b not in start:
                excl["no_commit_history_unordered_pairs"] += a < b  # permutations visit each pair twice
                continue
            if not start[a] < start[b]:
                continue
            if c_final.get(b) is None:
                excl["b_sha_not_in_history"] += 1
                continue
            j, ref, sha_a = first_pub[(a, t)]
            r_a = ctime[a].get(sha_a)
            if r_a is None:
                excl["a_sha_not_in_history"] += 1
                continue
            c_b = c_final[b]
            d = (c_b - r_a) / DAY
            if d < 0:
                excl["d_negative"] += 1
                continue
            rec_id = recs.setdefault((a, sha_a, j), (len(recs), r_a, j - r_a))[0]
            vb = raters[b]
            rows.append(dict(
                paper=pidx[orid], stratum=d_bin(d) * (len(WEEK_CUTS) + 1) + week_index(by_sid[b]["judged_at"]),
                rec=rec_id, c_b=c_b, s_b=start[b], v_a=j, ref=ref,
                agree=int(vb == ref),
                agree_bin=int((vb in ("verified", "falsified")) == (ref in ("verified", "falsified"))),
                copy=int(frozenset((a, b)) in edges),
                late=int(j >= ts(LATE_FROM + "T00:00:00+00:00") and by_sid[b]["judged_at"][:10] >= LATE_FROM),
                org=int(C.ORGANISER in (A.namespace(a), A.namespace(b))),
                org_a=int(A.namespace(a) == C.ORGANISER),
                author=int(C.AUTHOR in (A.namespace(a), A.namespace(b)))))
    P = {k: np.array([r[k] for r in rows]) for k in rows[0]}
    exposed = (P["c_b"] > P["v_a"]).astype(int)
    strict = np.where(P["s_b"] > P["v_a"], 1, np.where(P["c_b"] <= P["v_a"], 0, -1))
    n_p = len(papers)

    def run(mask, exp=exposed, agree=P["agree"], draws=None):
        return rd_with_ci(pair_table(P["paper"][mask], P["stratum"][mask], exp[mask], agree[mask], n_p, n_strata),
                          np.random.default_rng(A.SEED), draws)

    base = P["copy"] == 0
    rng = np.random.default_rng(A.SEED)
    draws = [rng.integers(0, n_p, n_p) for _ in range(B_RES)]
    h1 = run(base, draws=draws)

    # latency permutation within same UTC date of r_A
    rec_r = np.array([v[1] for v in sorted(recs.values())])
    rec_l = np.array([v[2] for v in sorted(recs.values())])
    day = np.floor(rec_r / DAY).astype(int)
    groups = [np.where(day == d)[0] for d in np.unique(day)]
    prng = np.random.default_rng(A.SEED)
    null = []
    m = base
    for _ in range(B_RES):
        perm_l = rec_l.copy()
        for g in groups:
            perm_l[g] = rec_l[prng.permutation(g)]
        v_perm = rec_r[P["rec"]] + perm_l[P["rec"]]
        e_perm = (P["c_b"] > v_perm).astype(int)
        null.append(mh_rd(pair_table(P["paper"][m], P["stratum"][m], e_perm[m], P["agree"][m], n_p, n_strata).sum((0))))
    null = np.array([x for x in null if x is not None])
    h1["permutation"] = dict(n=len(null), null_mean=float(null.mean()), null_q975=float(np.percentile(null, 97.5)),
                             p_one_sided=float((1 + (null >= h1["rd"]).sum()) / (len(null) + 1)))
    h1["label_herding_on_published_verdicts"] = bool(h1["ci"] and h1["ci"][0] > 0 and h1["permutation"]["p_one_sided"] < 0.05)
    h1["sensitivity"] = dict(
        strict_exposure=run(base & (strict >= 0), exp=strict, draws=draws),
        both_judged_from_2026_07_22=run(base & (P["late"] == 1), draws=draws),
        binary_decisive=run(base, agree=P["agree_bin"], draws=draws),
        copy_pairs_included=run(np.ones(len(rows), bool), draws=draws),
        exclude_organiser=run(base & (P["org"] == 0), draws=draws),
        exclude_author=run(base & (P["author"] == 0), draws=draws),
    )

    # H2
    ver = run(base & (P["ref"] == "verified"), draws=draws)
    fal = run(base & (P["ref"] == "falsified"), draws=draws)
    def ref_table(cat):
        mk = base & (P["ref"] == cat)
        return pair_table(P["paper"][mk], P["stratum"][mk], exposed[mk], P["agree"][mk], n_p, n_strata)

    tv_full, tf_full = ref_table("verified"), ref_table("falsified")
    diffs = [x - y for x, y in ((mh_rd(tv_full[d].sum(0)), mh_rd(tf_full[d].sum(0))) for d in draws)
             if x is not None and y is not None]
    h2 = dict(ref_verified=ver, ref_falsified=fal,
              difference=dict(rd=(ver["rd"] - fal["rd"]) if None not in (ver["rd"], fal["rd"]) else None,
                              ci=[float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))] if diffs else None))

    # H3
    per_paper = defaultdict(lambda: [0.0, 0])
    skipped = Counter()
    for (orid, t), raters in multi.items():
        dated = sorted((s for s in raters if s in start), key=lambda s: (start[s], s))
        if len(dated) < 3:
            skipped["lt3_dated"] += 1
            continue
        def agree_x(x):
            p = unique_plurality([raters[y] for y in dated if y != x])
            return None if p is None else int(raters[x] == p)
        af = agree_x(dated[0])
        others = [v for v in (agree_x(x) for x in dated[1:]) if v is not None]
        if af is None or not others:
            skipped["no_plurality"] += 1
            continue
        per_paper[pidx[orid]][0] += af - np.mean(others)
        per_paper[pidx[orid]][1] += 1
    h3_arr = np.array([per_paper[i] if i in per_paper else [0.0, 0] for i in range(n_p)])
    h3_stat = float(h3_arr[:, 0].sum() / h3_arr[:, 1].sum())
    h3_boot = [h3_arr[d, 0].sum() / h3_arr[d, 1].sum() for d in draws if h3_arr[d, 1].sum()]
    h3 = dict(n_claims=int(h3_arr[:, 1].sum()), skipped=dict(skipped), first_minus_nonfirst=h3_stat,
              ci=[float(np.percentile(h3_boot, 2.5)), float(np.percentile(h3_boot, 97.5))])

    # H4
    q = np.zeros((n_p, 2))  # sum of quality diff, n
    q_fal = np.zeros((n_p, 2))
    outcome, outcome_fal = Counter(), Counter()
    ref_rows = {}
    with (C.HERE / "reference_falsifications.csv").open(newline="", encoding="utf-8") as fh:
        ref_claims = {(r["orid"], A.norm(raw_anch[r["orid"]][int(r["mapped_claim_index"])]["text"])): r["short_name"]
                      for r in csv.DictReader(fh)}
    for (orid, t), raters in multi.items():
        dated = sorted((s for s in raters if s in start), key=lambda s: (start[s], s))
        if len(dated) < 2:
            continue
        f = dated[0]
        plural = unique_plurality(list(raters.values()))
        for b in dated[1:]:
            if raters[b] == raters[f]:
                continue
            qd = QUALITY.get(by_sid[b].get("quality"), np.nan) - QUALITY.get(by_sid[f].get("quality"), np.nan)
            res = "tie" if plural is None else "later" if plural == raters[b] else "first" if plural == raters[f] else "neither"
            for arr, cnt, cond in ((q, outcome, True), (q_fal, outcome_fal, raters[b] == "falsified")):
                if cond:
                    cnt[res] += 1
                    if not np.isnan(qd):
                        arr[pidx[orid]] += [qd, 1]
        if (orid, t) in ref_claims:
            ref_rows[ref_claims[(orid, t)]] = dict(
                first_verdict=raters[f], first_quality=by_sid[f].get("quality"),
                later=[dict(verdict=raters[b], quality=by_sid[b].get("quality"), order=i + 2)
                       for i, b in enumerate(dated[1:])])

    def qsummary(arr, cnt):
        boot = [arr[d, 0].sum() / arr[d, 1].sum() for d in draws if arr[d, 1].sum()]
        n = int(arr[:, 1].sum())
        return dict(n_pairs=sum(cnt.values()), n_with_quality=n,
                    mean_quality_later_minus_first=float(arr[:, 0].sum() / n) if n else None,
                    ci=[float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))] if boot else None,
                    final_plurality={k: cnt[k] / sum(cnt.values()) for k in ("later", "first", "neither", "tie")})
    h4 = dict(all_disagreeing=qsummary(q, outcome), later_falsified=qsummary(q_fal, outcome_fal), reference_claims=ref_rows)

    # exploratory
    org = run(base & (P["org_a"] == 1), draws=draws)
    q2_claims = [list(r.values()) for r in multi.values()]
    explo = dict(h1_a_is_organiser=org, protocol1_q2_prevalence_robust=ac1_and_specific(q2_claims))

    # H5
    syn_path = C.HERE / "synthesis_logbooks.csv"
    h5 = dict(status="pending: synthesis_logbooks.csv not provided yet") if not syn_path.exists() else None

    lat_h = rec_l / 3600
    results = dict(
        pairs=dict(n_pairs=len(rows), n_pairs_no_copy=int(base.sum()), excluded=dict(excl),
                   exposed_share=float(exposed[base].mean()),
                   commit_history=dict(Counter(e.get("status") for e in commits_raw.values())),
                   judge_latency_hours=dict(q25=float(np.percentile(lat_h, 25)), median=float(np.median(lat_h)),
                                            q75=float(np.percentile(lat_h, 75)), n_records=len(rec_l)),
                   copy_edges=len(edges)),
        H1=h1, H2=h2, H3=h3, H4=h4, H5=h5, exploratory=explo)
    out_file.write_text(json.dumps(results, indent=2, default=float), encoding="utf-8")
    manifest = dict(protocol_sha256=A.sha256(C.HERE / "PROTOCOL.md"),
                    code_sha256={f: A.sha256(C.HERE / f) for f in ("analyze_exposure.py", "analyze_voters.py", "common.py")},
                    commits_manifest_sha256=A.sha256(C.DATA / "commits_manifest.json"),
                    pages_manifest_sha256=A.sha256(C.DATA / "pages_manifest.json"),
                    git_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=C.ROOT, text=True).strip(),
                    inputs=hashes, seed=A.SEED, resamples=B_RES, numpy=np.__version__, python=sys.version.split()[0],
                    seconds=round(time.monotonic() - started, 1), results_sha256=A.sha256(out_file))
    (OUT / "exposure_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(dict(rd=h1["rd"], ci=h1["ci"], p=h1["permutation"]["p_one_sided"],
                          label=h1["label_herding_on_published_verdicts"])), f"{manifest['seconds']}s")


if __name__ == "__main__":
    main()
