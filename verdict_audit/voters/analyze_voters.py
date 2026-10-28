"""Q1, Q2 and Q4 of verdict_audit/voters/PROTOCOL.md, plus the Q3 regex counts.

Reads frozen inputs and downloaded pages (hash-checked), writes aggregates only to results/.
"""
from __future__ import annotations

import csv
import hashlib
import itertools
import json
import subprocess
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime

import numpy as np

import common as C
from patterns import COMPILED

A = C.A
OUT = C.HERE / "results"
TAU = 0.5
TAU_LADDER = [0.6, 0.7, 0.8, 0.9]
K = 5
MIN_SHINGLES = 50
N_CONTROL = 20000
GAP_BINS = [("lt_1d", 0, 1), ("1_7d", 1, 7), ("gt_7d", 7, float("inf"))]


# ---------- text ----------

def is_claim_page(path: str) -> bool:
    parts = path.split("/")
    return len(parts) >= 2 and parts[0] == "pages" and parts[1].lower().startswith("claim")


def logbook_text(pages: dict[str, str]) -> tuple[str, bool]:
    """Claim pages in sorted path order; all pages if there is none. Returns (text, had_claim_pages)."""
    md = {p: t for p, t in pages.items() if p.startswith("pages/")}
    claim = {p: t for p, t in md.items() if is_claim_page(p)}
    use = claim or md
    return "\n".join(use[p] for p in sorted(use)), bool(claim)


def shingles(text: str, k: int = K) -> np.ndarray:
    """Sorted unique uint64 hashes (blake2b, stable across runs) of word k-shingles."""
    tokens = A.norm(text).split()
    hs = [int.from_bytes(hashlib.blake2b(" ".join(tokens[i:i + k]).encode("utf-8"), digest_size=8).digest(), "little")
          for i in range(len(tokens) - k + 1)]
    return np.unique(np.array(hs, dtype=np.uint64))


def _inter(a: np.ndarray, b: np.ndarray) -> int:
    return len(np.intersect1d(a, b, assume_unique=True))


def jaccard(a: np.ndarray, b: np.ndarray) -> float:
    if not len(a) and not len(b):
        return 0.0
    inter = _inter(a, b)
    return inter / (len(a) + len(b) - inter)


def containment(a: np.ndarray, b: np.ndarray) -> float:
    return _inter(a, b) / min(len(a), len(b)) if len(a) and len(b) else 0.0


def choose_tau(control: np.ndarray) -> float | None:
    q = float(np.quantile(control, 0.999))
    if q < TAU:
        return TAU
    return next((t for t in TAU_LADDER if t > q), None)


# ---------- copy collapse ----------

def created_key(created: dict, sid: str):
    ts = created.get(sid)
    return (ts is None, ts or "", sid)


def components(nodes: list[str], edges: set[frozenset]) -> list[list[str]]:
    parent = {n: n for n in nodes}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a, b in itertools.combinations(nodes, 2):
        if frozenset((a, b)) in edges:
            parent[find(a)] = find(b)
    groups = defaultdict(list)
    for n in nodes:
        groups[find(n)].append(n)
    return list(groups.values())


def collapse(raters: dict[str, str], edges: set[frozenset], created: dict) -> list[str]:
    """One rating per copy component: the verdict of its earliest-created logbook."""
    out = []
    for comp in components(sorted(raters), edges):
        out.append(raters[min(comp, key=lambda s: created_key(created, s))])
    return out


def groups_by_paper(claims: dict[tuple[str, str], list[str]]) -> dict[str, list[list[str]]]:
    g = defaultdict(list)
    for (orid, _), ratings in claims.items():
        if len(ratings) >= 2:
            g[orid].append(ratings)
    return dict(g)


def spearman_brown(r: float, target: float) -> float | None:
    if not (0 < r < 1):
        return None
    return float(target * (1 - r) / (r * (1 - target)))


def teams_needed(kappa: float, ci: list[float]) -> dict:
    return {f"R_{t}": dict(point=spearman_brown(kappa, t), at_ci_low=spearman_brown(ci[0], t),
                           at_ci_high=spearman_brown(ci[1], t)) for t in (0.7, 0.8)}


def days_between(a: str | None, b: str | None) -> float | None:
    if not a or not b:
        return None
    pa, pb = (datetime.fromisoformat(x.replace("Z", "+00:00")) for x in (a, b))
    return abs((pa - pb).total_seconds()) / 86400


# ---------- Dawid-Skene (binary) ----------

def dawid_skene(items: list[list[tuple[str, int]]], tol: float = 1e-6, max_iter: int = 1000) -> dict:
    """items[i] = [(annotator, label 1=verified / 0=falsified)]. MAP EM with Beta(2,2) pseudo-counts."""
    annots = sorted({a for it in items for a, _ in it})
    aidx = {a: j for j, a in enumerate(annots)}
    rows = np.array([i for i, it in enumerate(items) for _ in it])
    cols = np.array([aidx[a] for it in items for a, _ in it])
    labs = np.array([y for it in items for _, y in it], float)
    n, m = len(items), len(annots)
    pos = np.bincount(rows, weights=labs, minlength=n)
    tot = np.bincount(rows, minlength=n).astype(float)
    mu = np.where(pos * 2 > tot, 1.0, np.where(pos * 2 < tot, 0.0, 0.5))
    converged, it = False, 0
    for it in range(1, max_iter + 1):
        w1, w0 = mu[rows], 1 - mu[rows]
        sens = (np.bincount(cols, w1 * labs, m) + 1) / (np.bincount(cols, w1, m) + 2)
        spec = (np.bincount(cols, w0 * (1 - labs), m) + 1) / (np.bincount(cols, w0, m) + 2)
        prior = (mu.sum() + 1) / (n + 2)
        l1 = np.bincount(rows, labs * np.log(sens[cols]) + (1 - labs) * np.log(1 - sens[cols]), n) + np.log(prior)
        l0 = np.bincount(rows, (1 - labs) * np.log(spec[cols]) + labs * np.log(1 - spec[cols]), n) + np.log(1 - prior)
        new = 1 / (1 + np.exp(l0 - l1))
        delta = float(np.max(np.abs(new - mu)))
        mu = new
        if delta < tol:
            converged = True
            break
    return dict(posterior=mu, converged=converged, iterations=it, n_items=n, n_annotators=m,
                prior_verified=float(prior))


def majority(ratings: list[str]) -> str:
    v, f = ratings.count("verified"), ratings.count("falsified")
    return "verified" if v > f else "falsified" if f > v else ("tie" if v else "no_decisive")


# ---------- pipeline ----------

def build_q2(logbooks, anchored, created, edges, rng_seed=A.SEED) -> dict:
    multi = C.multi_rated_claims(logbooks, anchored)
    nominal = {k: list(v.values()) for k, v in multi.items()}
    collapsed = {k: collapse(v, edges, created) for k, v in multi.items()}
    n_nom = np.array([len(v) for v in nominal.values()])
    n_eff = np.array([len(collapsed[k]) for k in nominal])
    unc = A.agreement(groups_by_paper(nominal), np.random.default_rng(rng_seed), permute=False)
    col = A.agreement(groups_by_paper(collapsed), np.random.default_rng(rng_seed), permute=False)
    out = dict(
        n_claims=len(multi),
        nominal_teams=dict(mean=float(n_nom.mean()), median=float(np.median(n_nom)), dist=dict(Counter(n_nom.tolist()))),
        effective_teams=dict(mean=float(n_eff.mean()), median=float(np.median(n_eff)), dist=dict(Counter(n_eff.tolist()))),
        relative_drop_mean_teams=float(1 - n_eff.mean() / n_nom.mean()),
        claims_with_fewer_than_2_effective=int((n_eff < 2).sum()),
        kappa_uncollapsed=unc, kappa_collapsed=col,
    )
    for name, res in (("uncollapsed", unc), ("collapsed", col)):
        if res.get("n_claims"):
            out[f"teams_needed_{name}"] = teams_needed(res["kappa"], res["kappa_ci"])
    if unc.get("n_claims") and col.get("n_claims"):
        out["kappa_change"] = col["kappa"] - unc["kappa"]
    return out


def main() -> None:
    started = time.monotonic()
    OUT.mkdir(exist_ok=False)
    hashes = C.check_inputs()
    manifest = json.loads((C.DATA / "pages_manifest.json").read_text(encoding="utf-8"))
    logbooks, dropped, raw_anch, anchored, created = C.load()
    by_sid = {r["space_id"]: r for r in logbooks}
    multi = C.multi_rated_claims(logbooks, anchored)
    wanted = sorted(set().union(*[set(v) for v in multi.values()]))

    status = Counter(manifest.get(s, {}).get("status", "not_attempted") for s in wanted)
    sh, no_claim, too_short, pages_all = {}, 0, 0, {}
    for s in wanted:
        entry = manifest.get(s, {})
        if entry.get("status") != "ok":
            continue
        pages = C.read_pages(s, entry)
        pages_all[s] = pages
        text, had = logbook_text(pages)
        no_claim += not had
        feats = shingles(text)
        if len(feats) < MIN_SHINGLES:
            too_short += 1
            continue
        sh[s] = feats

    # candidate pairs (same paper, shared multi-rated claim, different teams by construction)
    cand = set()
    for raters in multi.values():
        for a, b in itertools.combinations(sorted(raters), 2):
            cand.add(frozenset((a, b)))
    tested = [p for p in cand if all(x in sh for x in p)]
    jac = {p: jaccard(*(sh[x] for x in p)) for p in tested}
    con = {p: containment(*(sh[x] for x in p)) for p in tested}

    # template control: random pairs on different papers
    rng = np.random.default_rng(A.SEED)
    pool = sorted(sh)
    control = []
    while len(control) < N_CONTROL:
        i, j = rng.integers(0, len(pool), 2)
        a, b = pool[i], pool[j]
        if by_sid[a]["orid"] != by_sid[b]["orid"]:
            control.append(jaccard(sh[a], sh[b]))
    control = np.array(control)
    tau = choose_tau(control)
    jv = np.array(list(jac.values())) if jac else np.array([0.0])

    def flagged(score: dict, t: float) -> set:
        return {p for p, v in score.items() if v >= t}

    def pair_stats(edges: set) -> dict:
        same, n = Counter(), Counter()
        for raters in multi.values():
            for a, b in itertools.combinations(sorted(raters), 2):
                p = frozenset((a, b))
                if p not in jac:
                    continue
                kind = "copy" if p in edges else "non_copy"
                n[kind] += 1
                same[kind] += raters[a] == raters[b]
        return {k: dict(n_claim_pairs=n[k], exact_agreement=same[k] / n[k] if n[k] else None) for k in ("copy", "non_copy")}

    def edge_summary(edges: set) -> dict:
        claims_hit = sum(any(frozenset(p) in edges for p in itertools.combinations(sorted(r), 2)) for r in multi.values())
        directed = sum(created.get(min(p)) != created.get(max(p)) and None not in (created.get(min(p)), created.get(max(p)))
                       for p in edges)
        return dict(n_copy_pairs=len(edges), share_of_tested_pairs=len(edges) / len(jac) if jac else None,
                    claims_with_copy_pair=claims_hit, share_of_multi_rated_claims=claims_hit / len(multi),
                    directed_pairs=int(directed), undirected_pairs=len(edges) - int(directed))

    q1 = dict(
        logbooks_wanted=len(wanted), download_status=dict(status), no_claim_pages=no_claim, too_short=too_short,
        logbooks_tested=len(sh), candidate_pairs=len(cand), tested_pairs=len(tested),
        control=dict(n=N_CONTROL, q50=float(np.quantile(control, 0.5)), q99=float(np.quantile(control, 0.99)),
                     q999=float(np.quantile(control, 0.999)), max=float(control.max())),
        tau_used=tau,
        jaccard_tested=dict(q50=float(np.quantile(jv, 0.5)), q90=float(np.quantile(jv, 0.9)),
                            q99=float(np.quantile(jv, 0.99)), share_ge_0_3=float((jv >= 0.3).mean()),
                            share_ge_0_5=float((jv >= 0.5).mean()), share_ge_0_8=float((jv >= 0.8).mean())),
    )
    results = dict(snapshot=dict(logbooks_dedup=len(logbooks), dropped_by_dedup=dropped, multi_rated_claims=len(multi),
                                 papers=len({k[0] for k in multi})))
    if tau is None:
        q1["evaluable"] = False
        edges = set()
    else:
        q1["evaluable"] = True
        edges = flagged(jac, tau)
        q1.update(primary=edge_summary(edges), verdict_agreement=pair_stats(edges),
                  sensitivity={f"jaccard_{t}": edge_summary(flagged(jac, t)) for t in (0.3, 0.8)}
                  | {f"containment_{tau}": edge_summary(flagged(con, tau))})
    results["Q1_copies"] = q1

    # Q2
    q2 = build_q2(logbooks, anchored, created, edges)
    gaps = {name: Counter() for name, *_ in GAP_BINS}
    for raters in multi.values():
        for a, b in itertools.combinations(sorted(raters), 2):
            d = days_between(created.get(a), created.get(b))
            if d is None:
                continue
            name = next(n for n, lo, hi in GAP_BINS if lo <= d < hi)
            is_copy = frozenset((a, b)) in edges
            agree = raters[a] == raters[b]
            gaps[name]["n_all"] += 1
            gaps[name]["agree_all"] += agree
            if not is_copy:
                gaps[name]["n_no_copy"] += 1
                gaps[name]["agree_no_copy"] += agree
    q2["agreement_by_created_gap"] = {
        k: dict(n_all=v["n_all"], agreement_all=v["agree_all"] / v["n_all"] if v["n_all"] else None,
                n_no_copy=v["n_no_copy"], agreement_no_copy=v["agree_no_copy"] / v["n_no_copy"] if v["n_no_copy"] else None)
        for k, v in gaps.items()}
    q2["sensitivity"] = {
        "exclude_organiser": build_q2([r for r in logbooks if A.namespace(r["space_id"]) != C.ORGANISER], anchored, created, edges),
        "exclude_author": build_q2([r for r in logbooks if A.namespace(r["space_id"]) != C.AUTHOR], anchored, created, edges),
    }
    q2["material_copying_label"] = bool(tau is not None and q2["relative_drop_mean_teams"] >= 0.10
                                        and abs(q2.get("kappa_change", 0)) > 0.05)
    results["Q2_effective_teams"] = q2

    # Q4
    allr = C.claim_raters(logbooks, anchored)
    keys, items, collapsed_all = [], [], {}
    for k, raters in allr.items():
        col = collapse(raters, edges, created) if len(raters) >= 2 else list(raters.values())
        comps = components(sorted(raters), edges) if len(raters) >= 2 else [[s] for s in raters]
        reps = [min(c, key=lambda s: created_key(created, s)) for c in comps]
        votes = [(A.namespace(s), 1 if raters[s] == "verified" else 0) for s in reps if raters[s] in ("verified", "falsified")]
        collapsed_all[k] = col
        if votes:
            keys.append(k)
            items.append(votes)
    ds = dawid_skene(items)
    post = dict(zip(keys, ds["posterior"].tolist()))
    with (C.HERE / "reference_falsifications.csv").open(newline="", encoding="utf-8") as fh:
        ref = list(csv.DictReader(fh))
    q4 = dict(dawid_skene=dict(converged=ds["converged"], iterations=ds["iterations"], n_items=ds["n_items"],
                               n_annotators=ds["n_annotators"], prior_verified=ds["prior_verified"]),
              reference_sha256=A.sha256(C.HERE / "reference_falsifications.csv"), papers={})
    for row in ref:
        claims = []
        for idx, c in enumerate(raw_anch.get(row["orid"], [])):
            k = (row["orid"], A.norm(c["text"]))
            raters = allr.get(k, {})
            nominal = list(raters.values())
            col = collapsed_all.get(k, [])
            claims.append(dict(
                anchored_index=idx, mapped=idx == int(row["mapped_claim_index"]),
                nominal_votes=dict(Counter(nominal)), collapsed_votes=dict(Counter(col)),
                majority_nominal=majority(nominal), majority_collapsed=majority(col),
                ds_posterior_verified=post.get(k) if ds["converged"] else None))
        mapped = next(c for c in claims if c["mapped"])
        q4["papers"][row["short_name"]] = dict(
            orid=row["orid"], mapping_confidence=row["mapping_confidence"], claims=claims,
            mapped_claim_majority_agrees_with_organisers=mapped["majority_collapsed"] == "falsified",
            mapped_claim_ds_agrees_with_organisers=(None if mapped["ds_posterior_verified"] is None
                                                     else mapped["ds_posterior_verified"] < 0.5))
    results["Q4_aggregation"] = q4

    # Q3 regex counts (validation gate applied in q3_score.py)
    hits = defaultdict(set)
    for s, pages in pages_all.items():
        title = ""
        try:
            title = json.loads(pages.get("logbook.json", "{}")).get("title", "")
        except json.JSONDecodeError:
            pass
        text = "\n".join([title] + [pages[p] for p in sorted(pages) if p.startswith("pages/")])
        for key, rx in COMPILED.items():
            if rx.search(text):
                hits[key].add(s)
    n = len(pages_all)
    any_tool = set().union(*[v for k, v in hits.items() if k[0] == "tool"] or [set()])
    any_model = set().union(*[v for k, v in hits.items() if k[0] == "model"] or [set()])
    results["Q3_regex_raw"] = dict(
        n_logbooks=n, any_tool=len(any_tool) / n, any_model=len(any_model) / n,
        no_match=1 - len(any_tool | any_model) / n,
        by_pattern={f"{c}:{k}": len(v) / n for (c, k), v in sorted(hits.items())},
        note="raw match shares; producer prevalence only after the validation gate (results/q3.json)")

    (OUT / "results.json").write_text(json.dumps(results, indent=2, default=int), encoding="utf-8")
    out_manifest = dict(protocol_sha256=A.sha256(C.HERE / "PROTOCOL.md"),
                        code_sha256={f: A.sha256(C.HERE / f) for f in ("analyze_voters.py", "common.py", "patterns.py")},
                        analyze_py_sha256=A.sha256(C.HERE.parent / "analyze.py"),
                        pages_manifest_sha256=A.sha256(C.DATA / "pages_manifest.json"),
                        git_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=C.ROOT, text=True).strip(),
                        inputs=hashes, seed=A.SEED, resamples=A.B, numpy=np.__version__, python=sys.version.split()[0],
                        seconds=round(time.monotonic() - started, 1), results_sha256=A.sha256(OUT / "results.json"))
    (OUT / "manifest.json").write_text(json.dumps(out_manifest, indent=2), encoding="utf-8")
    print(json.dumps(dict(tau=tau, material=q2["material_copying_label"], ds_converged=ds["converged"])),
          f"{out_manifest['seconds']}s")


if __name__ == "__main__":
    main()
