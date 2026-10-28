"""Protocol 7 analyses R-A1, R-A2, R-A3, R-B1 (verdict_audit/readiness/PROTOCOL.md, Amendments 1-2).

Reads data/papers.json, codes/labels.csv, the challenge verdicts and RECLAIM files; writes aggregates only
to results/readiness.json (+ readiness_manifest.json).
"""
from __future__ import annotations

import csv
import hashlib
import json
import re
import statistics
import subprocess
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from urllib.parse import urlparse

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
DATA = HERE / "data"
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parent / "falsifiability"))
import analyze as A  # noqa: E402  (frozen helpers: dedupe, category, namespace)

SEED, B = 20261017, 2000
AUTHOR = "kondratevakate"
VERDICTS = ROOT / "data/verdicts_confirmation_2026-10-17/verdicts.json"
VERDICTS_SHA = "652f1c162462d7ad142ff32e2bd13d5853e21c639681ef416971776fdd33c526"
CODE_HOSTS = {"github.com", "gitlab.com", "bitbucket.org", "huggingface.co", "codeberg.org", "sourceforge.net",
              "gitee.com", "anonymous.4open.science"}
NON_ARTIFACT = {"openreview.net", "youtube.com", "youtu.be", "slideslive.com", "x.com", "twitter.com"}
VENUE_DOMAINS = {"iclr": "iclr.cc", "neurips": "neurips.cc", "icml": "icml.cc", "iccv": "thecvf.com",
                 "cvpr": "thecvf.com", "eccv": "ecva.net"}
RESTRICTED = re.compile(
    r"\b(MIMIC\w*|eICU|UK Biobank|ADNI|PhysioNet|electronic health records?|EHRs?|in-house|proprietary|"
    r"private (data|dataset|cohort)s?|institutional (data|dataset|cohort)s?|hospital (data|records|cohort)s?)\b",
    re.IGNORECASE)
ML = ("iclr", "neurips", "icml")


def host(link: str) -> str | None:
    if not link or link == "None" or not re.match(r"https?://", link):
        return None
    h = urlparse(link).netloc.lower().split(":")[0]
    return h[4:] if h.startswith("www.") else h


def outcomes(p: dict) -> dict:
    hosts = [h for h in (host(l) for l in [p["url"], *p["media_urls"]]) if h]
    venue = VENUE_DOMAINS[p["stratum"].split("-")[0]]
    code = any(h in CODE_HOSTS for h in hosts)
    anon = any(h == "anonymous.4open.science" for h in hosts)
    artifact = any(h not in NON_ARTIFACT and not h.endswith(venue) for h in hosts)
    return dict(o1=int(code), o1_anonymous=int(anon), o2=int(artifact),
                o3=int(bool(RESTRICTED.search(p["title"] + " " + p["abstract"]))))


def mh_rd(t: np.ndarray) -> float | None:
    """t[s] = [n_exposed, events_exposed, n_unexposed, events_unexposed]; exposed = medical."""
    ne, ae, nu, au = t[:, 0], t[:, 1], t[:, 2], t[:, 3]
    tot = ne + nu
    ok = (ne > 0) & (nu > 0)
    if not ok.any():
        return None
    w = np.where(ok, ne * nu / np.where(ok, tot, 1), 0)
    num = np.where(ok, (ae * nu - au * ne) / np.where(ok, tot, 1), 0)
    return float(num.sum() / w.sum())


def stratified(units: list[dict], group_of, outcome: str, strata: list[str], rng) -> dict:
    """MH risk difference (group 1 minus group 0) with stratified bootstrap within group x stratum."""
    cells = defaultdict(list)
    for u in units:
        g = group_of(u)
        if g is not None and u["stratum"] in strata:
            cells[(u["stratum"], g)].append(u[outcome])
    arr = {k: np.array(v, float) for k, v in cells.items()}

    def table(sample):
        return np.array([[len(sample.get((s, 1), [])), sample.get((s, 1), np.array([])).sum(),
                          len(sample.get((s, 0), [])), sample.get((s, 0), np.array([])).sum()] for s in strata], float)

    obs = table(arr)
    est = mh_rd(obs)
    per = {}
    boots, per_boot = [], defaultdict(list)
    for _ in range(B):
        samp = {k: v[rng.integers(0, len(v), len(v))] for k, v in arr.items() if len(v)}
        t = table(samp)
        boots.append(mh_rd(t))
        for i, s in enumerate(strata):
            if t[i, 0] and t[i, 2]:
                per_boot[s].append(t[i, 1] / t[i, 0] - t[i, 3] / t[i, 2])
    for i, s in enumerate(strata):
        ne, ae, nu, au = obs[i]
        per[s] = dict(n_medical=int(ne), n_other=int(nu),
                      share_medical=float(ae / ne) if ne else None, share_other=float(au / nu) if nu else None,
                      rd=float(ae / ne - au / nu) if ne and nu else None,
                      ci=[float(np.percentile(per_boot[s], 2.5)), float(np.percentile(per_boot[s], 97.5))] if per_boot[s] else None)
    boots = [b for b in boots if b is not None]
    return dict(rd=est, ci=[float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))] if boots else None,
                n_medical=int(obs[:, 0].sum()), n_other=int(obs[:, 2].sum()),
                share_medical=float(obs[:, 1].sum() / obs[:, 0].sum()) if obs[:, 0].sum() else None,
                share_other=float(obs[:, 3].sum() / obs[:, 2].sum()) if obs[:, 2].sum() else None,
                negative_strata=sum(1 for v in per.values() if v["rd"] is not None and v["rd"] < 0),
                strata_with_both=sum(1 for v in per.values() if v["rd"] is not None), per_stratum=per)


def load_units() -> list[dict]:
    units = json.loads((DATA / "papers.json").read_text(encoding="utf-8"))
    labels = {}
    with (HERE / "codes" / "labels.csv").open(newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            if r["kind"] == "cand":
                labels[r["unit"]] = r["label"]
    for u in units:
        u.update(outcomes(u))
        u["label"] = labels.get(f"{u['stratum']}:{u['forum_id']}") if u["candidate"] else None
        if u["candidate"] and u["label"] is None:
            raise SystemExit(f"candidate without label: {u['stratum']}:{u['forum_id']}")
    return units


def group(kind: str):
    def g(u):
        if not u["candidate"]:
            return 0
        if kind == "medical":
            return 1 if u["label"] in ("clinical", "life_science") else None
        if kind in ("clinical", "life_science"):
            return 1 if u["label"] == kind else None
        if kind == "topic":
            return 1 if u["by_topic"] else None
    return g


def r_a3(units: list[dict], rng) -> dict:
    if A.sha256(VERDICTS) != VERDICTS_SHA:
        raise SystemExit("verdict snapshot hash mismatch")
    logs, _ = A.dedupe(list(A.load_json(VERDICTS).values()))
    logs = [r for r in logs if A.namespace(r["space_id"]) != AUTHOR]
    attempted, decided = set(), set()
    for r in logs:
        attempted.add(r["orid"])
        if any(A.category(c.get("verdict")) in ("verified", "falsified") for c in r.get("claims") or []):
            decided.add(r["orid"])
    icml = {u["forum_id"]: u for u in units if u["stratum"] == "icml-2026"}

    def analyse(med_of) -> dict:
        rows = []
        for orid, u in icml.items():
            g = med_of(orid, u)
            if g is None or orid not in attempted:
                continue
            rows.append(dict(stratum=f"o1={u['o1']}", g=g, decided=int(orid in decided), o1=u["o1"]))
        cells = {}
        for g in (1, 0):
            for o in (1, 0):
                v = [r["decided"] for r in rows if r["g"] == g and r["o1"] == o]
                b = [np.mean(np.array(v)[rng.integers(0, len(v), len(v))]) for _ in range(B)] if v else []
                cells[f"{'medical' if g else 'other'}_code{o}"] = dict(
                    n=len(v), decided=float(np.mean(v)) if v else None,
                    ci=[float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))] if b else None)
        units_like = [dict(stratum=r["stratum"], g=r["g"], decided=r["decided"]) for r in rows]
        adj = stratified(units_like, lambda u: u["g"], "decided", ["o1=1", "o1=0"], rng)
        unadj = stratified([dict(u, stratum="all") for u in units_like], lambda u: u["g"], "decided", ["all"], rng)
        return dict(n_attempted=len(rows), cells=cells, unadjusted=dict(rd=unadj["rd"], ci=unadj["ci"]),
                    adjusted_for_o1=dict(rd=adj["rd"], ci=adj["ci"]))

    p7 = analyse(lambda orid, u: (1 if u["label"] in ("clinical", "life_science") else None) if u["candidate"] else 0)
    import analyze6 as P6
    a, b, cons, dis, missing = P6.consensus()
    sheet = {r["item"]: r["orid"] for r in P6.read(HERE.parent / "falsifiability" / "sheets" / "candidates.csv")}
    med6 = {sheet[i] for i in cons if cons[i] in ("clinical", "life_science")}
    cand6 = set(sheet.values())
    broad = A.load_orids(ROOT / "audit/public_repro_audit.csv")
    p6 = analyse(lambda orid, u: 1 if orid in med6 else (0 if orid not in cand6 and orid not in broad else None))
    return dict(protocol7_label=p7, sensitivity_protocol6_label=p6,
                challenge_attempted_papers=len(attempted), icml2026_units=len(icml),
                icml2026_units_attempted=sum(1 for o in icml if o in attempted))


def r_b1() -> dict:
    labels = {}
    with (HERE / "codes" / "labels.csv").open(newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            if r["kind"] == "reclaim":
                labels[r["unit"]] = r["label"]
    eval_rows = [json.loads(l) for l in (DATA / "reclaim_eval_100.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    index = json.loads((DATA / "reclaim_runs_index.json").read_text(encoding="utf-8"))
    release_tier = {p["arxiv_id"]: p["tier"] for p in index["papers"]}
    runs = defaultdict(list)
    for r in index["runs"]:
        runs[r["arxiv_id"]].append(r["mode"])
    out = {}
    for name, keep in (("medical", ("clinical", "life_science")), ("other", ("no",))):
        ids = [r["custom_id"] for r in eval_rows if labels.get(r["custom_id"]) in keep]
        rows = [r for r in eval_rows if r["custom_id"] in ids]
        hours = [float(r["h100_hours_adjudicated"]) for r in rows if r.get("h100_hours_adjudicated") not in (None, "")]
        out[name] = dict(
            n=len(rows), labels=dict(Counter(labels[i] for i in ids)),
            dataset_tier=dict(Counter(str(r.get("tier")) for r in rows)),
            release_tier=dict(Counter(release_tier.get(i) for i in ids)),
            median_h100_hours_adjudicated=statistics.median(hours) if hours else None,
            runs=sum(len(runs[i]) for i in ids),
            papers_reproduced_clean_by_any_agent=sum(1 for i in ids if "reproduced-clean" in runs[i]),
            run_modes=dict(Counter(m for i in ids for m in runs[i])))
    return out


def main() -> None:
    started = time.monotonic()
    out_file = HERE / "results" / "readiness.json"
    if out_file.exists():
        sys.exit("results/readiness.json exists; refusing to overwrite")
    units = load_units()
    strata = sorted({u["stratum"] for u in units})
    ml = [s for s in strata if s.split("-")[0] in ML]
    cv = [s for s in strata if s not in ml]
    early = [s for s in strata if not s.endswith("2026")]
    rng = np.random.default_rng(SEED)
    primary = stratified(units, group("medical"), "o1", strata, rng)
    primary["label_less_often_released_with_code"] = bool(
        primary["ci"] and primary["ci"][1] < 0 and primary["negative_strata"] >= 8)
    res = dict(
        units=len(units), strata=strata,
        groups=dict(medical=sum(1 for u in units if group("medical")(u) == 1),
                    other=sum(1 for u in units if group("medical")(u) == 0),
                    excluded_candidates_labelled_no=sum(1 for u in units if u["candidate"] and u["label"] == "no")),
        R_A1=primary,
        R_A1_secondary=dict(
            clinical=stratified(units, group("clinical"), "o1", strata, np.random.default_rng(SEED)),
            life_science=stratified(units, group("life_science"), "o1", strata, np.random.default_rng(SEED)),
            o2_any_artifact_link=stratified(units, group("medical"), "o2", strata, np.random.default_rng(SEED)),
            topic_only_label=stratified(units, group("topic"), "o1", strata, np.random.default_rng(SEED)),
            ml_venues=stratified(units, group("medical"), "o1", ml, np.random.default_rng(SEED)),
            cv_venues=stratified(units, group("medical"), "o1", cv, np.random.default_rng(SEED)),
            strata_2024_2025=stratified(units, group("medical"), "o1", early, np.random.default_rng(SEED)),
            excluding_icml2026=stratified(units, group("medical"), "o1", [s for s in strata if s != "icml-2026"],
                                          np.random.default_rng(SEED)),
            anonymous_repo_share=dict(medical=float(np.mean([u["o1_anonymous"] for u in units if group("medical")(u) == 1])),
                                      other=float(np.mean([u["o1_anonymous"] for u in units if group("medical")(u) == 0])))),
        R_A2=stratified(units, group("medical"), "o3", strata, np.random.default_rng(SEED)),
        R_A3=r_a3(units, np.random.default_rng(SEED)),
        R_B1=r_b1(),
    )
    out_file.write_text(json.dumps(res, indent=2), encoding="utf-8")
    sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    manifest = dict(protocol_sha256=sha(HERE / "PROTOCOL.md"), code_sha256=sha(Path(__file__)),
                    labels_sha256=sha(HERE / "codes" / "labels.csv"), papers_sha256=sha(DATA / "papers.json"),
                    reclaim_runs_index_sha256=sha(DATA / "reclaim_runs_index.json"),
                    reclaim_repo_commit="dd9b876ba6b6dfc67361d167af5110c24d6a2ee0",
                    git_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                    seed=SEED, resamples=B, numpy=np.__version__, python=sys.version.split()[0],
                    seconds=round(time.monotonic() - started, 1), results_sha256=sha(out_file))
    (HERE / "results" / "readiness_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(dict(rd=primary["rd"], ci=primary["ci"], negative=primary["negative_strata"],
                          label=primary["label_less_often_released_with_code"]), indent=1), f"{manifest['seconds']}s")


if __name__ == "__main__":
    main()
