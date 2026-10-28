"""Verdict reliability audit for the ICML 2026 reproduction challenge.

Implements verdict_audit/PROTOCOL.md exactly. Reads local snapshots, checks their
hashes, and writes aggregate results only (no participant-level rows).
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import time
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent
SEED = 20261017
B = 2000
CATS = ["verified", "falsified", "toy", "inconclusive", "other"]
ORGANISER = "ICML-2026-agent-repro"
INPUTS = {
    "verdicts_latest_tmp.json": "b4bae4eb8a333dc98c3edd6865452ff96009b5b1a6dd1685d96a61bdaf31c239",
    "verdicts_live.json": "76eeb935cc0972ec5b0ef9464a43d0920e00af1328db5460a71b61971351d544",
    "claims_anchored.json": "eb3f2d878646ca5c40da121741a613c1f9e1e10c14845f373db107e74a4ef439",
    "claims.json": "af5ab2d62f786ae36861957cbd08b4188f6d4c86e67152becc661a9c5bbb9d57",
    "audit/public_repro_audit.csv": "459ae5d58aaeae2c777870e664e49920eb2d9aa75052201a9d9ea85e51addcd8",
    "medical_neuro_papers.csv": "92dc7e4cb51bef127428975c7372fd05f288df0566eff6454275887e68eeabe9",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def norm(text: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", str(text)).casefold().split())


def exact(text: str) -> str:
    return str(text).strip()


def category(verdict) -> str:
    value = str(verdict or "").strip().lower()
    return value if value in CATS[:4] else "other"


def load_json(path: Path):
    data = json.loads(path.read_text(encoding="utf-8-sig"))
    return data if isinstance(data, dict) else {r["space_id"]: r for r in data}


def load_orids(path: Path) -> set[str]:
    import csv
    with path.open(newline="", encoding="utf-8") as handle:
        return {row["orid"] for row in csv.DictReader(handle) if row.get("orid")}


def namespace(space_id: str) -> str:
    return space_id.split("/", 1)[0]


def dedupe(records: list[dict]) -> tuple[list[dict], int]:
    """Keep the latest judged logbook per (reproducer, paper); ties by space_id."""
    best: dict[tuple[str, str], dict] = {}
    for r in records:
        key = (namespace(r["space_id"]), r["orid"])
        rank = (r.get("judged_at") or "", r["space_id"])
        if key not in best or rank > (best[key].get("judged_at") or "", best[key]["space_id"]):
            best[key] = r
    return list(best.values()), len(records) - len(best)


# ---------- agreement ----------

def claim_terms(ratings: list[str], cats: list[str]) -> tuple[float, np.ndarray]:
    counts = np.array([ratings.count(c) for c in cats], float)
    n = counts.sum()
    return float((counts * (counts - 1)).sum() / (n * (n - 1))), counts


def paper_table(groups: dict[str, list[list[str]]], cats: list[str]) -> np.ndarray:
    """Rows per paper: [sum of P_i, number of claims, rating counts per category]."""
    rows = []
    for claims in groups.values():
        row = np.zeros(2 + len(cats))
        for ratings in claims:
            p, counts = claim_terms(ratings, cats)
            row[0] += p
            row[1] += 1
            row[2:] += counts
        rows.append(row)
    return np.array(rows).reshape(-1, 2 + len(cats))


def kappa_from_table(table: np.ndarray) -> dict:
    total = table.sum(0)
    po = total[0] / total[1]
    p = total[2:] / total[2:].sum()
    pe = float((p ** 2).sum())
    return dict(po=float(po), pe=pe, kappa=float((po - pe) / (1 - pe)) if pe < 1 else float("nan"))


def cluster_ci(table: np.ndarray, rng: np.random.Generator, stat=lambda t: kappa_from_table(t)["kappa"]):
    n = len(table)
    draws = np.array([stat(table[rng.integers(0, n, n)]) for _ in range(B)])
    return [float(np.nanpercentile(draws, 2.5)), float(np.nanpercentile(draws, 97.5))]


def permutation_po(claims: list[list[str]], rng: np.random.Generator) -> np.ndarray:
    """Po under shuffling labels across ratings, keeping each claim's rating count."""
    flat = np.array([CATS.index(r) for c in claims for r in c])
    sizes = np.array([len(c) for c in claims])
    owner = np.repeat(np.arange(len(claims)), sizes)
    null = np.empty(B)
    for b in range(B):
        counts = np.zeros((len(claims), len(CATS)))
        np.add.at(counts, (owner, rng.permutation(flat)), 1)
        null[b] = np.mean((counts * (counts - 1)).sum(1) / (sizes * (sizes - 1)))
    return null


def agreement(groups: dict[str, list[list[str]]], rng: np.random.Generator, permute: bool) -> dict:
    claims = [c for cl in groups.values() for c in cl]
    if not claims:
        return dict(n_claims=0)
    table = paper_table(groups, CATS)
    out = dict(n_papers=len(groups), n_claims=len(claims), n_ratings=int(sum(len(c) for c in claims)),
               **kappa_from_table(table))
    out["kappa_ci"] = cluster_ci(table, rng)
    out["vf_conflict_share"] = float(np.mean([("verified" in c) and ("falsified" in c) for c in claims]))
    binary = {k: [["decisive" if r in ("verified", "falsified") else "not" for r in c] for c in v]
              for k, v in groups.items()}
    btable = paper_table(binary, ["decisive", "not"])
    out["binary_decisive"] = dict(**kappa_from_table(btable), kappa_ci=cluster_ci(btable, rng))
    if permute:
        null = permutation_po(claims, rng)
        out["null_po_mean"] = float(null.mean())
        out["null_po_ci"] = [float(np.percentile(null, 2.5)), float(np.percentile(null, 97.5))]
        out["perm_p_po"] = float((1 + (null >= out["po"]).sum()) / (B + 1))
    return out


def multi_rated(logbooks: list[dict], anchored: dict[str, set[str]], key) -> dict[str, list[list[str]]]:
    per_claim: dict[tuple[str, str], dict[str, str]] = defaultdict(dict)
    for r in logbooks:
        ns = namespace(r["space_id"])
        for c in r.get("claims") or []:
            t = key(c.get("claim", ""))
            if t in anchored.get(r["orid"], set()) and ns not in per_claim[(r["orid"], t)]:
                per_claim[(r["orid"], t)][ns] = category(c.get("verdict"))
    groups: dict[str, list[list[str]]] = defaultdict(list)
    for (orid, _), raters in per_claim.items():
        if len(raters) >= 2:
            groups[orid].append(list(raters.values()))
    return dict(groups)


# ---------- medical gap ----------

def medical_gap(logbooks: list[dict], medical: set[str], rng: np.random.Generator) -> dict:
    per_paper: dict[str, np.ndarray] = defaultdict(lambda: np.zeros(len(CATS)))
    for r in logbooks:
        for c in r.get("claims") or []:
            per_paper[r["orid"]][CATS.index(category(c.get("verdict")))] += 1
    orids = sorted(per_paper)
    counts = np.array([per_paper[o] for o in orids])
    is_med = np.array([o in medical for o in orids])

    def shares(mask, rows=counts):
        tot = rows[mask].sum(0)
        return tot / tot.sum()

    def diff(mask, rows=counts):
        return shares(mask, rows) - shares(~mask, rows)

    observed = diff(is_med)
    med_idx, oth_idx = np.where(is_med)[0], np.where(~is_med)[0]
    boots = []
    for _ in range(B):
        idx = np.concatenate([rng.choice(med_idx, len(med_idx)), rng.choice(oth_idx, len(oth_idx))])
        boots.append(diff(is_med[idx], counts[idx]))
    boots = np.array(boots)
    perm = np.array([diff(rng.permutation(is_med))[0] for _ in range(B)])
    v = CATS.index("verified")
    return dict(
        n_papers_medical=int(is_med.sum()), n_papers_other=int((~is_med).sum()),
        n_claims_medical=int(counts[is_med].sum()), n_claims_other=int(counts[~is_med].sum()),
        share_medical=dict(zip(CATS, shares(is_med).round(4).tolist())),
        share_other=dict(zip(CATS, shares(~is_med).round(4).tolist())),
        diff=dict(zip(CATS, observed.round(4).tolist())),
        diff_ci={c: [float(np.percentile(boots[:, i], 2.5)), float(np.percentile(boots[:, i], 97.5))]
                 for i, c in enumerate(CATS)},
        perm_p_verified=float((1 + (np.abs(perm) >= abs(observed[v])).sum()) / (B + 1)),
    )


# ---------- anchoring ----------

def anchoring(logbooks: list[dict], anchored: dict[str, set[str]], loose: dict[str, set[str]]) -> dict:
    tally = Counter()
    for r in logbooks:
        for c in r.get("claims") or []:
            t = norm(c.get("claim", ""))
            if t in anchored.get(r["orid"], set()):
                tally["anchored"] += 1
            elif t in loose.get(r["orid"], set()):
                tally["loose_only"] += 1
            else:
                tally["neither"] += 1
    n = sum(tally.values())
    return dict(n_claims=n, **{k: tally[k] for k in ("anchored", "loose_only", "neither")},
                **{f"share_{k}": round(tally[k] / n, 4) for k in ("anchored", "loose_only", "neither")})


# ---------- judge test-retest ----------

def retest(early: dict, late: dict, rng: np.random.Generator) -> dict:
    groups: dict[str, list[list[str]]] = {}
    unpaired = 0
    for sid, b in late.items():
        a = early.get(sid)
        if not a or a.get("sha") != b.get("sha") or a.get("judged_at") == b.get("judged_at"):
            continue
        pairs = []
        for ca, cb in zip(a.get("claims") or [], b.get("claims") or []):
            if norm(ca.get("claim", "")) == norm(cb.get("claim", "")):
                pairs.append([category(ca.get("verdict")), category(cb.get("verdict"))])
            else:
                unpaired += 1
        if pairs:
            groups[sid] = pairs
    out = agreement(groups, rng, permute=False)
    claims = [c for v in groups.values() for c in v]
    out["n_logbooks"] = out.pop("n_papers", 0)
    out["unpaired_claims"] = unpaired
    out["vf_flip_share"] = out.pop("vf_conflict_share", None)
    out["any_change_share"] = float(np.mean([c[0] != c[1] for c in claims])) if claims else None
    return out


def main() -> None:
    started = time.monotonic()
    out_dir = HERE / "results"
    out_dir.mkdir(exist_ok=False)
    hashes = {}
    for name, expected in INPUTS.items():
        got = sha256(ROOT / name)
        if got != expected:
            sys.exit(f"Hash mismatch for {name}: {got}")
        hashes[name] = got

    late = load_json(ROOT / "verdicts_latest_tmp.json")
    early = load_json(ROOT / "verdicts_live.json")
    raw_anch = json.loads((ROOT / "claims_anchored.json").read_text(encoding="utf-8-sig"))
    raw_loose = json.loads((ROOT / "claims.json").read_text(encoding="utf-8-sig"))
    anch_norm = {k: {norm(c["text"]) for c in v} for k, v in raw_anch.items()}
    anch_exact = {k: {exact(c["text"]) for c in v} for k, v in raw_anch.items()}
    loose_norm = {k: {norm(c["text"]) for c in v} for k, v in raw_loose.items()}
    medical = load_orids(ROOT / "audit/public_repro_audit.csv")
    medical_narrow = load_orids(ROOT / "medical_neuro_papers.csv")

    records = list(late.values())
    models = Counter(r.get("model") for r in records)
    logbooks, dropped = dedupe(records)
    verdict_counts = Counter(category(c.get("verdict")) for r in logbooks for c in r.get("claims") or [])

    def rng():
        return np.random.default_rng(SEED)

    results = dict(
        snapshot=dict(logbooks_total=len(records), logbooks_dedup=len(logbooks), dropped_by_dedup=dropped,
                      papers=len({r["orid"] for r in records}),
                      reproducers=len({namespace(r["space_id"]) for r in records}),
                      judged_from=min(r["judged_at"] for r in records), judged_to=max(r["judged_at"] for r in records),
                      judge_models=dict(models), verdict_counts=dict(verdict_counts),
                      logbooks_without_anchored_entry=sum(r["orid"] not in raw_anch for r in logbooks)),
        C1_anchoring=anchoring(logbooks, anch_norm, loose_norm),
        C2_agreement=agreement(multi_rated(logbooks, anch_norm, norm), rng(), permute=True),
        C3_medical_gap=medical_gap(logbooks, medical, rng()),
        C4_judge_retest=retest(early, late, rng()),
    )
    med_groups = multi_rated([r for r in logbooks if r["orid"] in medical], anch_norm, norm)
    oth_groups = multi_rated([r for r in logbooks if r["orid"] not in medical], anch_norm, norm)
    results["C3_secondary_agreement"] = dict(medical=agreement(med_groups, rng(), permute=False),
                                             other=agreement(oth_groups, rng(), permute=False))
    no_org = [r for r in logbooks if namespace(r["space_id"]) != ORGANISER]
    all_books = records
    results["sensitivity"] = dict(
        exclude_organiser=dict(C2=agreement(multi_rated(no_org, anch_norm, norm), rng(), permute=False),
                               C3=medical_gap(no_org, medical, rng())),
        exact_text=dict(C2=agreement(multi_rated(logbooks, anch_exact, exact), rng(), permute=False)),
        narrow_medical_label=dict(C3=medical_gap(logbooks, medical_narrow, rng())),
        no_dedup=dict(C3=medical_gap(all_books, medical, rng())),
    )

    c2, c4, c3 = results["C2_agreement"], results["C4_judge_retest"], results["C3_medical_gap"]
    results["prespecified_labels"] = dict(
        substantial_reproducer_disagreement=bool(c2["kappa_ci"][1] < 0.60 and c2["kappa_ci"][1] < c4["kappa_ci"][0]),
        medical_gap_in_verified_share=bool((c3["diff_ci"]["verified"][0] > 0 or c3["diff_ci"]["verified"][1] < 0)
                                           and c3["perm_p_verified"] < 0.05),
    )
    (out_dir / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    manifest = dict(protocol_sha256=sha256(HERE / "PROTOCOL.md"), code_sha256=sha256(Path(__file__)),
                    git_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                    inputs=hashes, seed=SEED, resamples=B, numpy=np.__version__, python=sys.version.split()[0],
                    seconds=round(time.monotonic() - started, 1),
                    results_sha256=sha256(out_dir / "results.json"))
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(results["prespecified_labels"]), f"{manifest['seconds']}s")


if __name__ == "__main__":
    main()
