"""Draw the protocol 4 coding samples (E0 papers, E1 claims, E1 evidence texts).

Writes blind coding sheets to sheets/ and the linking keys to keys/. Coders get
only sheets/. Refuses to overwrite existing files.
"""
import csv
import hashlib
import json
import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE.parent))
import analyze as A  # noqa: E402  (protocol 1 helpers: norm, dedupe, namespace, category)

SNAPSHOT = ROOT / "data/verdicts_confirmation_2026-10-17/verdicts.json"
SNAPSHOT_SHA = "652f1c162462d7ad142ff32e2bd13d5853e21c639681ef416971776fdd33c526"
SEED = 20261018


def write(path: Path, header, rows):
    path.parent.mkdir(exist_ok=True)
    if path.exists():
        sys.exit(f"refusing to overwrite {path}")
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


def main():
    assert hashlib.sha256(SNAPSHOT.read_bytes()).hexdigest() == SNAPSHOT_SHA
    meta = json.loads((ROOT / "papers.json").read_text(encoding="utf-8-sig"))
    papers = {p["orid"]: p for p in meta["papers"]}
    abstracts = meta["abstracts"]
    medical = A.load_orids(ROOT / "audit/public_repro_audit.csv")

    # E0
    rng = random.Random(SEED)
    med = sorted(o for o in papers if o in medical)
    oth = sorted(o for o in papers if o not in medical)
    chosen = [(o, "medical") for o in rng.sample(med, 100)] + [(o, "other") for o in rng.sample(oth, 100)]
    rng.shuffle(chosen)
    write(HERE / "sheets/e0_papers.csv", ["item", "orid", "title", "abstract"],
          [[i + 1, o, papers[o]["title"], " ".join(str(abstracts.get(o, "")).split())] for i, (o, _) in enumerate(chosen)])
    write(HERE / "keys/e0_key.csv", ["item", "orid", "audit_label"], [[i + 1, o, lab] for i, (o, lab) in enumerate(chosen)])

    # E1 claims and evidence
    logbooks, _ = A.dedupe(list(A.load_json(SNAPSHOT).values()))
    texts = sorted({A.norm(c.get("claim", "")) for r in logbooks for c in r.get("claims") or [] if c.get("claim")})
    rng = random.Random(SEED)
    claim_sample = rng.sample(texts, 200)
    write(HERE / "sheets/e1_claims.csv", ["item", "claim"], [[i + 1, t] for i, t in enumerate(claim_sample)])

    pool = sorted(((r["space_id"], j, " ".join(str(c.get("evidence") or "").split()))
                   for r in logbooks for j, c in enumerate(r.get("claims") or [])
                   if A.category(c.get("verdict")) in ("inconclusive", "toy")), key=lambda x: (x[0], x[1]))
    rng = random.Random(SEED)
    ev = rng.sample(pool, 200)
    write(HERE / "sheets/e1_evidence.csv", ["item", "evidence"], [[i + 1, e[2]] for i, e in enumerate(ev)])
    write(HERE / "keys/e1_evidence_key.csv", ["item", "space_id", "claim_index"], [[i + 1, e[0], e[1]] for i, e in enumerate(ev)])
    print("E0 papers 200; claims pool", len(texts), "sampled 200; evidence pool", len(pool), "sampled 200")


if __name__ == "__main__":
    main()
