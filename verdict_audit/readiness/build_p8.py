"""Protocol 8 samples and blind sheets for H8.3 and H8.4, and the H8.5 paper list (seed 20261018)."""
from __future__ import annotations

import csv
import json
import random
import re
from collections import Counter, defaultdict

import p8_common as P

BATCH = 250


def write(path, rows, fields):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fields)
        w.writeheader()
        w.writerows(rows)


def batches(prefix: str, rows: list[dict], fields: list[str], size: int = BATCH) -> int:
    for b in range(0, len(rows), size):
        write(P.P8 / "sheets" / f"{prefix}_batch_{b // size:02d}.csv", rows[b:b + size], fields)
    return (len(rows) + size - 1) // size


def mask_title(text: str, title: str) -> str:
    return re.sub(re.escape(title), "[PAPER]", text, flags=re.IGNORECASE) if title else text


def main() -> None:
    if (P.P8 / "keys.json").exists():
        raise SystemExit("data/p8/keys.json exists; refusing to redraw samples")
    grp = P.groups()
    cv = P.claim_verdicts()
    attempted = {o: g for o, g in grp.items() if o in cv}
    rng = random.Random(P.SEED_SAMPLE)
    keys, summary = {}, dict(attempted=dict(Counter(attempted.values())))

    # H8.3: non-decisive claim verdicts, 150 per group, at most 3 per paper
    h83 = []
    for g in ("medical", "other"):
        pool = [(o, i) for o, gg in sorted(attempted.items()) if gg == g
                for i, c in enumerate(cv[o]) if c["verdict"] in ("toy", "inconclusive")]
        rng.shuffle(pool)
        per, chosen = Counter(), []
        for o, i in pool:
            if per[o] < 3:
                per[o] += 1
                chosen.append((o, i))
            if len(chosen) == 150:
                break
        summary[f"h83_{g}"] = dict(pool=len(pool), chosen=len(chosen), papers=len(per))
        h83 += [(g, o, i) for o, i in chosen]
    rng.shuffle(h83)
    rows = []
    for n, (g, o, i) in enumerate(h83):
        c = cv[o][i]
        iid = f"R{n:04d}"
        keys[iid] = dict(task="h83", group=g, orid=o, index=i)
        rows.append(dict(item_id=iid, claim=mask_title(c["raw"], c["title"]), evidence=mask_title(c["evidence"], c["title"])))
    summary["h83_batches"] = batches("h83_coder1", rows, ["item_id", "claim", "evidence"], 150)
    batches("h83_coder2", rows, ["item_id", "claim", "evidence"], 150)

    # H8.4: unique claim texts; all medical, equal-size random sample of other
    uniq = defaultdict(dict)
    for o, g in attempted.items():
        for c in cv[o]:
            if c["text"]:
                uniq[g].setdefault((o, c["text"]), c)
    med = sorted(uniq["medical"])
    oth = sorted(uniq["other"])
    oth = rng.sample(oth, min(len(med), len(oth)))
    items = [("medical", k) for k in med] + [("other", k) for k in oth]
    rng.shuffle(items)
    rows = []
    for n, (g, (o, t)) in enumerate(items):
        iid = f"C{n:05d}"
        c = uniq[g][(o, t)]
        keys[iid] = dict(task="h84", group=g, orid=o, text=t)
        rows.append(dict(item_id=iid, claim=mask_title(c["raw"], c["title"])))
    overlap = sorted(rng.sample(range(len(rows)), round(0.25 * len(rows))))
    summary["h84"] = dict(medical_claims=len(med), other_claims=len(oth), overlap=len(overlap))
    summary["h84_batches_coder1"] = batches("h84_coder1", rows, ["item_id", "claim"])
    summary["h84_batches_coder2"] = batches("h84_coder2", [rows[i] for i in overlap], ["item_id", "claim"])

    # H8.5: paper list (all attempted medical, equal random sample of attempted other)
    papers = json.loads((P.ROOT / "papers.json").read_text(encoding="utf-8-sig"))["papers"]
    arxiv = {p["orid"]: p.get("arxiv") for p in papers}
    med_p = sorted(o for o, g in attempted.items() if g == "medical")
    oth_p = rng.sample(sorted(o for o, g in attempted.items() if g == "other"), len(med_p))
    h85 = [dict(orid=o, group=g, arxiv=arxiv.get(o)) for g, lst in (("medical", med_p), ("other", oth_p)) for o in lst]
    (P.P8 / "h85_papers.json").write_text(json.dumps(h85, indent=1), encoding="utf-8")
    summary["h85"] = dict(medical=len(med_p), other=len(oth_p), without_arxiv=sum(1 for x in h85 if not x["arxiv"]))
    (P.P8 / "keys.json").write_text(json.dumps(keys), encoding="utf-8")
    (P.HERE / "results").mkdir(exist_ok=True)
    (P.HERE / "results" / "p8_samples_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
