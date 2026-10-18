"""Score Protocol 7 labelling: kappa gate, prefilter miss rate, items needing adjudication, final labels.

Reads data/keys.json, codes/coder{1,2}_batch_*.csv and (if present) codes/adjudication.csv.
Writes results/labelling.json, codes/to_adjudicate.csv and, once adjudication is complete,
codes/labels.csv (public ids only).
"""
from __future__ import annotations

import csv
import json
import math
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
CODES = HERE / "codes"
LABELS = ("clinical", "life_science", "no")
GATE = 0.80


def read_codes(role: str) -> dict[str, str]:
    out = {}
    for f in sorted(CODES.glob(f"{role}_batch_*.csv")):
        with f.open(newline="", encoding="utf-8") as fh:
            for r in csv.DictReader(fh):
                lab = r["label"].strip().lower()
                if lab not in LABELS:
                    raise SystemExit(f"{f.name}: unknown label {lab!r} for {r['item_id']}")
                if r["item_id"] in out:
                    raise SystemExit(f"{f.name}: duplicate item {r['item_id']}")
                out[r["item_id"]] = lab
    return out


def kappa(a: dict, b: dict, ids: list[str]) -> dict:
    po = sum(a[i] == b[i] for i in ids) / len(ids)
    ca, cb = Counter(a[i] for i in ids), Counter(b[i] for i in ids)
    pe = sum(ca[l] * cb[l] for l in LABELS) / len(ids) ** 2
    return dict(n=len(ids), po=po, kappa=(po - pe) / (1 - pe) if pe < 1 else None)


def wilson(k: int, n: int, z: float = 1.96) -> list[float]:
    p = k / n
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return [max(0.0, c - h), min(1.0, c + h)]


def main() -> None:
    keys = json.loads((HERE / "data" / "keys.json").read_text(encoding="utf-8"))
    c1, c2 = read_codes("coder1"), read_codes("coder2")
    with (HERE / "data" / "sheets" / "coder1_all.csv").open(newline="", encoding="utf-8") as fh:
        need1 = {r["item_id"] for r in csv.DictReader(fh)}
    with (HERE / "data" / "sheets" / "coder2_all.csv").open(newline="", encoding="utf-8") as fh:
        need2 = {r["item_id"] for r in csv.DictReader(fh)}
    missing = dict(coder1=sorted(need1 - set(c1))[:10], coder2=sorted(need2 - set(c2))[:10],
                   n_missing1=len(need1 - set(c1)), n_missing2=len(need2 - set(c2)))
    if missing["n_missing1"] or missing["n_missing2"]:
        raise SystemExit(f"Incomplete coding: {missing}")
    kind = {i: keys[i].split(":", 1)[0] for i in keys}
    overlap = sorted(i for i in need2 if kind[i] == "cand")
    non = sorted(i for i in need2 if kind[i] == "non")
    rec = sorted(i for i in need2 if kind[i] == "reclaim")
    gate = kappa(c1, c2, overlap)
    gate_met = gate["kappa"] is not None and gate["kappa"] >= GATE
    disagree = [i for i in overlap + non + rec if c1[i] != c2[i]]
    adj = {}
    if (CODES / "adjudication.csv").exists():
        with (CODES / "adjudication.csv").open(newline="", encoding="utf-8") as fh:
            for r in csv.DictReader(fh):
                if r["label"] not in LABELS or not r.get("reason", "").strip():
                    raise SystemExit(f"adjudication row needs a valid label and a reason: {r}")
                adj[r["item_id"]] = r["label"]
    with (CODES / "to_adjudicate.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["item_id", "coder1", "coder2"])
        w.writerows([i, c1[i], c2[i]] for i in disagree if i not in adj)

    def final(i: str):
        if i in need2:
            return c1[i] if c1[i] == c2[i] else adj.get(i)
        return c1[i]

    pending = [i for i in disagree if i not in adj]
    non_final = [final(i) for i in non]
    out = dict(gate=dict(**gate, threshold=GATE, met=gate_met),
               agreement_non_candidates=kappa(c1, c2, non), agreement_reclaim=kappa(c1, c2, rec),
               n_items=dict(coder1=len(c1), coder2=len(c2), overlap=len(overlap), non_candidates=len(non), reclaim=len(rec)),
               disagreements=len(disagree), pending_adjudication=len(pending))
    if not pending:
        k = sum(l in ("clinical", "life_science") for l in non_final)
        out["prefilter_miss_rate"] = dict(medical_among_non_candidates=k, n=len(non), share=k / len(non),
                                          wilson=wilson(k, len(non)))
        rows = [dict(unit=keys[i].split(":", 1)[1], kind=kind[i], label=final(i)) for i in sorted(keys) if kind[i] in ("cand", "reclaim")]
        with (CODES / "labels.csv").open("w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, ["unit", "kind", "label"])
            w.writeheader()
            w.writerows(rows)
        out["label_counts"] = {k2: dict(Counter(r["label"] for r in rows if r["kind"] == k2)) for k2 in ("cand", "reclaim")}
    (HERE / "results" / "labelling.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
