"""Score Q3 blind coding against the frozen gate (PROTOCOL.md, Q3). Writes results/q3.json."""
from __future__ import annotations

import csv
import json
from collections import Counter

import common as C

CODES = C.HERE / "q3_codes"
LABELS_A = ["producer", "subject", "unclear"]
LABELS_B = ["named", "not_named", "unclear"]
GATE = 0.80


def read(path):
    with path.open(newline="", encoding="utf-8") as fh:
        return {r["item_id"]: r["label"].strip().lower() for r in csv.DictReader(fh)}


def cohen(a: dict, b: dict, labels: list[str]) -> dict:
    ids = sorted(set(a) & set(b))
    po = sum(a[i] == b[i] for i in ids) / len(ids)
    pa, pb = Counter(a[i] for i in ids), Counter(b[i] for i in ids)
    pe = sum(pa[l] * pb[l] for l in labels) / len(ids) ** 2
    return dict(n=len(ids), po=po, kappa=(po - pe) / (1 - pe) if pe < 1 else None)


def main() -> None:
    q3 = C.DATA / "q3"
    with (q3 / "key_A.csv").open(newline="", encoding="utf-8") as fh:
        key_a = {r["item_id"]: r for r in csv.DictReader(fh)}
    a1, a2 = read(CODES / "coder1_A.csv"), read(CODES / "coder2_A.csv")
    b1, b2 = read(CODES / "coder1_B.csv"), read(CODES / "coder2_B.csv")
    for codes, labels in ((a1, LABELS_A), (a2, LABELS_A), (b1, LABELS_B), (b2, LABELS_B)):
        bad = {v for v in codes.values() if v not in labels}
        if bad:
            raise SystemExit(f"Unknown labels {bad}")
    if set(a1) != set(key_a) or set(a2) != set(key_a):
        raise SystemExit("Sample A codes do not cover the sheet")
    out = dict(agreement_A=cohen(a1, a2, LABELS_A), agreement_B=cohen(b1, b2, LABELS_B), categories={})
    results = json.loads((C.HERE / "results" / "results.json").read_text(encoding="utf-8"))["Q3_regex_raw"]
    for cat in ("tool", "model"):
        ids = [i for i, k in key_a.items() if k["category"] == cat]
        producer = sum(a1[i] == "producer" and a2[i] == "producer" for i in ids)
        precision = producer / len(ids)
        raw = results[f"any_{cat}"]
        out["categories"][cat] = dict(
            n_sampled=len(ids), both_producer=producer, precision=precision, gate_passed=precision >= GATE,
            raw_logbook_match_share=raw,
            approx_producer_share=raw * precision if precision >= GATE else None,
            label="validated" if precision >= GATE else "not validated: raw match count only")
        if precision >= GATE:
            out["categories"][cat]["patterns_among_producer"] = dict(Counter(
                key_a[i]["pattern"] for i in ids if a1[i] == "producer" and a2[i] == "producer"))
    named = sum(b1[i] == "named" and b2[i] == "named" for i in b1)
    out["miss_rate_B"] = dict(n=len(b1), both_named=named, share=named / len(b1), no_match_share=results["no_match"])
    (C.HERE / "results" / "q3.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
