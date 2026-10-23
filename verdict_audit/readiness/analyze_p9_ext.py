"""Protocol 9, Note 9-17: extension sample — agreement, adjudication, and A9.4 tables (extension alone and pooled).

Usage: python analyze_p9_ext.py score | build_adj | merge | run
"""
from __future__ import annotations

import csv
import json
import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from analyze_p8 import kappa  # noqa: E402
from analyze_p9 import REF_FIELDS, REF_LABELS, binary_agreement, final_codes, reference_codes  # noqa: E402
from build_p9_adjudication import DEFINITION  # noqa: E402
from build_p9_sheets import CODES as P9CODES, FIELDS, LABELS, P9, excerpts  # noqa: E402

CODES = P9CODES / "ext"
KEYS = P9 / "ext_keys.json"
SHEETS = P9 / "ext_sheets"
ADJ_SHEETS, ADJ_BATCHES, ADJ = P9 / "ext_adj_sheets", P9 / "ext_adj_batches.json", CODES / "ext_adjudication.csv"
FT = ("code", "main_data_access", "weights_release")
PAIRS = (("code", "released", "code_used"), ("main_data_access", "open", "data_obtained"),
         ("weights_release", "yes", "weights_used"))
DEFS = {**DEFINITION,
        "code": "code: released (a link to code is given), promised (code will be released), no (no code statement, or "
                "code not released), unclear",
        "main_data_access": "main_data_access: open (main evaluation data downloadable without a data-use agreement), "
                            "credentialed_or_request, private, synthetic_only, unclear (including a named dataset whose "
                            "access terms are not stated)",
        "weights_release": "weights_release: yes (trained weights or checkpoints stated as released), no, unclear; if the "
                           "excerpt says nothing, unclear when truncated is yes and no when truncated is no"}


def read(role, fields, labels):
    out = {}
    for f in sorted(CODES.glob(f"p9_{role}_batch_*.csv")):
        with f.open(newline="", encoding="utf-8") as fh:
            for r in csv.DictReader(fh):
                v = {k: (r.get(k) or "").strip() for k in fields}
                if any(v[k] not in labels[k] for k in fields):
                    raise SystemExit(f"{f.name}: invalid label for {r['item_id']}: {v}")
                out[r["item_id"]] = v
    return out


def coded():
    return (read("ext1", FIELDS, LABELS), read("ext2", FIELDS, LABELS),
            read("refx1", REF_FIELDS, REF_LABELS), read("refx2", REF_FIELDS, REF_LABELS))


def score() -> dict:
    keys = json.loads(KEYS.read_text(encoding="utf-8"))
    e1, e2, r1, r2 = coded()
    se, sr = sorted(set(e1) & set(e2)), sorted(set(r1) & set(r2))
    res = dict(drawn=len(keys), with_fulltext=sum(v["fulltext"] for v in keys.values()), coded_ext=len(se), coded_ref=len(sr),
               fulltext_kappa={f: kappa([e1[i][f] for i in se], [e2[i][f] for i in se], LABELS[f]) for f in FT},
               reference_kappa={f: kappa([r1[i][f] for i in sr], [r2[i][f] for i in sr], REF_LABELS[f]) for f in REF_FIELDS})
    res["fields_adjudicated"] = [f for f in FT if (res["fulltext_kappa"][f]["kappa"] or 0) < 0.80]
    print(json.dumps(res, indent=1))
    return res


def build_adj() -> None:
    sc = score()
    keys = json.loads(KEYS.read_text(encoding="utf-8"))
    e1, e2, r1, r2 = coded()
    ex = excerpts()
    ref_text = {}
    for f in sorted(SHEETS.glob("p9_refx1_batch_*.csv")):
        with f.open(newline="", encoding="utf-8") as fh:
            ref_text.update({r["item_id"]: (r["truncated"], r["evidence"]) for r in csv.DictReader(fh)})
    todo = [(i, f, e1[i][f], e2[i][f]) for i in sorted(e1) for f in sc["fields_adjudicated"] if e1[i][f] != e2[i][f]]
    todo += [(i, f, r1[i][f], r2[i][f]) for i in sorted(r1) for f in REF_FIELDS if r1[i][f] != r2[i][f]]
    rng = random.Random(20261023)
    ADJ_SHEETS.mkdir(parents=True, exist_ok=True)
    batches = []
    for n, start in enumerate(range(0, len(todo), 20)):
        chunk = todo[start:start + 20]
        with (ADJ_SHEETS / f"p9_extadj_batch_{n:03d}.csv").open("w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, ["item_id", "field", "definition", "label_a", "label_b", "title", "truncated", "text"])
            w.writeheader()
            for i, f, x, y in chunk:
                a, b = (x, y) if rng.random() < 0.5 else (y, x)
                if f in REF_FIELDS:
                    trunc, text = ref_text[i]
                else:
                    r = ex[keys[i]["key"]]
                    trunc, text = ("yes" if r["truncated"] else "no"), r["excerpt"]
                w.writerow(dict(item_id=i, field=f, definition=DEFS[f], label_a=a, label_b=b, title=keys[i]["title"],
                                truncated=trunc, text=text))
        batches.append(dict(batch=n, items=[[i, f] for i, f, _, _ in chunk]))
    ADJ_BATCHES.write_text(json.dumps(batches), encoding="utf-8")
    print(len(todo), "disagreements in", len(batches), "batches")


def merge() -> None:
    cats = {**LABELS, **REF_LABELS}
    rows = []
    for b in json.loads(ADJ_BATCHES.read_text(encoding="utf-8")):
        with (CODES / f"p9_extadj_batch_{b['batch']:03d}.csv").open(newline="", encoding="utf-8") as fh:
            got = {(r["item_id"], r["field"]): r for r in csv.DictReader(fh)}
        for i, f in b["items"]:
            r = got.get((i, f))
            if r is None or r["label"].strip() not in cats[f] or not r.get("reason", "").strip():
                raise SystemExit(f"batch {b['batch']}: bad or missing {i} {f}")
            rows.append(dict(item_id=i, field=f, label=r["label"].strip(), reason=" ".join(r["reason"].split())))
    with ADJ.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, ["item_id", "field", "label", "reason"])
        w.writeheader()
        w.writerows(rows)
    print(len(rows), "adjudications merged")


def run() -> dict:
    sc = score()
    e1, e2, r1, r2 = coded()
    adj = {}
    if ADJ.exists():
        with ADJ.open(newline="", encoding="utf-8") as fh:
            adj = {(r["item_id"], r["field"]): r["label"] for r in csv.DictReader(fh)}

    def pick(a, b, i, f):
        if a[i][f] == b[i][f]:
            return a[i][f]
        if (i, f) in adj:
            return adj[(i, f)]
        raise SystemExit(f"not adjudicated: {i} {f}")
    ft = {i: {f: (pick(e1, e2, i, f) if f in sc["fields_adjudicated"] else e1[i][f]) for f in FT} for i in e1}
    ref = {i: {f: pick(r1, r2, i, f) for f in REF_FIELDS} for i in r1}

    def tables(ftc, refc):
        out = {}
        for fld, pos, rf in PAIRS:
            pairs = [(ftc[i][fld] == pos, refc[i][rf] == "yes") for i in sorted(set(ftc) & set(refc))
                     if ftc[i][fld] != "unclear" and refc[i][rf] != "unclear"]
            out[f"{fld}_vs_{rf}"] = binary_agreement(pairs)
        return out
    frame_ft, frame_ref = final_codes(), reference_codes()
    frame_ft = {i: frame_ft[i] for i in frame_ref}
    res = dict(score=sc, extension=tables(ft, ref), frame=tables(frame_ft, frame_ref),
               pooled=tables({**frame_ft, **ft}, {**frame_ref, **ref}))
    res["claim7_weakened"] = any((res["pooled"][k]["kappa"] or 0) >= 0.40
                                 for k in ("code_vs_code_used", "main_data_access_vs_data_obtained"))
    (HERE / "results" / "p9_ext.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    return res


if __name__ == "__main__":
    {"score": score, "build_adj": build_adj, "merge": merge,
     "run": lambda: print(json.dumps(run(), indent=1)[:3000])}[sys.argv[1]]()
