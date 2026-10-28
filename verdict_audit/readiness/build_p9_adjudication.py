"""Protocol 9 adjudication (Notes 9-6, 9-7): sheets of coder disagreements for adjudicator subagents, and merging.

`build` adds every pending disagreement not yet in a sheet to new batches of 20 (rows sorted by item, so a
paper's fields sit together). The two labels appear in random order (seed 20261019) as label_a and label_b, with
the field definition and the text the coders saw (full-text excerpt, or logbook evidence for reference fields).
`merge` collects the adjudicators' files into codes/p9/adjudication.csv and checks them.

Usage: python build_p9_adjudication.py build | merge
"""
from __future__ import annotations

import csv
import json
import random
import sys
from pathlib import Path

from analyze_p9 import ADJ, REF_FIELDS, REF_LABELS, pending_items
from build_p9_sheets import BATCH, CODES, KEYS, LABELS, P9, excerpts
from build_p9_reference import SHEETS as REF_SHEETS

HERE = Path(__file__).resolve().parent
SHEETS = P9 / "adj_sheets"
BATCHES = P9 / "adj_batches.json"
SEED = 20261019
DEFINITION = {
    "variance": "variance: yes (several seeds, error bars, standard deviations or intervals for the main results), no, unclear",
    "compute": "compute: amount (hardware and time or GPU-hours stated), hardware_only (hardware stated without time), none, unclear",
    "code_used": "code_used: yes (at least one logbook obtained the authors' code and ran it), no (the evidence shows the "
                 "authors' code was not obtained or not run, for example the method was reimplemented or the code was "
                 "unavailable), unclear",
    "data_obtained": "data_obtained: yes (at least one logbook obtained the paper's main evaluation data), no (the evidence "
                     "shows the main data was not obtained, for example it was unavailable, substituted, or replaced by "
                     "synthetic or toy data), unclear",
    "weights_used": "weights_used: yes (at least one logbook obtained and used the authors' trained weights or checkpoints), "
                    "no (the evidence shows the authors' weights were not used, for example the model was retrained or "
                    "weights were unavailable), unclear",
}


def ref_texts() -> dict:
    out = {}
    for f in sorted(REF_SHEETS.glob("p9_ref1_batch_*.csv")):
        with f.open(newline="", encoding="utf-8") as fh:
            for r in csv.DictReader(fh):
                out[r["item_id"]] = (r["truncated"], r["evidence"])
    return out


def build() -> None:
    batches = json.loads(BATCHES.read_text(encoding="utf-8")) if BATCHES.exists() else []
    done = {(i, f) for b in batches for i, f in b["items"]}
    todo = [p for p in pending_items() if (p[0], p[1]) not in done]
    if not todo:
        print("nothing new to adjudicate")
        return
    keys = json.loads(KEYS.read_text(encoding="utf-8"))
    ex, refs = excerpts(), ref_texts()
    rng = random.Random(SEED + len(batches))
    SHEETS.mkdir(parents=True, exist_ok=True)
    new = []
    for start in range(0, len(todo), BATCH):
        chunk = todo[start:start + BATCH]
        n = len(batches)
        with (SHEETS / f"p9_adj_batch_{n:03d}.csv").open("w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, ["item_id", "field", "definition", "label_a", "label_b", "title", "truncated", "text"])
            w.writeheader()
            for i, f, x, y in chunk:
                a, b = (x, y) if rng.random() < 0.5 else (y, x)
                if f in REF_FIELDS:
                    trunc, text = refs[i]
                else:
                    r = ex[keys[i]["key"]]
                    trunc, text = ("yes" if r["truncated"] else "no"), r["excerpt"]
                w.writerow(dict(item_id=i, field=f, definition=DEFINITION[f], label_a=a, label_b=b,
                                title=keys[i]["title"], truncated=trunc, text=text))
        batches.append(dict(batch=n, items=[[i, f] for i, f, _, _ in chunk]))
        new.append(n)
    BATCHES.write_text(json.dumps(batches), encoding="utf-8")
    print("new adjudication batches:", new[0], "to", new[-1], "|", len(todo), "items")


def merge() -> None:
    batches = json.loads(BATCHES.read_text(encoding="utf-8"))
    cats = {**LABELS, **REF_LABELS}
    rows, missing = [], []
    for b in batches:
        path = CODES / f"p9_adj_batch_{b['batch']:03d}.csv"
        if not path.exists():
            missing.append(b["batch"])
            continue
        with path.open(newline="", encoding="utf-8") as fh:
            got = {(r["item_id"], r["field"]): r for r in csv.DictReader(fh)}
        for i, f in b["items"]:
            r = got.get((i, f))
            if r is None or r["label"].strip() not in cats[f] or not r.get("reason", "").strip():
                raise SystemExit(f"adjudication batch {b['batch']}: bad or missing row {i} {f}")
            rows.append(dict(item_id=i, field=f, label=r["label"].strip(), reason=" ".join(r["reason"].split())))
    with ADJ.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, ["item_id", "field", "label", "reason"])
        w.writeheader()
        w.writerows(rows)
    print(len(rows), "adjudications merged; batches without a file:", missing)


if __name__ == "__main__":
    {"build": build, "merge": merge}[sys.argv[1]]()
