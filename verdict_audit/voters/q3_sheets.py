"""Build blind coding sheets for Q3 validation (PROTOCOL.md, Q3). Sheets and keys stay in data/ (git-ignored)."""
from __future__ import annotations

import csv
import json
import re

import numpy as np

import common as C
from patterns import COMPILED

OUT = C.DATA / "q3"
N_A_EACH, N_B, HALF = 60, 40, 200


def logbook_full_text(pages: dict[str, str]) -> str:
    try:
        title = json.loads(pages.get("logbook.json", "{}")).get("title", "")
    except json.JSONDecodeError:
        title = ""
    return "\n".join([title] + [pages[p] for p in sorted(pages) if p.startswith("pages/")])


def mask(text: str, space_id: str) -> str:
    ns, name = space_id.split("/", 1)
    text = re.sub(r"huggingface\.co/spaces/\S+", "[SPACE]", text, flags=re.IGNORECASE)
    for token in (space_id, name, ns):
        text = re.sub(re.escape(token), "[TEAM]", text, flags=re.IGNORECASE)
    return text


def summary_text(pages: dict[str, str]) -> str:
    parts = [pages.get("pages/index.md", "")]
    parts += [pages[p] for p in sorted(pages) if "executive-summary" in p.lower()]
    return "\n\n".join(x for x in parts if x)[:6000]


def main() -> None:
    manifest = json.loads((C.DATA / "pages_manifest.json").read_text(encoding="utf-8"))
    ok = sorted(s for s, e in manifest.items() if e.get("status") == "ok")
    matches = {"tool": [], "model": []}
    unmatched = []
    for s in ok:
        text = logbook_full_text(C.read_pages(s, manifest[s]))
        found = False
        for (cat, name), rx in COMPILED.items():
            for m in rx.finditer(text):
                matches[cat].append((s, name, m.start(), m.end()))
                found = True
        if not found:
            unmatched.append(s)
    rng = np.random.default_rng(C.A.SEED)
    OUT.mkdir(parents=True, exist_ok=False)
    rows_a, key_a = [], []
    for cat in ("tool", "model"):
        pool = matches[cat]
        pick = rng.choice(len(pool), size=min(N_A_EACH, len(pool)), replace=False)
        for i in pick:
            s, name, a, b = pool[i]
            text = logbook_full_text(C.read_pages(s, manifest[s]))
            lo, hi = max(0, a - HALF), min(len(text), b + HALF)
            window = text[lo:a] + "<<" + text[a:b] + ">>" + text[b:hi]
            key_a.append(dict(space_id=s, category=cat, pattern=name))
            rows_a.append(dict(window=mask(window, s)))
    order = rng.permutation(len(rows_a))  # interleave categories so coders cannot infer them
    with (OUT / "sheet_A.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, ["item_id", "window"])
        w.writeheader()
        for new_id, i in enumerate(order):
            w.writerow(dict(item_id=f"A{new_id:03d}", window=rows_a[i]["window"]))
    with (OUT / "key_A.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, ["item_id", "space_id", "category", "pattern"])
        w.writeheader()
        for new_id, i in enumerate(order):
            w.writerow(dict(item_id=f"A{new_id:03d}", **key_a[i]))
    pick_b = rng.choice(len(unmatched), size=min(N_B, len(unmatched)), replace=False)
    with (OUT / "sheet_B.csv").open("w", newline="", encoding="utf-8") as fh, \
         (OUT / "key_B.csv").open("w", newline="", encoding="utf-8") as fk:
        w, wk = csv.DictWriter(fh, ["item_id", "text"]), csv.DictWriter(fk, ["item_id", "space_id"])
        w.writeheader(); wk.writeheader()
        for n, i in enumerate(pick_b):
            s = unmatched[i]
            w.writerow(dict(item_id=f"B{n:03d}", text=mask(summary_text(C.read_pages(s, manifest[s])), s)))
            wk.writerow(dict(item_id=f"B{n:03d}", space_id=s))
    counts = dict(n_logbooks=len(ok), tool_matches=len(matches["tool"]), model_matches=len(matches["model"]),
                  unmatched_logbooks=len(unmatched), sample_A=len(rows_a), sample_B=len(pick_b))
    (OUT / "counts.json").write_text(json.dumps(counts, indent=2), encoding="utf-8")
    print(counts)


if __name__ == "__main__":
    main()
