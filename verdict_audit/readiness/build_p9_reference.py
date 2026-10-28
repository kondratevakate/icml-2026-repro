"""Protocol 9, Q9.4 reference standard (Note 9-7): one sheet row per ICML 2026 frame paper with a logbook.

Each row holds the logbooks' claims, the judge's verdicts and evidence text, capped at 15,000 characters.
Sheets go to data/p9/ref_sheets/ (git-ignored); both coders (ref1, ref2) get the same batches of 20.

Usage: python build_p9_reference.py
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import p8_common as P8  # noqa: E402
from build_p9_sheets import BATCH, KEYS, P9  # noqa: E402

CAP = 15000
SHEETS = P9 / "ref_sheets"
BATCHES = P9 / "ref_batches.json"


def evidence(claims: list[dict]) -> tuple[str, bool]:
    parts, spaces = [], []
    for c in claims:
        if c["space"] not in spaces:
            spaces.append(c["space"])
    for n, s in enumerate(spaces, 1):
        parts.append(f"[Logbook {n}]")
        for c in (c for c in claims if c["space"] == s):
            parts.append(f"Claim: {c['raw']}\nVerdict: {c['verdict']}\nJudge's evidence: {c['evidence']}")
    text = "\n".join(parts)
    return text[:CAP], len(text) > CAP


def main() -> None:
    cv = P8.claim_verdicts()
    keys = json.loads(KEYS.read_text(encoding="utf-8"))
    items = sorted(i for i, v in keys.items() if v["stratum"] == "icml-2026" and v["forum_id"] in cv)
    SHEETS.mkdir(parents=True, exist_ok=True)
    batches = []
    for n, start in enumerate(range(0, len(items), BATCH)):
        chunk = items[start:start + BATCH]
        for role in ("ref1", "ref2"):
            with (SHEETS / f"p9_{role}_batch_{n:03d}.csv").open("w", newline="", encoding="utf-8") as fh:
                w = csv.DictWriter(fh, ["item_id", "title", "truncated", "evidence"])
                w.writeheader()
                for i in chunk:
                    text, cut = evidence(cv[keys[i]["forum_id"]])
                    w.writerow(dict(item_id=i, title=keys[i]["title"], truncated="yes" if cut else "no", evidence=text))
        batches.append(dict(batch=n, items=chunk))
    BATCHES.write_text(json.dumps(batches), encoding="utf-8")
    print(len(items), "papers;", len(batches), "batches per coder")


if __name__ == "__main__":
    main()
