"""Protocol 9, Note 9-15: sheets of distinct claim texts for two blind coders (claim text only).

Units: claims rated in logbooks of all 110 attempted medical ICML 2026 papers and a random 110 attempted other papers
(seed 20261023). Random item ids; keys in data/p9/claim_keys.json (git-ignored).

Usage: python build_p9_claims.py
"""
from __future__ import annotations

import csv
import json
import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import p8_common as P8  # noqa: E402
from build_p9_sheets import P9  # noqa: E402
from posthoc_p9_adversarial import icml_units  # noqa: E402
from posthoc_p9_areas import LIFE  # noqa: E402

SEED, BATCH = 20261023, 50
KEYS = P9 / "claim_keys.json"
SHEETS = P9 / "claim_sheets"


def main() -> None:
    if KEYS.exists():
        raise SystemExit("claim_keys.json exists; units are drawn once")
    lab, cv = icml_units(), P8.claim_verdicts()
    att = sorted(o for o in lab if o in cv)
    med = [o for o in att if lab[o] in LIFE]
    oth = random.Random(SEED).sample([o for o in att if lab[o] not in LIFE], len(med))
    units = []
    for o in sorted(med + oth):
        texts = {}
        for c in cv[o]:
            texts.setdefault(c["text"], c["raw"])
        units += [(o, t, raw) for t, raw in sorted(texts.items())]
    order = list(range(len(units)))
    random.Random(SEED).shuffle(order)
    keys = {f"C{n:04d}": dict(orid=units[i][0], text=units[i][1], group="medical" if lab[units[i][0]] in LIFE else "other")
            for n, i in enumerate(order)}
    raw = {f"C{n:04d}": units[i][2] for n, i in enumerate(order)}
    KEYS.write_text(json.dumps(keys, ensure_ascii=False), encoding="utf-8")
    SHEETS.mkdir(parents=True, exist_ok=True)
    items = sorted(keys)
    for b, start in enumerate(range(0, len(items), BATCH)):
        for role in ("claim1", "claim2"):
            with (SHEETS / f"p9_{role}_batch_{b:03d}.csv").open("w", newline="", encoding="utf-8") as fh:
                w = csv.DictWriter(fh, ["item_id", "claim"])
                w.writeheader()
                for i in items[start:start + BATCH]:
                    w.writerow(dict(item_id=i, claim=raw[i]))
    print(len(keys), "claims;", sum(v["group"] == "medical" for v in keys.values()), "medical;",
          -(-len(items) // BATCH), "batches per coder")


if __name__ == "__main__":
    main()
