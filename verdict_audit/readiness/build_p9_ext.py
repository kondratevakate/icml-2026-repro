"""Protocol 9, Note 9-17: extension sample of 300 attempted ICML 2026 papers outside the frame, for Q9.4.

`frame`  draws the sample once and writes results/p9_ext_frame.json (+ SHA-256 in p9_ext_frame_manifest.json).
`sheets` (after fetch_p9.py run --frame and build_p9_excerpts.py) writes full-text sheets for ext1/ext2 and
         logbook sheets for refx1/refx2 under data/p9/ (git-ignored), item ids E0000...

Usage: python build_p9_ext.py frame | sheets
"""
from __future__ import annotations

import csv
import hashlib
import json
import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import p8_common as P8  # noqa: E402
from build_p9_reference import evidence  # noqa: E402
from build_p9_sheets import BATCH, P9, excerpts  # noqa: E402
from fetch_p9 import key_of  # noqa: E402
from posthoc_p9_adversarial import icml_units  # noqa: E402

SEED, N = 20261023, 300
FRAME = HERE / "results" / "p9_ext_frame.json"
KEYS = P9 / "ext_keys.json"


def frame() -> None:
    if FRAME.exists():
        raise SystemExit("extension frame exists; drawn once")
    lab, cv = icml_units(), P8.claim_verdicts()
    inframe = {p["forum_id"] for p in json.loads((HERE / "results" / "p9_frame.json").read_text(encoding="utf-8"))
               if p["stratum"] == "icml-2026"}
    pool = sorted(o for o in lab if o in cv and o not in inframe)
    titles = {u["forum_id"]: u["title"] for u in json.loads((P8.DATA / "papers.json").read_text(encoding="utf-8"))
              if u["stratum"] == "icml-2026"}
    pick = sorted(random.Random(SEED).sample(pool, N))
    out = [dict(stratum="icml-2026", forum_id=o, group="ext", title=titles[o]) for o in pick]
    FRAME.write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")
    sha = hashlib.sha256(FRAME.read_bytes()).hexdigest()
    (HERE / "results" / "p9_ext_frame_manifest.json").write_text(json.dumps(dict(
        seed=SEED, pool=len(pool), drawn=N, any_medical=any(lab[o] in ("clinical", "life_science") for o in pick),
        frame_sha256=sha), indent=1), encoding="utf-8")
    print(len(pool), "in pool;", N, "drawn; sha", sha[:12])


def sheets() -> None:
    fr = json.loads(FRAME.read_text(encoding="utf-8"))
    ex, cv = excerpts(), P8.claim_verdicts()
    order = sorted(fr, key=lambda p: p["forum_id"])
    random.Random(SEED + 1).shuffle(order)
    keys = {f"E{n:04d}": dict(key=key_of(p), forum_id=p["forum_id"], title=p["title"], fulltext=key_of(p) in ex)
            for n, p in enumerate(order)}
    KEYS.write_text(json.dumps(keys, ensure_ascii=False), encoding="utf-8")
    ft = sorted(i for i, v in keys.items() if v["fulltext"])
    out = P9 / "ext_sheets"
    out.mkdir(parents=True, exist_ok=True)
    for b, start in enumerate(range(0, len(ft), BATCH)):
        chunk = ft[start:start + BATCH]
        for role in ("ext1", "ext2"):
            with (out / f"p9_{role}_batch_{b:03d}.csv").open("w", newline="", encoding="utf-8") as fh:
                w = csv.DictWriter(fh, ["item_id", "title", "truncated", "excerpt"])
                w.writeheader()
                for i in chunk:
                    r = ex[keys[i]["key"]]
                    w.writerow(dict(item_id=i, title=keys[i]["title"], truncated="yes" if r["truncated"] else "no",
                                    excerpt=r["excerpt"]))
        for role in ("refx1", "refx2"):
            with (out / f"p9_{role}_batch_{b:03d}.csv").open("w", newline="", encoding="utf-8") as fh:
                w = csv.DictWriter(fh, ["item_id", "title", "truncated", "evidence"])
                w.writeheader()
                for i in chunk:
                    text, cut = evidence(cv[keys[i]["forum_id"]])
                    w.writerow(dict(item_id=i, title=keys[i]["title"], truncated="yes" if cut else "no", evidence=text))
    print(len(keys), "drawn;", len(ft), "with full text;", -(-len(ft) // BATCH), "batches per role")


if __name__ == "__main__":
    {"frame": frame, "sheets": sheets}[sys.argv[1]]()
