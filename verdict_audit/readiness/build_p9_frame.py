"""Protocol 9 frame: all life-science papers (protocol 7 label) at the nine 2025-2026 venue-years and an
equal-size random sample of non-candidates per venue-year (seed 20261019).

Usage: python build_p9_frame.py   -> results/p9_frame.json, results/p9_frame_manifest.json (refuses to overwrite)
"""
from __future__ import annotations

import csv
import hashlib
import json
import random
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
STRATA = ("cvpr-2025", "iccv-2025", "eccv-2026", "iclr-2025", "iclr-2026", "icml-2025", "icml-2026",
          "neurips-2025", "neurips-2026")
LIFE = ("clinical", "life_science")
SEED = 20261019
OUT = HERE / "results" / "p9_frame.json"


def build() -> list[dict]:
    labels = {}
    with (HERE / "codes" / "labels.csv").open(newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            if r["kind"] == "cand":
                labels[r["unit"]] = r["label"]
    papers = [u for u in json.loads((HERE / "data" / "papers.json").read_text(encoding="utf-8")) if u["stratum"] in STRATA]
    rng = random.Random(SEED)
    frame = []
    for s in STRATA:
        units = sorted((u for u in papers if u["stratum"] == s), key=lambda u: u["forum_id"])
        life = [u for u in units if u["candidate"] and labels.get(f"{s}:{u['forum_id']}") in LIFE]
        pool = [u for u in units if not u["candidate"]]
        other = rng.sample(pool, len(life))
        for g, lst in (("life_science", life), ("other", sorted(other, key=lambda u: u["forum_id"]))):
            frame += [dict(stratum=s, forum_id=u["forum_id"], group=g, title=u["title"],
                           label=labels.get(f"{s}:{u['forum_id']}") if g == "life_science" else None) for u in lst]
    return frame


def main() -> None:
    if OUT.exists():
        raise SystemExit(f"{OUT.name} exists; the frame is drawn once")
    frame = build()
    data = json.dumps(frame, indent=1, ensure_ascii=False)
    OUT.write_text(data, encoding="utf-8")
    sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    counts = Counter((f["stratum"], f["group"]) for f in frame)
    (HERE / "results" / "p9_frame_manifest.json").write_text(json.dumps(dict(
        seed=SEED, strata=STRATA, n=len(frame), counts={f"{s}:{g}": n for (s, g), n in sorted(counts.items())},
        frame_sha256=sha(OUT), papers_json_sha256=sha(HERE / "data" / "papers.json"),
        labels_csv_sha256=sha(HERE / "codes" / "labels.csv"), code_sha256=sha(Path(__file__))), indent=1), encoding="utf-8")
    print(len(frame), Counter(f["group"] for f in frame))


if __name__ == "__main__":
    main()
